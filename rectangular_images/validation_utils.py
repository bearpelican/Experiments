import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler
import sys, os, shutil, time, warnings
from datetime import datetime
import pickle
from tqdm import tqdm
import numpy as np
import torch

# Step 1: sort images by aspect ratio
def sort_ar(data, valdir):
    idx2ar_file = data/'sorted_idxar.p'
    if os.path.isfile(idx2ar_file): return pickle.load(open(idx2ar_file, 'rb'))
    print('Creating AR indexes. Please be patient this may take a couple minutes...')
    val_dataset = datasets.ImageFolder(valdir)
    sizes = [img[0].size for img in tqdm(val_dataset, total=len(val_dataset))]
    idx_ar = [(i, round(s[0]/s[1], 5)) for i,s in enumerate(sizes)]
    sorted_idxar = sorted(idx_ar, key=lambda x: x[1])
    pickle.dump(sorted_idxar, open(idx2ar_file, 'wb'))
    return sorted_idxar

# Step 2: chunk images by batch size. This way we can crop each image to the batch aspect ratio mean 
def chunks(l, n):
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))

# Step 3: map image index to batch aspect ratio mean so our transform function knows where to crop
def map_idx2ar(idx_ar_sorted, batch_size):
    ar_chunks = list(chunks(idx_ar_sorted, batch_size))
    idx2ar = {}
    ar_means = []
    for chunk in ar_chunks:
        idxs, ars = list(zip(*chunk))
        mean = round(np.mean(ars), 5)
        ar_means.append(mean)
        for idx in idxs:
            idx2ar[idx] = mean
    return idx2ar, ar_means

class ValDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(root, transform, target_transform)
    def __getitem__(self, index):
        path, target = self.imgs[index]
        sample = self.loader(path)
        if self.transform is not None:
            for tfm in self.transform:
                if isinstance(tfm, RectangularCropTfm): sample = tfm(sample, index)
                else: sample = tfm(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    
# Essentially a sequential sampler
class SequentialIndexSampler(Sampler):
    def __init__(self, indices): self.indices = indices
    def __len__(self): return len(self.indices)
    def __iter__(self): return iter(self.indices)
    
class RectangularCropTfm(object):
    def __init__(self, idx2ar, target_size):
        self.idx2ar, self.target_size = idx2ar, target_size
    def __call__(self, img, idx):
        target_ar = self.idx2ar[idx]
        if target_ar < 1: 
            w = int(self.target_size/target_ar)
            size = (w//8*8, self.target_size)
        else: 
            h = int(self.target_size*target_ar)
            size = (self.target_size, h//8*8)
        return transforms.functional.center_crop(img, size)
    
def validate(val_loader, model, criterion, fp16=False, aug_loader=None, num_augmentations=0):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    start_time = datetime.now()

    model.eval()
    end = time.time()

    val_iter = iter(val_loader)
    aug_iters = [iter(aug_loader) for i in range(num_augmentations)]
    prec5_arr = []
    for i in range(len(val_loader)):
        def get_output(dl_iter):
            input,target = next(dl_iter)
            input, target = input.cuda(), target.cuda()
            if fp16: input = input.half()

            # compute output
            with torch.no_grad():
                output = model(input)
                loss = criterion(output, target)
            return output, loss, input, target
        
        # Normal Validation
        output,loss,input,target = get_output(val_iter)
        
        # TTA
        for aug_iter in aug_iters:
            o,l,_,_ = get_output(aug_iter)
            output.add_(o)
            loss.add_(l)
        loss.div_(num_augmentations+1)
        
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        reduced_loss = loss.data
            
        losses.update(to_python_float(reduced_loss), input.size(0))
        top1.update(to_python_float(prec1), input.size(0))
        top5.update(to_python_float(prec5), input.size(0))
        prec5_arr.append(to_python_float(prec5))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if ((i+1)%100 == 0) or ((i+1) == len(val_loader)):
            output = ('Test: [{0}/{1}]\t' \
                    + 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                    + 'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                    + 'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t' \
                    + 'Prec@5 {top5.val:.3f} ({top5.avg:.3f})').format(
                    i+1, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5)
            print(output)

    time_diff = datetime.now()-start_time
    print(f'Total Time:{float(time_diff.total_seconds() / 3600.0)}\t Top 5 Accuracy: {top5.avg:.3f}\n')
    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return prec5_arr

# item() is a recent addition, so this helps with backward compatibility.
def to_python_float(t):
    if hasattr(t, 'item'): return t.item()
    else: return t[0]
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self): self.reset()
    def reset(self): self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res