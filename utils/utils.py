import time, os
import numpy as np
from torch.autograd import Variable

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms

N_CLASS = 21 - 1


# minus 1 background
# voc 2007 classes len(pv._classes) == 21

class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def prep_im_for_blob(im, pixel_means, target_size, max_size):
    """Mean subtract and scale an image for use in a blob."""
    im = im.astype(np.float32, copy=False)
    im -= pixel_means
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)
    return im, im_scale


def roi_pooling_2d_pytorch(input, rois, output_size=(7, 7), spatial_scale=1.0):
    """Spatial Region of Interest (ROI) pooling function in pure pytorch/python
    This function acts similarly to `~roi_pooling_2d`, but performs a python
    loop over ROI. Note that this is not a direct replacement of
    `~roi_pooling_2d` (viceversa).
    See :function:`~roi_pooling_2d` for details and output shape.
    Args:
        output_size (int or tuple): the target output size of the image of the
            form H x W. Can be a tuple (H, W) or a single number H for a square
            image H x H.
        spatial_scale (float): scale of the rois if resized.
    """
    assert rois.dim() == 2
    assert rois.size(1) == 5
    output = []
    rois = rois.data.float()
    num_rois = rois.size(0)

    rois[:, 1:].mul_(spatial_scale)
    rois = rois.long()
    for i in range(num_rois):
        roi = rois[i]
        im_idx = roi[0]
        im = input.narrow(0, im_idx, 1)[...,
             roi[2]:(roi[4] + 1),
             roi[1]:(roi[3] + 1)]
        output.append(F.adaptive_max_pool2d(im, output_size))
    return torch.cat(output, 0)


if __name__ == '__main__':
    net = RCNN()
    print(net)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu

    loader = transforms.Compose([
        transforms.Resize(imsize),  # scale imported image
        transforms.ToTensor()])  # transform it into a torch tensor
    style_img = image_loader(os.path.expanduser("datasets/000005.jpg"))
    out = net(style_img).detach()
    import torchvision
    import torchvision.transforms as transforms

    Transform = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=Transform)

    trainloader = torch.utils.data.dataloader.DataLoader(trainset, batch_size=4)
    '''
    >>> sc, r_bbox = net(img.cpu(), rois, ridx)
    >>> sc
    tensor([[ -1.6817, -14.2437, -24.5490,  ..., -66.0772, -39.4640, -11.6070],
            [ -1.6161,   9.1441,   2.1071,  ..., -52.8161, -36.2271, -15.2764],
            [ 14.0493, -21.1037, -13.9566,  ...,  24.6168,   3.7548,   2.9498],
            ...,
            [ 16.2419,  -0.4113,  -2.0588,  ...,   8.9016,  15.8348, -10.5107],
            [ 13.2066,  -7.6195,  -8.9663,  ...,  15.4599,  15.1479, -19.2281],
            [ 33.1462,  -8.0503, -40.9404,  ...,  34.6084,  -4.2957,  -4.2131]],
           grad_fn=<AddmmBackward>)
    >>> sc.shape
    torch.Size([128, 21])
    >>> r_bbox.shape
    torch.Size([128, 21, 4])
    >>> '''
