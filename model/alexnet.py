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


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class SlowROIPool(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(output_size)
        self.size = output_size

    def forward(self, images, rois, roi_idx):
        n = rois.shape[0]
        h = images.size(2)
        w = images.size(3)
        x1 = rois[:, 0]
        y1 = rois[:, 1]
        x2 = rois[:, 2]
        y2 = rois[:, 3]

        x1 = np.floor(x1 * w).astype(int)
        x2 = np.ceil(x2 * w).astype(int)
        y1 = np.floor(y1 * h).astype(int)
        y2 = np.ceil(y2 * h).astype(int)

        res = []
        for i in range(n):
            img = images[roi_idx[i]].unsqueeze(0)
            img = img[:, :, y1[i]:y2[i], x1[i]:x2[i]]
            img = self.maxpool(img)
            res.append(img)
        res = torch.cat(res, dim=0)
        return res


class RCNN_ALEXNET(nn.Module):

    def __init__(self):
        super(RCNN_ALEXNET, self).__init__()
        rawnet = AlexNet()
        rawnet.load_state_dict(torch.load(os.path.expanduser('~/.cache/torch/checkpoints/alexnet-owt-4df8aa71.pth')))
        self.seq = nn.Sequential(*list(rawnet.features.children())[:-1])
        self.roipool = SlowROIPool(output_size=(6, 6))
        # 006823.jpg: JPEG image data, JFIF standard 1.01, aspect ratio, density 1x1, segment length 16, baseline, precision 8, 500x375, frames 3
        self.feature = nn.Sequential(*list(rawnet.classifier.children())[:-1])
        _x = Variable(torch.Tensor(1, 3, 224, 224))
        _r = np.array([[0., 0., 1., 1.]])
        _ri = np.array([0])
        output = self.seq(_x)
        # print(f'output seq shape is {output.shape}')  #output seq shape is torch.Size([1, 256, 13, 13])
        output = self.roipool(output, _r, _ri)
        # print(f'output roipool shape is {output.shape}') #output roipool shape is torch.Size([1, 256, 7, 7])
        _x = self.feature(output.view(1, -1))  # size mismatch, m1: [1792 x 7], m2: [9216 x 4096]
        feature_dim = _x.size(1)
        self.cls_score = nn.Linear(feature_dim, N_CLASS + 1)
        self.bbox = nn.Linear(feature_dim, 4 * (N_CLASS + 1))
        self.cel = nn.CrossEntropyLoss()
        self.sl1 = nn.SmoothL1Loss()

    def forward(self, inp, rois, ridx):
        res = inp
        res = self.seq(res)
        res = self.roipool(res, rois, ridx)
        res = res.detach()
        res = res.view(res.size(0), -1)
        feat = self.feature(res)
        cls_score = self.cls_score(feat)
        bbox = self.bbox(feat).view(-1, N_CLASS + 1, 4)
        return cls_score, bbox

    def calc_loss(self, probs, bbox, labels, gt_bbox):
        loss_sc = self.cel(probs, labels)
        lbl = labels.view(-1, 1, 1).expand(labels.size(0), 1, 4)
        mask = (labels != 0).float().view(-1, 1).expand(labels.size(0), 4)
        loss_loc = self.sl1(bbox.gather(1, lbl).squeeze(1) * mask, gt_bbox * mask)
        lmb = 1.0
        loss = loss_sc + lmb * loss_loc
        return loss, loss_sc, loss_loc
