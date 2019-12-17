# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import torch
from .alexnet import RCNN_ALEXNET
import os
import torch.nn as nn


def build_model(cfg):
    model = RCNN_ALEXNET()
    model.load_state_dict(torch.load(os.path.expanduser('/opt/ml/fastrcnn/model/hao123.mdl')))
    # if torch.cuda.device_count() > 1:
    #     print(f"Let's use {torch.cuda.device_count()} GPUs!")
    #     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    #     model = nn.DataParallel(model)
    if cfg.MODEL.DEVICE == 'cpu':
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # device1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        model.to(device)
        # model.to(device1)
    return model
