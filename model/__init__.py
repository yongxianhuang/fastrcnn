# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import torch
from .alexnet import RCNN_ALEXNET


def build_model(cfg):
    model = RCNN_ALEXNET()
    if cfg.MODEL.DEVICE == 'cuda':
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device)
