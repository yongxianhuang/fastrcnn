# coding=utf-8

from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
from scipy.io import loadmat
import matplotlib.pyplot as plt
import pylab
import os
import tensorflow as tf
import tensorflow_datasets as tfds
import cv2

pylab.rcParams['figure.figsize'] = (8.0, 10.0)
dataDir = '/work/ml/stanford-dogs-dataset'
dataDir = 'C:\\Users\\yuan\\Desktop\\dogs-datasets\\'
dataType = 'train2017'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
annFile = os.path.join(dataDir, 'Annotation', 'n02087394-Rhodesian_ridgeback', 'n02087394_36')
# initialize COCO api for instance annotations
# coco=COCO(annFile)
dogsdict = loadmat(os.path.join(dataDir, 'train_list.mat'))
data, info = tfds.load("mnist", with_info=True)