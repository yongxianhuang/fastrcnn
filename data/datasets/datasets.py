from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms, utils
import os
from skimage import io, transform
import numpy as np


class RcnnDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, roidb_path, transform=None):
        """
        Args:
            roidb_path (str): read from pickle
        """
        super(RcnnDataset, self).__init__(roidb_path, transform=transform)

        if os.path.exists(roidb_path):
            raise RuntimeError("roidb is null")
        with open('/opt/ml/fastrcnn/data/cache/train.npz', 'rb') as fid:
            npz = np.load(fid)
            self.train_imgs = npz['train_imgs']
            self.train_img_info = npz['train_img_info']
            self.train_roi = npz['train_roi']
            self.train_cls = npz['train_cls']
            self.train_tbbox = npz['train_tbbox']
        self.transform = transform

    def __len__(self):
        length = self.train_imgs.shape[0]
        return 0 if length is None else length

    def __getitem__(self, idx):
        image = self.train_imgs[idx].astype(np.uint8)  # (3, 224, 224)
        # image = np.transpose(image, (1, 2, 0)).astype(np.uint8)

        '''transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])'''
        if self.transform is not None:
            image = self.transform(image)

        gt_classes = np.asarray(np.where(self.roidb[idx]['gt_classes'] > 0)).reshape(-1)
        rois = self.roidb[idx]['boxes'][gt_classes].astype('float')
        sample = {'image': image, 'rois': rois.reshape(-1, 4), 'gt_classes': gt_classes}
        if 0 and self.transform:
            image, boxes, gt_classes = sample['image'], sample['rois'], sample['gt_classes']
            sample = {
                'image': self.transform(image),
                'rois': boxes,
                'gt_classes': gt_classes
            }
            '''to be completed '''
        return sample

    def _vis(self, datasets):
        import matplotlib.pyplot as plt
        for i in range(100):
            sample = datasets[i]
            print(i, sample['image'].shape, sample['gt_classes'].shape)
            ax = plt.subplot(1, 4, i + 1)
            plt.tight_layout()
            ax.set_title('Sample #{}'.format(i))
            ax.axis('off')
            plt.imshow(sample['image'])
            plt.pause(0.001)  # pause a bit so that plots are updated
            if i == 3:
                plt.show()
                break
