import torchvision.transforms as transforms
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import torch.nn.functional as F

def image_loader(image_name):
    loader = transforms.Compose([
        # transforms.Resize(imsize),  # scale imported image
        transforms.ToTensor()])  # transform it into a torch tensor
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image


def roi_pooling_2d_pytorch(input, rois, output_size=(6, 6), spatial_scale=1.0):
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


class SlowROIPool(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(output_size)
        # self.size = output_size

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
            print(
                f'the i\s loop is : {i}, roi_idx[i] is {roi_idx[i]}, images[roi_idx[i]].shape is {images[roi_idx[i]].shape}')
            img = images[roi_idx[i]].unsqueeze(0)
            img = img[:, :, y1[i]:y2[i], x1[i]:x2[i]]
            # img = self.maxpool(img)
            # img = nn.AdaptiveMaxPool2d((6, 6))(img)
            img = roi_pooling_2d_pytorch(img)
            res.append(img)
        res = torch.cat(res, dim=0)
        return res


if __name__ == '__main__':
    img = image_loader('test.jpg')
    features = nn.Sequential(
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
    out = features(img)
    slroi = SlowROIPool((6, 6))
    ridx = np.array([0, 1, 2, 3, 4], dtype=np.int)
    rois = np.asarray(
        [[262, 210, 323, 338],
         [164, 263, 252, 371],
         [4, 243, 66, 373],
         [240, 193, 294, 298],
         [276, 185, 311, 219]], dtype=np.float)
    out = roi_pooling_2d_pytorch(out, torch.from_numpy(rois))
    # out = slroi(out, rois, ridx)
    print(out.shape)
    import torchvision
    ds=torchvision.datasets.CIFAR10()
