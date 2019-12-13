from tqdm import trange
import argparse
from config.config import cfg, get_output_dir
import roi_data_layer.roidb as rdl_roidb
import sys, os
import pickle
import numpy as np
from PIL import Image
from data.datasets import pascal_voc

__sets = {}

'''/opt/ml/fastrcnn/data/cache
all.pkl
bbox_mean.npy
bbox_std.npy
voc_2007_trainval_gt_roidb.pkl
voc_2007_trainval_selective_search_roidb.pkl'''


def rel_bbox(size, bbox):
    bbox = bbox.astype(np.float32)
    bbox[:, 0] /= size[0]
    bbox[:, 1] /= size[1]
    bbox[:, 2] += 1
    bbox[:, 2] /= size[0]
    bbox[:, 3] += 1
    bbox[:, 3] /= size[1]
    return bbox


def calc_ious(ex_rois, gt_rois):
    ex_area = (1. + ex_rois[:, 2] - ex_rois[:, 0]) * (1. + ex_rois[:, 3] - ex_rois[:, 1])
    gt_area = (1. + gt_rois[:, 2] - gt_rois[:, 0]) * (1. + gt_rois[:, 3] - gt_rois[:, 1])
    area_sum = ex_area.reshape((-1, 1)) + gt_area.reshape((1, -1))

    lb = np.maximum(ex_rois[:, 0].reshape((-1, 1)), gt_rois[:, 0].reshape((1, -1)))
    rb = np.minimum(ex_rois[:, 2].reshape((-1, 1)), gt_rois[:, 2].reshape((1, -1)))
    tb = np.maximum(ex_rois[:, 1].reshape((-1, 1)), gt_rois[:, 1].reshape((1, -1)))
    ub = np.minimum(ex_rois[:, 3].reshape((-1, 1)), gt_rois[:, 3].reshape((1, -1)))

    width = np.maximum(1. + rb - lb, 0.)
    height = np.maximum(1. + ub - tb, 0.)
    area_i = width * height
    area_u = area_sum - area_i
    ious = area_i / area_u
    return ious


def bbox_transform(ex_rois, gt_rois):
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.array([targets_dx, targets_dy, targets_dw, targets_dh]).T
    return targets


def parse_args():
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on [voc_2007_trainval]',
                        default='voc_2007_trainval', type=str)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=40000, type=int)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.__contains__(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()


def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
        print('Appending horizontally-flipped training examples...')
        imdb.append_flipped_images()
        print('done')
    print('Preparing training data...')
    rdl_roidb.prepare_roidb(imdb)
    print('done')
    return imdb.roidb


def prepare_roidb(imdb):
    """Enrich the imdb's roidb by adding some derived quantities that
    are useful for training. This function precomputes the maximum
    overlap, taken over ground-truth boxes, between each ROI and
    each ground-truth box. The class with maximum overlap is also
    recorded.
    """
    roidb = imdb.roidb
    for i in range(len(imdb.image_index)):
        roidb[i]['image'] = imdb.image_path_at(i)
        # need gt_overlaps as a dense array for argmax
        gt_overlaps = roidb[i]['gt_overlaps'].toarray()
        # max overlap with gt over classes (columns)
        max_overlaps = gt_overlaps.max(axis=1)
        # gt class that had the max overlap
        max_classes = gt_overlaps.argmax(axis=1)
        roidb[i]['max_classes'] = max_classes
        roidb[i]['max_overlaps'] = max_overlaps
        # sanity checks
        # max overlap of 0 => class should be zero (background)
        zero_inds = np.where(max_overlaps == 0)[0]
        assert all(max_classes[zero_inds] == 0)
        # max overlap > 0 => class should not be zero (must be a fg class)
        nonzero_inds = np.where(max_overlaps > 0)[0]
        assert all(max_classes[nonzero_inds] != 0)


if __name__ == '__main__':
    args = parse_args()
    # Set up voc_<year>_<split> using selective search "fast" mode
    for year in ['2007', '2012']:
        for split in ['train', 'val', 'trainval', 'test']:
            name = f'voc_{year}_{split}'
            __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

    imdb = get_imdb(args.imdb_name)
    print(f'loaded dataset {args.imdb_name} for training ...')
    roidb = get_training_roidb(imdb)

    train_pkl_path = '/opt/ml/fastrcnn/data/cache/all.pkl'
    if not os.path.exists(train_pkl_path):
        with open(train_pkl_path, 'wb') as fid:
            pickle.dump(roidb, fid, pickle.HIGHEST_PROTOCOL)

    output_dir = get_output_dir(imdb, None)
    print(f'Output will be saved to {output_dir}')

    train_imgs = []
    train_img_info = []
    train_roi = []
    train_cls = []
    train_tbbox = []

    gt_pkl_path = '/opt/ml/fastrcnn/data/cache/voc_2007_trainval_gt_roidb.pkl'
    ss_pkl_path = '/opt/ml/fastrcnn/data/cache/voc_2007_trainval_selective_search_roidb.pkl'
    image_root_dir = '/opt/ml/fastrcnn/data/VOCdevkit2007/VOC2007/JPEGImages'

    with open(train_pkl_path, 'rb') as fid:
        data = pickle.load(fid)
    with open(gt_pkl_path, 'rb') as fid:
        gt_data = pickle.load(fid)
        # with open(ss_pkl_path, 'rb') as fid:
        #     ss_data = pickle.load(fid)
        '''xmin,ymin,xmax,ymax
        array([[262, 210, 323, 338],
               [164, 263, 252, 371],
               [  4, 243,  66, 373],
               [240, 193, 294, 298],
               [276, 185, 311, 219]], dtype=uint16)'''
    N_train = len(data)
    for i in trange(N_train):
        img_path = data[i]['image']
        gt_boxs = gt_data[i]['boxes']
        gt_classes = gt_data[i]['gt_classes']
        nobj = gt_boxs.shape[0]
        bboxs = data[i]['boxes'][nobj:]
        nroi = len(bboxs)

        img = Image.open(img_path)
        img_size = img.size
        img = img.resize((224, 224))
        img = np.array(img).astype(np.float32)
        img = np.transpose(img, [2, 0, 1])

        rbboxs = rel_bbox(img_size, bboxs)
        ious = calc_ious(bboxs, gt_boxs)
        max_ious = ious.max(axis=1)
        max_idx = ious.argmax(axis=1)
        tbbox = bbox_transform(bboxs, gt_boxs[max_idx])

        pos_idx = []
        neg_idx = []

        for j in range(nroi):
            if max_ious[j] < 0.1:
                continue

            gid = len(train_roi)
            train_roi.append(rbboxs[j])
            train_tbbox.append(tbbox[j])

            if max_ious[j] >= 0.5:
                pos_idx.append(gid)
                train_cls.append(gt_classes[max_idx[j]])
            else:
                neg_idx.append(gid)
                train_cls.append(0)

        pos_idx = np.array(pos_idx)
        neg_idx = np.array(neg_idx)
        train_imgs.append(img)
        train_img_info.append({
            'img_size': img_size,
            'pos_idx': pos_idx,
            'neg_idx': neg_idx,
        })
        # print(len(pos_idx), len(neg_idx))
    train_imgs = np.array(train_imgs)
    train_img_info = np.array(train_img_info)
    train_roi = np.array(train_roi)
    train_cls = np.array(train_cls)
    train_tbbox = np.array(train_tbbox).astype(np.float32)

    print(f'Training image dataset shape : {train_imgs.shape}')
    print(f'ROI : {train_roi.shape}, Train_cls : {train_cls.shape}, Train_tbbox : {train_tbbox.shape}')
    '''Training image dataset shape : (5011, 3, 224, 224)
    ROI : (2043372, 4), Train_cls : (2043372,), Train_tbbox : (2043372, 4)'''
    np.savez(open('/opt/ml/fastrcnn/data/cache/train.npz', 'wb'),
             train_imgs=train_imgs, train_img_info=train_img_info,
             train_roi=train_roi, train_cls=train_cls, train_tbbox=train_tbbox)
