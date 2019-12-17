import sys

sys.path.append('.')
from config import config
import os
from os import mkdir
import torch
from torchvision import transforms
from torch.autograd import Variable
import argparse
from utils.logger import setup_logger
from model import build_model
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt


def plot(name, title, legendx, legendy, x, y, n_epoch, frame_size=256, labelx='Epoch', labely='Loss'):
    i = 0
    x = np.array(x).flatten('F')
    y = np.array(y).flatten('F')
    framex = []
    framey = []

    while i * frame_size < len(x):
        framex.append(np.mean(x[i * frame_size:min(len(x), (i + 1) * frame_size)]))
        framey.append(np.mean(y[i * frame_size:min(len(y), (i + 1) * frame_size)]))
        i += 1

    a = np.arange(0, len(x), len(x) / len(framex))
    b = a / len(y) * n_epoch
    a = a / len(x) * n_epoch

    plt.figure()
    plt.plot(a, framex)
    plt.plot(b, framey)
    plt.xlabel(labelx)
    plt.ylabel(labely)
    plt.title(title)
    plt.legend([legendx, legendy])
    plt.savefig(name, dpi=600)
    plt.show()


def train_batch(model, optimizer, img, rois, ridx, gt_cls, gt_tbbox, is_val=False):
    sc, r_bbox = model(img, rois, ridx)
    loss, loss_sc, loss_loc = model.calc_loss(sc, r_bbox, gt_cls, gt_tbbox)
    fl = loss.data.cpu().numpy()
    fl_sc = loss_sc.data.cpu().numpy()
    fl_loc = loss_loc.data.cpu().numpy()

    if not is_val:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return fl, fl_sc, fl_loc


def train_epoch(run_set, train_imgs, train_img_info, train_roi, train_cls, train_tbbox, model, optimizer, is_val=False):
    I = 2  # N pics
    B = 64  # number of rois per image, == R=128/N=2
    POS = int(B * 0.25)  # 16
    NEG = B - POS  # 48
    Nimg = len(run_set)
    perm = np.random.permutation(Nimg)
    perm = run_set[perm]
    losses = []
    losses_sc = []
    losses_loc = []

    for i in trange(0, Nimg, I):
        lb = i
        rb = min(i + I, Nimg)
        torch_seg = torch.from_numpy(perm[lb:rb])
        img = Variable(train_imgs[torch_seg], volatile=is_val).cuda()
        ridx = []
        glo_ids = []

        for j in range(lb, rb):
            info = train_img_info[perm[j]]
            pos_idx = info['pos_idx']
            neg_idx = info['neg_idx']
            ids = []

            if len(pos_idx) > 0:
                ids.append(np.random.choice(pos_idx, size=POS))
            if len(neg_idx) > 0:
                ids.append(np.random.choice(neg_idx, size=NEG))
            if len(ids) == 0:
                continue
            ids = np.concatenate(ids, axis=0)
            glo_ids.append(ids)
            ridx += [j - lb] * ids.shape[0]

        if len(ridx) == 0:
            continue
        glo_ids = np.concatenate(glo_ids, axis=0)
        ridx = np.array(ridx)

        rois = train_roi[glo_ids]
        gt_cls = Variable(torch.from_numpy(train_cls[glo_ids]), volatile=is_val).cuda()
        gt_tbbox = Variable(torch.from_numpy(train_tbbox[glo_ids]), volatile=is_val).cuda()

        loss, loss_sc, loss_loc = train_batch(model, optimizer, img, rois, ridx, gt_cls, gt_tbbox, is_val=is_val)
        losses.append(loss)
        losses_sc.append(loss_sc)
        losses_loc.append(loss_loc)

    avg_loss = np.mean(losses)
    avg_loss_sc = np.mean(losses_sc)
    avg_loss_loc = np.mean(losses_loc)
    print(f'Avg loss = {avg_loss:.4f}; loss_sc = {avg_loss_sc:.4f}, loss_loc = {avg_loss_loc:.4f}')

    return losses, losses_sc, losses_loc


def train(cfg):
    n_epoch = cfg.TRAIN.N_EPOCH
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    model = build_model(cfg)
    # print(model)
    tl = []
    ts = []
    to = []
    vl = []
    vs = []
    vo = []
    dataset_path = '/opt/ml/fastrcnn/data/cache/train.npz'
    fid = open(dataset_path, 'rb')
    npz = np.load(fid, allow_pickle=True)
    train_imgs = npz['train_imgs']
    train_img_info = npz['train_img_info']
    train_roi = npz['train_roi']
    train_cls = npz['train_cls']
    train_tbbox = npz['train_tbbox']
    fid.close()
    trans_img_list = []
    for _, img in enumerate(train_imgs):
        img = transform(img.astype(np.uint8))
        trans_img_list.append(img.unsqueeze(0))
    train_imgs = torch.cat(trans_img_list, dim=0)
    del trans_img_list
    Ntotal = train_imgs.size(0)
    Ntrain = int(Ntotal * 0.8)
    pm = np.random.permutation(Ntotal)
    train_set = pm[:Ntrain]
    val_set = pm[Ntrain:]

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    '''train_epoch(run_set, train_imgs, train_img_info, train_roi, train_cls, train_tbbox, is_val=False):'''

    for i in range(n_epoch):
        print(f'===========================================')
        print(f'[Training Epoch {i + 1}]')
        train_loss, train_sc, train_loc = train_epoch(train_set, train_imgs, train_img_info, train_roi, train_cls,
                                                      train_tbbox, model, optimizer, False)
        print(f'[Validation Epoch {i + 1}]')
        val_loss, val_sc, val_loc = train_epoch(val_set, train_imgs, train_img_info, train_roi, train_cls,
                                                train_tbbox, model, optimizer, True)

        tl.append(train_loss)
        ts.append(train_sc)
        to.append(train_loc)
        vl.append(val_loss)
        vs.append(val_sc)
        vo.append(val_loc)

    plot('loss', 'Train/Val : Loss', 'Train', 'Validation', tl, vl, n_epoch)
    plot('loss_sc', 'Train/Val : Loss_sc', 'Train', 'Validation', ts, vs, n_epoch)
    plot('loss_loc', 'Train/Val : Loss_loc', 'Train', 'Validation', to, vo, n_epoch)
    torch.save(model.state_dict(), 'model/hao123.mdl')


def main():
    parser = argparse.ArgumentParser(description="PyTorch fast-rcnn Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    output_dir = config.get_output_dir('voc_2007_trainval')
    if output_dir and not os.path.exists(output_dir):
        mkdir(output_dir)

    logger = setup_logger("fast-rcnn", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info(f"Running with config:{config.cfg.CONFIG_FILE}")

    train(config.cfg)


if __name__ == '__main__':
    main()
