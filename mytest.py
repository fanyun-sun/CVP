# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function

import os
import torch
from cfgs.test_cfgs import TestOptions
from utils import model_utils
import cvp.vis as vis_utils
from cvp.logger import Logger
from cvp.losses import LossManager


def main(args):
    torch.manual_seed(123)

    data_loader = model_utils.build_loaders(args)  # change to image
    model = model_utils.build_all_model(args)  # CNN, GCN, Encoder, Decoder

    loss_mng = LossManager(args)

    if args.dataset == 'ss3':
        save_iters = [50000]
    elif args.dataset == 'vvn':
        save_iters = [10, 100]
    elif args.dataset.startswith('penn'):
        save_iters = [100000, 300000]
    else:
        raise NotImplementedError

    cnt = 0
    sum_losses = {'bbox_loss':0., 'appr_pixel_loss':0.}
    from tqdm import tqdm
    for batch in tqdm(data_loader):
        predictions = model(batch)
        total_loss_list, losses = loss_mng.separate_losses(batch, predictions)
        if cnt == 0:
            print(losses.keys())
        sum_losses['bbox_loss'] += losses['bbox_loss']
        sum_losses['appr_pixel_loss'] += losses['appr_pixel_loss']
        cnt += args.batch_size
    
    print(sum_losses, cnt)
    for k, s in sum_losses.items():
        print(k, s/cnt)


if __name__ == '__main__':
    args = TestOptions().parse()
    # default
    args.kl_loss_weight = 1e-2
    args.l1_dst_loss_weight = 1.
    args.bbox_loss_weight = 1.#100
    args.l1_src_loss_weight = 1.
    main(args)
