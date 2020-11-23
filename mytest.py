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

    feats = []
    from tqdm import tqdm
    import numpy as np
    for batch in tqdm(data_loader):
        with torch.no_grad():
            #predictions = model.forward_inception(batch, args.dt)
            predictions = model.forward(batch)
        feats.append(predictions['appr'].squeeze().cpu().numpy().reshape(10, 2, 32*2*2))
        total_loss_list, losses = loss_mng.separate_losses(batch, predictions)
        if cnt == 0:
            print(losses.keys())
            print(losses)
        sum_losses['bbox_loss'] += losses['bbox_loss']
        sum_losses['appr_pixel_loss'] += losses['appr_pixel_loss']
        cnt += args.batch_size
    
    print(sum_losses, cnt)
    for k, s in sum_losses.items():
        print(k, s/cnt)

    np.save('tmp.npy', np.array(feats))

class BinaryClassification(torch.nn.Module):
    def __init__(self, input_dimension):
            super().__init__()
            self.linear = torch.nn.Linear(input_dimension, 1)
    def forward(self, input_dimension):
            return self.linear(input_dimension)

if __name__ == '__main__':
    # args = TestOptions().parse()
    # # default
    # args.kl_loss_weight = 1e-2
    # args.l1_dst_loss_weight = 1.
    # args.bbox_loss_weight = 1.#100
    # args.l1_src_loss_weight = 1.
    # main(args)
    # input()

    import json
    import numpy as np
    from tqdm import tqdm
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open('results/train_labels.json', 'r') as f:
        y_train = np.array(json.load(f)).astype(float)
    # with open('results/test_labels.json', 'r') as f:
    with open('results/human_labels.json', 'r') as f:
        y_test = np.array(json.load(f)).astype(float)

    X_train = np.load('results/train_10000.npy',) 
    X_train = X_train.reshape(X_train.shape[0], 10, -1)
    # X_test = np.load('results/test_10000.npy')
    X_test = np.load('results/human_10000.npy')
    X_test = X_test.reshape(X_test.shape[0], 10, -1)

    target_field = 0
    X_tmp, y_tmp = [], []
    cnt = 0
    for i in range(y_test.shape[0]):
        if y_test[i, target_field] == 1.:
            cnt += 1
            X_tmp.append(X_test[i, ...])
            y_tmp.append(y_test[i, :])
        elif cnt > 0:
            cnt -= 1
            X_tmp.append(X_test[i, ...])
            y_tmp.append(y_test[i, :])
    X_test, y_test = np.array(X_tmp), np.array(y_tmp)


    X_train = X_train[:, 0:4, ...]
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test[:, 0:4, ...]
    X_test = X_test.reshape(X_test.shape[0], -1)
    y_train = y_train[:, target_field:target_field+1]
    y_test = y_test[:, target_field:target_field+1]

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    tmp = np.sum(y_train)
    class_sample_count = np.array([y_train.shape[0]-tmp, tmp])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[int(t)] for t in y_train])
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, len(samples_weight))

    batch_size = X_train.shape[0]
    train_dataset = torch.utils.data.TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                          sampler=sampler, num_workers=2)

    test_dataset = torch.utils.data.TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)


    model = BinaryClassification(X_train.shape[-1])
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    iterations = 100
    n_epochs = 10

    train_losses = np.zeros(n_epochs)
    test_losses = np.zeros(n_epochs)
    for it in range(n_epochs): 

        train_loss = 0.
        for data in train_loader:
            X_batch, y_batch = data
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print('===== Train loss =====', train_loss)
        train_losses[it] = train_loss

        test_loss = 0.
        for data in test_loader:
            X_batch, y_batch = data
            with torch.no_grad():
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
            test_loss += loss.item()
        print(test_loss)
        test_losses[it] = test_loss

        with torch.no_grad():
            p_train = model(torch.Tensor(X_train))
            p_train = (p_train.numpy() > 0)
            print(p_train.shape, np.sum(p_train), np.sum(y_train))

            train_acc = np.mean(y_train == p_train)

            p_test = model(torch.Tensor(X_test))
            p_test = (p_test.numpy() > 0)
            print(p_test.shape, np.sum(p_test), np.sum(y_test))

            test_acc = np.mean(y_test == p_test)

        # print(train_acc)
        print(test_acc)
