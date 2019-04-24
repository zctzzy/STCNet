# -*- coding: utf-8 -*-
"""
/*******************************************
** This is a file created by Chuanting Zhang
** Name: transfer
** Date: 5/18/18
** Email: chuanting.zhang@gmail.com
** BSD license
********************************************/
"""

import os
import sys
import argparse
import numpy as np
from datetime import datetime
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
sys.path.append('../../')
from Github_Version.utils.dataset import read_data
from Github_Version.utils.model import DenseNet

torch.manual_seed(22)

device = torch.device("cuda")

parse = argparse.ArgumentParser()
parse.add_argument('-height', type=int, default=100)
parse.add_argument('-width', type=int, default=100)
parse.add_argument('-traffic', type=str, default='sms')
parse.add_argument('-meta', type=int, default=1)
parse.add_argument('-cross', type=int, default=1)
parse.add_argument('-close_size', type=int, default=3)
parse.add_argument('-period_size', type=int, default=0)
parse.add_argument('-trend_size', type=int, default=0)
parse.add_argument('-test_size', type=int, default=24*7)
parse.add_argument('-nb_flow', type=int, default=1)
parse.add_argument('-cluster', type=int, default=1)
parse.add_argument('-fusion', type=int, default=1)
parse.add_argument('-transfer', type=int, default=0)
parse.add_argument('-target_type', type=str, default='internet')
parse.add_argument('-crop', dest='crop', action='store_true')
parse.add_argument('-no-crop', dest='crop', action='store_false')
parse.set_defaults(crop=True)
parse.add_argument('-train', dest='train', action='store_true')
parse.add_argument('-no-train', dest='train', action='store_false')
parse.set_defaults(train=True)
parse.add_argument('-rows', nargs='+', type=int, default=[40, 60])
parse.add_argument('-cols', nargs='+', type=int, default=[40, 60])
parse.add_argument('-loss', type=str, default='l2', help='l1 | l2')
parse.add_argument('-lr', type=float, default=0.01)
parse.add_argument('-batch_size', type=int, default=32, help='batch size')
parse.add_argument('-epoch_size', type=int, default=30, help='epochs')
parse.add_argument('-test_row', type=int, default=10, help='test row')
parse.add_argument('-test_col', type=int, default=18, help='test col')

parse.add_argument('-save_dir', type=str, default='results')

opt = parse.parse_args()
print(opt)
opt.save_dir = '{}/{}'.format(opt.save_dir, opt.traffic)


def log(fname, s):
    if not os.path.isdir(os.path.dirname(fname)):
        os.system("mkdir -p " + os.path.dirname(fname))
    f = open(fname, 'a')
    f.write(str(datetime.now()) + ': ' + s + '\n')
    f.close()


def train_epoch(data_type='train'):
    total_loss = 0
    if data_type == 'train':
        model.train()
        data = train_loader
    if data_type == 'valid':
        model.eval()
        data = valid_loader

    if (opt.close_size > 0) & (opt.meta == 1) & (opt.cross ==1):
        for idx, (c, meta, cross, target) in enumerate(data):
            optimizer.zero_grad()
            model.zero_grad()

            x = c.float().to(device)
            meta = meta.float().to(device)
            cross = cross.float().to(device)
            target_var = target.float().to(device)

            pred = model(x, meta=meta, cross=cross)
            loss = criterion(pred, target_var)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
    elif (opt.close_size > 0) & (opt.meta == 1):
        for idx, (x, meta, target) in enumerate(data):
            optimizer.zero_grad()
            model.zero_grad()

            x = x.float().to(device)
            meta = meta.float().to(device)
            target_var = target.float().to(device)

            pred = model(x, meta=meta)
            loss = criterion(pred, target_var)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
    elif (opt.close_size > 0) & (opt.cross == 1):
        for idx, (x, cross, target) in enumerate(data):
            optimizer.zero_grad()
            model.zero_grad()

            x = x.float().to(device)
            cross = cross.float().to(device)
            target_var = target.float().to(device)

            pred = model(x, cross=cross)
            loss = criterion(pred, target_var)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
    elif opt.close_size > 0:
        for idx, (x, target) in enumerate(data):
            optimizer.zero_grad()
            model.zero_grad()
            x = x.float().to(device)
            y = target.float().to(device)

            pred = model(x)
            loss = criterion(pred, y)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

    return total_loss


def train():
    os.system("mkdir -p " + opt.save_dir)
    best_valid_loss = 1.0
    train_loss, valid_loss = [], []
    for i in range(opt.epoch_size):
        scheduler.step()
        train_loss.append(train_epoch('train'))
        valid_loss.append(train_epoch('valid'))

        if valid_loss[-1] < best_valid_loss:
            best_valid_loss = valid_loss[-1]

            torch.save({'epoch': i, 'model': model, 'train_loss': train_loss,
                        'valid_loss': valid_loss}, opt.model_filename + '.model')
            torch.save(optimizer, opt.model_filename + '.optim')
            torch.save(model.state_dict(), opt.model_filename + '.pt')

        log_string = ('iter: [{:d}/{:d}], train_loss: {:0.6f}, valid_loss: {:0.6f}, '
                      'best_valid_loss: {:0.6f}, lr: {:0.5f}').format((i + 1), opt.epoch_size,
                                                                      train_loss[-1],
                                                                      valid_loss[-1],
                                                                      best_valid_loss,
                                                                      opt.lr)
        if i % 2 == 0:
            print(log_string)
        log(opt.model_filename + '.log', log_string)

def predict(test_type='train'):
    predictions = []
    ground_truth = []
    loss = []
    model.eval()
    model.load_state_dict(torch.load(opt.model_filename + '.pt'))

    if test_type == 'train':
        data = train_loader
    elif test_type == 'test':
        data = test_loader
    elif test_type == 'valid':
        data = valid_loader

    with torch.no_grad():
        if (opt.close_size > 0) & (opt.meta == 1) & (opt.cross == 1):
            for idx, (c, meta, cross, target) in enumerate(data):
                optimizer.zero_grad()
                model.zero_grad()
                x = c.float().to(device)
                meta = meta.float().to(device)
                cross = cross.float().to(device)
                # input_var = [_.float().to(device) for _ in [c, meta, cross]]
                target_var = target.float().to(device)

                pred = model(x, meta=meta, cross=cross)
                predictions.append(pred.data.cpu())
                ground_truth.append(target.data)
                loss.append(criterion(pred, target_var).item())
        elif (opt.close_size > 0) & (opt.meta == 1):
            for idx, (x, meta, target) in enumerate(data):
                optimizer.zero_grad()
                model.zero_grad()
                # input_var = [_.float() for _ in [c, meta]]
                x = x.float().to(device)
                meta = meta.float().to(device)
                y = target.float().to(device)

                pred = model(x, meta=meta)
                predictions.append(pred.data.cpu())
                ground_truth.append(target.data)
                loss.append(criterion(pred, y).item())
        elif (opt.close_size > 0) & (opt.cross == 1):
            for idx, (x, cross, target) in enumerate(data):
                optimizer.zero_grad()
                model.zero_grad()
                # input_var = [_.float() for _ in [c, meta]]
                x = x.float().to(device)
                cross = cross.float().to(device)
                y = target.float().to(device)

                pred = model(x, cross=cross)
                predictions.append(pred.data.cpu())
                ground_truth.append(target.data)
                loss.append(criterion(pred, y).item())
        elif opt.close_size > 0:
            for idx, (c, target) in enumerate(data):
                optimizer.zero_grad()
                model.zero_grad()
                x = c.float().to(device)
                y = target.float().to(device)

                pred = model(x)
                predictions.append(pred.data.cpu())
                ground_truth.append(target.data)
                loss.append(criterion(pred, y).item())

    final_predict = np.concatenate(predictions)
    ground_truth = np.concatenate(ground_truth)
    print(final_predict.shape, ground_truth.shape)

    ground_truth = mmn.inverse_transform(ground_truth)
    final_predict = mmn.inverse_transform(final_predict)
    return final_predict, ground_truth


def train_valid_split(dataloader, test_size=0.2, shuffle=True, random_seed=0):
    length = len(dataloader)
    indices = list(range(0, length))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    if type(test_size) is float:
        split = int(np.floor(test_size * length))
    elif type(test_size) is int:
        split = test_size
    else:
        raise ValueError('%s should be an int or float'.format(str))
    return indices[split:], indices[:split]


if __name__ == '__main__':
    path = '/home/dl/ct/data/data_git_version.h5'
    feature_path = '/home/dl/ct/data/crawled_feature.csv'

    X, X_meta, X_cross, y, label, mmn = read_data(path, feature_path, opt)

    if opt.cluster > 1:
        labels_df = pd.read_csv('cluster_label.csv', header=None)
        labels_df.columns = ['cluster_label']
    else:
        labels_df = pd.DataFrame(np.ones(shape=(len(label),)), columns=['cluster_label'])

    samples, sequences, channels, height, width = X.shape

    x_train, x_test = X[:-opt.test_size], X[-opt.test_size:]
    meta_train, meta_test = X_meta[:-opt.test_size], X_meta[-opt.test_size:]
    cross_train, cross_test = X_cross[:-opt.test_size], X_cross[-opt.test_size:]
    y_tr = y[:-opt.test_size]
    y_te = y[-opt.test_size:]

    prediction_ct = 0
    truth_ct = 0
    for cluster_id in (set(labels_df['cluster_label'].values)):
        print('Cluster: %d' % cluster_id)

        opt.model_filename = '{}/model={}lr={}-close={}-period=' \
                             '{}-meta={}-cross={}-crop={}-' \
                             'cluster={}-target={}-transfer={}'.format(opt.save_dir,
                                                                       'densenet', opt.lr, opt.close_size,
                                                                       opt.period_size, opt.meta, opt.cross,
                                                                       opt.crop, cluster_id, opt.target_type,
                                                                       opt.transfer)
        print('Saving to ' + opt.model_filename)

        labels_df['cur_label'] = 0
        labels_df['cur_label'][labels_df['cluster_label'] == int(cluster_id)] = 1
        cell_idx = labels_df['cur_label'] == 1
        cell_idx = np.reshape(cell_idx, (height, width))

        y_train = y_tr * cell_idx
        y_test = y_te * cell_idx

        if (opt.meta == 1) & (opt.cross == 1):
            train_data = list(zip(*[x_train, meta_train, cross_train, y_train]))
            test_data = list(zip(*[x_test, meta_test, cross_test, y_test]))
        elif (opt.meta == 1) & (opt.cross == 0):
            train_data = list(zip(*[x_train, meta_train, y_train]))
            test_data = list(zip(*[x_test, meta_test, y_test]))
        elif (opt.cross == 1) & (opt.meta == 0):
            train_data = list(zip(*[x_train, cross_train, y_train]))
            test_data = list(zip(*[x_test, cross_test, y_test]))
        elif (opt.meta == 0) & (opt.cross == 0):
            train_data = list(zip(*[x_train, y_train]))
            test_data = list(zip(*[x_test, y_test]))

        print(len(train_data), len(test_data))

        # split the training data into train and validation
        train_idx, valid_idx = train_valid_split(train_data, 0.1)
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(train_data, batch_size=opt.batch_size, sampler=train_sampler,
                                  num_workers=8, pin_memory=True)
        valid_loader = DataLoader(train_data, batch_size=opt.batch_size, sampler=valid_sampler,
                                  num_workers=2, pin_memory=True)

        test_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False)

        input_shape = X.shape
        meta_shape = X_meta.shape
        cross_shape = X_cross.shape

        model = DenseNet(input_shape, meta_shape,
                         cross_shape, nb_flows=opt.nb_flow,
                         fusion=opt.fusion, maps=(opt.meta + opt.cross + 1)).to(device)

        if cluster_id > 1:
            model_name = '{}/model={}lr={}-close={}-period=' \
                                 '{}-meta={}-cross={}-crop={}-cluster={}'.format(opt.save_dir,
                                                                                 'densenet',
                                                                                 opt.lr,
                                                                                 opt.close_size,
                                                                                 opt.period_size,
                                                                                 opt.meta,
                                                                                 opt.cross, opt.crop, cluster_id-1)
            model.load_state_dict(torch.load(model_name + '.pt'))

        if opt.transfer == 1:
            opt.lr = opt.lr / 2.0
            if opt.traffic == 'sms':
                if opt.target_type == 'call':
                    model.load_state_dict(torch.load('./results/call/base_200.pt'))
                elif opt.target_type == 'internet':
                    model.load_state_dict(torch.load('./results/internet/base_200.pt'))
                else:
                    raise NotImplementedError()
            if opt.traffic == 'call':
                if opt.target_type == 'sms':
                    model.load_state_dict(torch.load('./results/sms/base_200.pt'))
                elif opt.target_type == 'internet':
                    model.load_state_dict(torch.load('./results/internet/base_200.pt'))
                else:
                    raise NotImplementedError()
            if opt.traffic == 'internet':
                if opt.target_type == 'call':
                    model.load_state_dict(torch.load('./results/call/base_200.pt'))
                elif opt.target_type == 'sms':
                    model.load_state_dict(torch.load('./results/sms/base_200.pt'))
                else:
                    raise NotImplementedError()


        optimizer = optim.Adam(model.parameters(), opt.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[0.5 * opt.epoch_size,
                                                                     0.75 * opt.epoch_size],
                                                         gamma=0.1)

        if not os.path.exists(opt.save_dir):
            os.makedirs(opt.save_dir)
        if not os.path.isdir(opt.save_dir):
            raise Exception('%s is not a dir' % opt.save_dir)

        if opt.loss == 'l1':
            criterion = nn.L1Loss().cuda()
        elif opt.loss == 'l2':
            criterion = nn.MSELoss().cuda()

        print('Training...')
        log(opt.model_filename + '.log', '[training]')
        if opt.train:
            train()

        pred, truth = predict('test')

        prediction_ct += pred * cell_idx
        truth_ct += truth * cell_idx

    # 2018-04-20 in_error and out_error
    if opt.traffic != 'internet':
        prediction_ct[-24] = ((truth_ct[-25] + truth_ct[-26] + truth_ct[-27]) / 3.0) * 2.5
    # plt.plot(prediction_ct[:, 0, opt.test_row, opt.test_col], 'r-', label='prediction')
    # plt.plot(truth_ct[:, 0, opt.test_row, opt.test_col], 'k-', label='truth')
    # plt.legend()
    #
    # train_loss_data = torch.load(opt.model_filename + '.model').get('train_loss')
    # valid_loss_data = torch.load(opt.model_filename + '.model').get('valid_loss')
    # plt.figure()
    # plt.plot(train_loss_data[10:-1], 'r-', label='train loss')
    # plt.plot(valid_loss_data[10:-1], 'k-', label='valid loss')
    # plt.legend()
    # plt.show()
    if opt.nb_flow > 1:
        print(
            'Final RMSE:{:0.5f}'.format(
                metrics.mean_squared_error(prediction_ct.ravel(), truth_ct.ravel()) ** 0.5))
        pred_in, pred_out = prediction_ct[:, 0], prediction_ct[:, 1]
        truth_in, truth_out = truth_ct[:, 0], truth_ct[:, 1]
        print('In traffic RMSE:{:0.5f}'.format(
                metrics.mean_squared_error(pred_in.ravel(), truth_in.ravel()) ** 0.5))
        print('Out traffic RMSE:{:0.5f}'.format(
            metrics.mean_squared_error(pred_out.ravel(), truth_out.ravel()) ** 0.5))
    else:
        print('Final RMSE:{:0.5f}'.format(
            metrics.mean_squared_error(prediction_ct.ravel(), truth_ct.ravel()) ** 0.5))
        print('Final MAE:{:0.5f}'.format(
            metrics.mean_absolute_error(prediction_ct.ravel(), truth_ct.ravel())))
        print('Final R2:{:0.5f}'.format(
            metrics.r2_score(prediction_ct.ravel(), truth_ct.ravel())))
        print('Final Variance:{:0.5f}'.format(
            metrics.explained_variance_score(prediction_ct.ravel(), truth_ct.ravel())))
