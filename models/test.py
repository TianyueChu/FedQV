#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def test_img(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
    attack_success = 0
    attack_case = 0
    
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.cuda(), target.cuda()
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
        if args.dataset == 'mnist':
            for i, item in enumerate(target):
                if item.cpu().numpy() == 1:
                    attack_case += 1
                    if y_pred[i][0].cpu().numpy() == 7:
                        attack_success += 1
        elif args.dataset == 'cifar':
            for i, item in enumerate(target):
                if item.cpu().numpy() == 3:
                    attack_case += 1
                    if y_pred[i][0].cpu().numpy() == 5:
                        attack_success += 1
        elif args.dataset == 'cifar100':
            for i, item in enumerate(target):
                if item.cpu().numpy() == 71:
                    attack_case += 1
                    if y_pred[i][0].cpu().numpy() == 73:
                        attack_success += 1
    # print('attack_success',attack_success)
    # print('attack case', attack_case)
    asr = 100.00 * attack_success/attack_case 
    
    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss, asr

