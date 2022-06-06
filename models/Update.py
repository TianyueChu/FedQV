#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs, attack_mode, attack_type, name_dataset):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.attack_mode = attack_mode
        self.attack_type = attack_type
        self.name_dataset = name_dataset
        
    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        if self.attack_mode:
            if self.attack_type == 'lableflip':
                if self.name_dataset == 'mnist':
                    # flip 1 to 7 for MNIST
                    if label == 1:   
                        label = label + 6
                elif self.name_dataset == 'cifar':
                    # flip 3 to 5 for CIFAR
                     if label == 3:   
                        label = label + 2
                elif self.name_dataset == 'cifar100':
                    # flip 71 to 73 for CIFAR
                     if label == 71:   
                        label = label + 2  
            elif self.attack_type =='backdoor':
                label = label
                
            elif self.attack_type == 'gaussian_noise':
                label = label
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, attack_mode=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs, attack_mode, args.attack_type, args.dataset), batch_size=self.args.local_bs, shuffle=True, drop_last=True)
        # self.poison_train = DatasetSplit(dataset, idxs, attack_mode)

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

