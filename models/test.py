#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.Update import PatternSynthesizer

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
import torch
from torchvision.transforms import transforms, functional


transform_to_image = transforms.ToPILImage()
transform_to_tensor = transforms.ToTensor()

class testDataset(Dataset):
    def __init__(self, dataset, attack_type):
        self.dataset = dataset
        self.attack_type = attack_type
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index:int):
        image, label = self.dataset[index]
        if self.attack_type == 'backdoor':
            image = PatternSynthesizer().synthesize_inputs(image=image)
        return image, label


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
        if args.attack_type == "labelflip":
            for i, item in enumerate(target):
                attack_case += 1
                if args.dataset == 'mnist' or 'fmnist' or 'cifar':
                    if y_pred[i][0].cpu().numpy() == 10 - item.cpu().numpy() - 1:
                        attack_success += 1
                elif args.dataset == 'cifar100':
                    if y_pred[i][0].cpu().numpy() == 100 - item.cpu().numpy() - 1:
                        attack_success += 1
        if args.attack_type == "backdoor":
            for i, item in enumerate(target):
                if item.cpu().numpy() != 5:
                    attack_case += 1
                    if y_pred[i][0].cpu().numpy() == 5:
                        attack_success += 1
        if args.attack_type == "scaling_attack":
            for i, item in enumerate(target):
                if item.cpu().numpy() != 7:
                    attack_case += 1
                    if y_pred[i][0].cpu().numpy() == 7:
                        attack_success += 1
    if attack_case != 0: 
        asr = 100.00 * attack_success/attack_case 
    else:
        asr = 0
    
    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss, asr

