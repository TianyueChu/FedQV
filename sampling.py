#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms
import random
import copy
from collections import defaultdict

import torch

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def noniid(dataset, num_users, alpha=0.9):
    """
    Sample non-I.I.D client data
    :param dataset:
    :param num_users:
    :return:
    """
    classes = {}
    for idx, x in enumerate(dataset):
        _, label = x
        # label=label.item() # for gpu
        if label in classes:
            classes[label].append(idx)
        else:
            classes[label] = [idx]
            
    num_classes = len(classes.keys())
    class_size = len(classes[0])
    num_participants= num_users 
    dict_users = {i: np.array([]) for i in range(num_users)}

    for n in range(num_classes):
        random.shuffle(classes[n])
        sampled_probabilities = class_size * np.random.dirichlet(np.array(num_participants * [alpha]))
        for user in range(num_participants):
            num_imgs = int(round(sampled_probabilities[user]))
            sampled_list = classes[n][:min(len(classes[n]), num_imgs)]
            dict_users[user] = np.concatenate((dict_users[user], np.array(sampled_list)), axis=0)
            classes[n] = classes[n][min(len(classes[n]), num_imgs):]

    # shuffle data
    for user in range(num_participants):
        dict_users[user] = dict_users[user].astype(np.int)
        # print(dict_users[user])
        random.shuffle(dict_users[user])
    return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
