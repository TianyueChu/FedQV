#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader, Dataset

from utils.sampling import mnist_iid, noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import build_model
from models.Fed import aggregate,Krum,trimmed_mean,Median
from models.test import test_img

import math

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            print("iid setting in MNIST")
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            print("Non-iid setting in MNIST")
            dict_users = noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            print("iid setting in CIFAR")
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            print("Non-iid setting in CIFAR")
            dict_users = noniid(dataset_train, args.num_users)
    
    elif args.dataset == 'cifar100':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR100('../data', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR100('../data', train=False, download=True, transform=trans_cifar)
        if args.iid:
            print("iid setting in CIFAR100")
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            print("Non-iid setting in CIFAR100")
            dict_users = noniid(dataset_train, args.num_users)
    
    elif args.dataset == 'fmnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3520,))])
        dataset_train = datasets.FashionMNIST('../data/fashionmnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.FashionMNIST('../data/fashionmnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            print("iid setting in FashionMNIST")
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            print("Non-iid setting in FashionMNIST")
            dict_users = noniid(dataset_train, args.num_users)
            
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    net_glob = build_model(args)
    print(net_glob)
   
    # copy weights
    w_glob = net_glob.state_dict()
    net_glob.train()
    
    # training
    loss_train = []
    accs_train = []
    accs_test = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    vote_budget = [args.budget]*args.num_users
    global_acc = 1
    np.random.seed = args.seed
    
    # the training round is under attack
    if args.num_attackers != 0:
        idxs_attackers = np.random.choice(range(args.num_users), args.num_attackers, replace=False)
    else:
        idxs_attackers = []
    print(idxs_attackers, "are attackers")
    
    
    if args.all_clients: 
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
    
    # parites paticipant in this round    
    for iter in range(args.epochs):
        loss_locals = []
        if not args.all_clients:
            w_locals = []
            num_samples = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        print(idxs_users, "paticipant in this round")
        
        # budget of parites participate in this epoch
        vote_budget_epoch = []
        attack_ind = []
        
        for idx in idxs_users:
            # the user is an attacker
            if idx in idxs_attackers:
                attack_mode = True
                print("attacker", idx, "joins")
            else:
                attack_mode = False
            
            vote_budget_epoch.append(vote_budget[idx])
            
            if vote_budget[idx] == 0:
                print('voter {} budget run out'.format(idx))

            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], attack_mode = attack_mode)
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
           
            if attack_mode:
                if args.attack_type ==  'gaussian_attack':
                    for k in w.keys():
                        mean = 0
                        std = 1
                        noisy = std * torch.randn(w[k].size()) + mean
                        w[k] += noisy.to('cuda:0').long()
                elif args.attack_type == 'scaling_attack':
                    if iter == args.epochs - 1 :
                        for k in w.keys():
                            w[k] = min(args.frac* args.num_users, 10) * w[k]
                elif args.attack_type ==  'krum_attack' or 'trimmed-mean_attack' or 'median_attack':
                    attack_ind.append(idxs_users.tolist().index(idx))
                 
                    
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
                num_samples.append(len(dict_users[idx]))
           
            loss_locals.append(copy.deepcopy(loss))
            
        
        if args.attack_type == 'krum_attack':
            w_krum = Krum(w_locals,w_glob)
            for k in w_krum.keys():
                w_krum[k] = - w_krum[k]
            
            for idx in attack_ind:
                for k in w_krum.keys():
                    w_locals[idx][k] = w_krum[k]
        
        
        if args.attack_type == 'trim_attack':
            w_trim = trimmed_mean(w_locals,0.4)
            for k in w_trim.keys():
                w_trim[k] = - w_trim[k]
                
            for idx in attack_ind:
                for k in w_trim.keys():
                    w_locals[idx][k] = w_trim[k] 
        
        if args.attack_type == 'median_attack':  
            w_med = Median(w_locals)
            for k in w_med.keys():
                w_med[k] = - w_med[k]
            
            for idx in attack_ind:
                for k in w_med.keys():
                    w_locals[idx][k] = w_med[k] 
        
             
        # update global weights
        w_glob, vote_budget_epoch = aggregate(args, w_locals, num_samples, w_glob, vote_budget_epoch, global_acc)
        
        # update vote budget
        for i, idx in enumerate(idxs_users):
            vote_budget[idx] = vote_budget_epoch[i]

        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)
        
        net_glob.eval()
        acc_train, _ , _ = test_img(net_glob, dataset_train, args)
        accs_train.append(float(acc_train))
        acc_test, _ , asr = test_img(net_glob, dataset_test, args)
        accs_test.append(float(acc_test))
        global_acc = acc_test.tolist()/100 
        
        if args.num_attackers == 0:
            asr = 0
            
        print("Training accuracy: {:.2f}".format(acc_train))
        print("Testing accuracy: {:.2f}".format(acc_test))
        print("Attack success rate: {:.2f}".format(asr))
        
        
    print("Training loss:",loss_train)
    print("Training accuracy: ", accs_train)
    print("Testing accuracy: ",accs_test)

    
    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    # testing
    net_glob.eval()
    acc_train, _ , _ = test_img(net_glob, dataset_train, args)
    acc_test, _ , asr = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))
    
    if args.num_attackers == 0:
        asr = 0
    print("Attack success rate: {:.2f}".format(asr))
        

