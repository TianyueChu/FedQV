#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn
import math
import numpy as np
import sklearn.metrics.pairwise as smp

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer


def aggregate(args,w_local,n,w_global,vote_budget):
    if args.agg == 'fedavg':
        print("using FedAvg")
        w_avg = FedAvg(w_local,n)
    elif args.agg == 'fedq':
        print("using FedQ Estimator")
        w_avg = FedQ(w_local,n)
    elif args.agg == 'fedqv':
        print("using FedQV Estimator")
        w_avg, vote_budget = FedQV(w_local,w_global,vote_budget,n)
    else:
        exit('Error: unrecognized aggregation method')
    return w_avg, vote_budget

def FedAvg(w,n):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = torch.mul(w[0][k],n[0]/sum(n))
        for i in range(1, len(w)):
            w_avg[k] += torch.mul(w[i][k],n[i]/sum(n))
    return w_avg

def FedQ(w,n):
    w_avg = copy.deepcopy(w[0])
    sum_weight = sum(np.sqrt(n))
    # k: layer in NN
    for k in w_avg.keys():
        w_avg[k] = torch.mul(w[0][k],math.sqrt(n[0])/sum_weight)
        # len(w): num_clients
        for i in range(1, len(w)):
            w_avg[k] += torch.mul(w[i][k],math.sqrt(n[i])/sum_weight)
    return w_avg

def FedQV(w, w_global, vote_budget,n):
    # type(w): list
    # type(w_global): OrderedDict
    # type(w[0]): OrderedDict
    # len(w[0]): how many layers in NN
    num_clients = len(w)
    glob_grad = []
    
    for j in list(w_global.values()):
        glob_grad = np.append(glob_grad, np.array(j.tolist()).flatten())
        # print(glob_grad.shape)
        # glob_grad += np.array(j.tolist()).reshape(1,-1).tolist()
        
    grad_len = len(glob_grad)
    grads = np.zeros((num_clients, grad_len))
    weight = []
    
    for i in range(num_clients):
        layer = []
        for j in list(w[i].values()):
            layer = np.append(layer, np.array(j.tolist()).flatten())
            # print(layer.shape)
        # grads[i] contains all the grads of client[i]
        grads[i] = layer
        weight.append(smp.cosine_similarity(grads[i].reshape(1,-1), glob_grad.reshape(1,-1)).flatten().tolist()[-1])
        # print('voice credit:',weight[-1])
    # print(weight)
    
    # Normalization
    # https://towardsdatascience.com/scale-standardize-or-normalize-with-scikit-learn-6ccc7d176a02
    # normalized_weight = Normalizer().fit_transform(np.array(weight))
    # normalized_weight = preprocessing.normalize([np.array(weight)])
    
    # Case 1: normalization
    scaler = MinMaxScaler()
    normalized_weight = scaler.fit_transform(np.array(weight).reshape(-1,1))
    # print(normalized_weight)
    voice_credit = []
    for i in range(len(normalized_weight)):
       if normalized_weight[i][0] in [0,1]:
          voice_credit.append(0)
       else:
          if -math.log(normalized_weight[i][0]) >= 0 :
              voice_credit.append(-math.log(normalized_weight[i][0]))
          else:
              voice_credit.append(0)
              print('abnormal weight is', normalized_weight[i][0])
    
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    # Case 2: without normalization
    # normalized_weight = weight
    # weight_sum = sum(weight)
    # for i in range(len(normalized_weight)):
        # normalized_weight[i] = normalized_weight[i]/weight_sum
    
    # print(normalized_weight)    
    # voice_credit = []
    # for i in range(len(normalized_weight)):
       # if normalized_weight[i] in [0,1]:
           # voice_credit.append(0)
       # else:
           # if -math.log(normalized_weight[i]) >= 0 :
               # voice_credit.append(-math.log(normalized_weight[i]))
           # else:
               # voice_credit.append(0)
               # print('abnormal weight is', normalized_weight[i]) 

    # print('voice_credit', voice_credit, 'before budget')
        
    # voice credit budget
    for i in range(len(vote_budget)):
        if vote_budget[i] == 0:
            voice_credit[i] = 0
            # print('voter {} budget run out'.format(i))
        elif vote_budget[i]- voice_credit[i] <= 0:
            voice_credit[i] = vote_budget[i]
            vote_budget[i] = 0
            # print('vote budget run out to', vote_budget[i] , 'voice credit is', voice_credit[i] )
        elif vote_budget[i]- voice_credit[i] > 0:
            vote_budget[i] = vote_budget[i]-voice_credit[i]
            # print('vote budget is', vote_budget[i] , 'voice credit is', voice_credit[i] )
        else:
            print('Error')
    
    print('After budget, voice_credit', voice_credit)
    
    voice_credit = [a*b for a,b in zip(voice_credit,n)]
    
    w_avg = copy.deepcopy(w[0])
    sum_weight = sum(np.sqrt(voice_credit))
    agg_weight = []
    for i in range(len(voice_credit)):
        agg_weight.append(math.sqrt(voice_credit[i])/sum_weight)
    print('aggregation weights are', agg_weight)
        
    # k: layer in NN
    for k in w_avg.keys():
        w_avg[k] = torch.mul(w[0][k],math.sqrt(voice_credit[0])/sum_weight)
        # len(w): num_clients
        for i in range(1, len(w)):
            w_avg[k] += torch.mul(w[i][k],math.sqrt(voice_credit[i])/sum_weight)
    return w_avg, vote_budget
    
    
def weighted_average(w_list, weights):
    w_avg = copy.deepcopy(w_list[0])
    weights = weights / weights.sum()
    assert len(weights) == len(w_list)
    for k in w_avg.keys():
        w_avg[k] = 0
        for i in range(0, len(w_list)):
            w_avg[k] += w_list[i][k] * weights[i]
        # w_avg[k] = torch.div(w_avg[k], len(w_list))
    return w_avg, weights
