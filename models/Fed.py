#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn
import math
import numpy as np
import sklearn.metrics.pairwise as smp
from functools import reduce
import statistics

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer

eps = np.finfo(float).eps

def aggregate(args,w_local,n,w_global,vote_budget):
    if args.agg == 'fedavg':
        print("using FedAvg")
        w_avg = FedAvg(w_local,n)
    elif args.agg == 'fedq':
        print("using FedQ Estimator")
        w_avg = FedQ(w_local,n)
    elif args.agg == 'fedqv':
        print("using FedQV Estimator")
        w_avg, vote_budget = FedQV(w_local,w_global,vote_budget,n,args.rep)
    elif args.agg == 'krum':
        print("using Krum Estimator")
        w_avg = Krum(w_local,w_global)
    elif args.agg == 'multi-krum':
        print("using Multi-Krum Estimator")
        w_avg = Mkrum(w_local,w_global, b= 0.5* args.num_users*args.frac)
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

def Krum(w,w_global):
    num_clients = len(w)
 
    # assert n >= 2 * b + 3, "Krum requirement: n >= 2b + 3."
    # num_selection = max(num_clients - b - 2, 1)
    scores = []
    glob_grad = []
    
    for j in list(w_global.values()):
        glob_grad = np.append(glob_grad, np.array(j.tolist()).flatten())
        
    grad_len = len(glob_grad)
    grads = np.zeros((num_clients, grad_len))
    
    for i in range(num_clients):
        layer = []
        for j in list(w[i].values()):
            layer = np.append(layer, np.array(j.tolist()).flatten())
        grads[i] = layer
    
    for i in range(num_clients):
        dists = []
        for j in range(num_clients):
            if j != i: 
                d = np.sqrt(np.sum(np.power(grads[i] - grads[j],2)))
                dists.append(d)
        sorted_index = torch.argsort(torch.tensor(dists), descending=False)  
        dists = torch.tensor(dists)
        scores.append(dists[sorted_index[0]].sum())
    ind_min = scores.index(min(scores))
    print('using updates from worker ', ind_min)
    return w[ind_min]

def Mkrum(w,w_global,b):
    
    num_clients = len(w)
 
    # assert n >= 2 * b + 3, "Krum requirement: n >= 2b + 3."
    num_selection = round(max(num_clients - b - 2, 1))
    scores = []
    glob_grad = []
    ind_dict = {}
    
    for j in list(w_global.values()):
        glob_grad = np.append(glob_grad, np.array(j.tolist()).flatten())
        
    grad_len = len(glob_grad)
    grads = np.zeros((num_clients, grad_len))
    
    for i in range(num_clients):
        layer = []
        for j in list(w[i].values()):
            layer = np.append(layer, np.array(j.tolist()).flatten())
        grads[i] = layer
    
    for i in range(num_clients-1):
        dists = []
        for j in range(i+1, num_clients):
            if j != i: 
                d = np.sqrt(np.sum(np.power(grads[i] - grads[j],2)))
                dists.append(d)
        sorted_index = torch.argsort(torch.tensor(dists), descending=False) 
        # print(sorted_index)
        dists = torch.tensor(dists)
        for k in range(len(sorted_index)):
            ind_dict[str(i) +','+ str(sorted_index[k].numpy()+i+1)] = dists[sorted_index[k]]
            scores.append(dists[sorted_index[k]])
    # print(scores)
    res = torch.sort(torch.tensor(scores), descending=False)[:num_selection]
    # print('using updates from worker ', ind_min)
    # print(ind_dict)
    keys=list(ind_dict.keys()) 
    values=list(ind_dict.values())
    ind_list = []
    
    for i in res[0][0:num_selection+1].numpy():
        ind_list = ind_list + list(map(int, keys[values.index(i)].rsplit(',')))
    # print(ind_list)
    select_list = []
    [select_list.append(v) for v in ind_list if v not in select_list]
    final_list = select_list[:num_selection]
    print('worker index', final_list)
    
    #### aggregation part
    ### without QV
    w_avg = copy.deepcopy(w[final_list[0]])
    for k in w_avg.keys():
        w_avg[k] = torch.mul(w[final_list[0]][k],1/len(final_list))
        for i in final_list[1:]:
            w_avg[k] += torch.mul(w[i][k],1/len(final_list))
        
    ### if using QV
    # for k in w_avg.keys():
       # w_avg[k] = torch.mul(w[final_list[0]][k],n[final_list[0]]/sum(n))
        # for i in final_list[1:]:
            # print(i)
            # w_avg[k] += torch.mul(w[i][k],n[i]/sum(n))
            
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

def FedQV(w, w_global, vote_budget,n, reputation_on):
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
        
    if reputation_on:
        rep_score = reputation_aggregation(w, LAMBDA=2, thresh=0.05)
        for i in range(len(vote_budget)):
            if rep_score[i] > statistics.median(rep_score):
                vote_budget[i] += rep_score[i]
                print('voter',i,'increase budget',rep_score[i])
                
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


def reputation_model(w, opinion_matrix, kappa, W, a):
    
    rep_score = []
    opinion_matrix = opinion_matrix.cpu()
    for i in range(len(w)):
        belief_count = 0
        disbelief_count = 0
        for k in range(opinion_matrix.shape[0]):
            if opinion_matrix[k][i].numpy():
                disbelief_count += 1
            else:
                belief_count += 1
        belief = (belief_count*kappa)/(belief_count*kappa + disbelief_count*(1-kappa)+W)
        uncertainty = W/(belief_count*kappa + disbelief_count*(1-kappa)+W)
        rep_score.append((belief + a * uncertainty))
       
    return rep_score


def reweight_algorithm_restricted(y, LAMBDA, thresh):
    num_models = y.shape[1]
    total_num = y.shape[0]
    slopes, intercepts = repeated_median(y)
    X_pure = y.sort()[1].sort()[1].type(torch.float)

    # calculate H matrix
    X_pure = X_pure.unsqueeze(2)
    X = torch.cat((torch.ones(total_num, num_models, 1).to(y.device), X_pure), dim=-1)
    X_X = torch.matmul(X.transpose(1, 2), X)
    X_X = torch.matmul(X, torch.inverse(X_X))
    H = torch.matmul(X_X, X.transpose(1, 2))

    diag = torch.eye(num_models).repeat(total_num, 1, 1).to(y.device)
    processed_H = (torch.sqrt(1 - H) * diag).sort()[0][..., -1]
    K = torch.FloatTensor([LAMBDA * np.sqrt(2. / num_models)]).to(y.device)

    beta = torch.cat((intercepts.repeat(num_models, 1).transpose(0, 1).unsqueeze(2),
                      slopes.repeat(num_models, 1).transpose(0, 1).unsqueeze(2)), dim=-1)
    line_y = (beta * X).sum(dim=-1)
    residual = y - line_y
    M = median_opt(residual.abs().sort()[0][..., 1:])
    tau = 1.4826 * (1 + 5 / (num_models - 1)) * M + 1e-7
    e = residual / tau.repeat(num_models, 1).transpose(0, 1)
    reweight = processed_H / e * torch.max(-K, torch.min(K, e / processed_H))

    reweight[reweight != reweight] = 1
    reweight_std = reweight.std(dim=1) 
    reshaped_std = torch.t(reweight_std.repeat(num_models, 1))
    reweight_regulized = reweight * reshaped_std  
    opinion_matrix = (reweight >= thresh)
    restricted_y = y * (reweight >= thresh).type(torch.cuda.FloatTensor) + line_y * (reweight < thresh).type(
        torch.cuda.FloatTensor)
    return reweight_regulized, restricted_y, opinion_matrix


def reputation_aggregation(w_locals, LAMBDA, thresh):
    SHARD_SIZE = 2000
    w, invalid_model_idx = get_valid_models(w_locals)
    w_med = copy.deepcopy(w[0])
    # w_selected = [w[i] for i in random_select(len(w))]
    device = w[0][list(w[0].keys())[0]].device
    reweight_sum = torch.zeros(len(w)).to(device)
    opinion_matrix_sum = None 

    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda x, y: x * y, shape)
        y_list = torch.FloatTensor(len(w), total_num).to(device)
        for i in range(len(w)):
            y_list[i] = torch.reshape(w[i][k], (-1,))
        transposed_y_list = torch.t(y_list)
        y_result = torch.zeros_like(transposed_y_list)
        assert total_num == transposed_y_list.shape[0]

        if total_num < SHARD_SIZE:
            reweight, restricted_y, opinion_matrix = reweight_algorithm_restricted(transposed_y_list, LAMBDA, thresh)
            reweight_sum += reweight.sum(dim=0)
            y_result = restricted_y
        else:
            num_shards = int(math.ceil(total_num / SHARD_SIZE))
            for i in range(num_shards):
                y = transposed_y_list[i * SHARD_SIZE: (i + 1) * SHARD_SIZE, ...]
                reweight, restricted_y , opinion_matrix = reweight_algorithm_restricted(y, LAMBDA, thresh)
                reweight_sum += reweight.sum(dim=0)
                y_result[i * SHARD_SIZE: (i + 1) * SHARD_SIZE, ...] = restricted_y
        
        # put restricted y back to w
        y_result = torch.t(y_result)
        for i in range(len(w)):
            w[i][k] = y_result[i].reshape(w[i][k].shape).to(device)
    
        # opinion_matrix_sum
        if opinion_matrix_sum is None:
            opinion_matrix_sum = opinion_matrix
        else:
            opinion_matrix_sum = torch.cat((opinion_matrix_sum,opinion_matrix),0)
        
    reweight_sum = reweight_sum / reweight_sum.max()
    reweight_sum = reweight_sum * reweight_sum

    rep_score = reputation_model(w, opinion_matrix_sum, kappa=0.3, W=2, a=0.5)
    scaler = MinMaxScaler()
    normalized_rep_score = scaler.fit_transform(np.array(rep_score).reshape(-1,1))
    for i in range(len(normalized_rep_score)):
        if normalized_rep_score[i] < 1/len(normalized_rep_score):
            normalized_rep_score[i] = 0
    normalized_rep_score = torch.tensor(normalized_rep_score.reshape(1,-1)[0])
    print('reputation after normalized', normalized_rep_score)
    
    return normalized_rep_score

def is_valid_model(w):
    if isinstance(w, list):
        w_keys = list(range(len(w)))
    else:
        w_keys = w.keys()
    for k in w_keys:
        params = w[k]
        if torch.isnan(params).any():
            return False
        if torch.isinf(params).any():
            return False
    return True

    
def get_valid_models(w_locals):
    w, invalid_model_idx = [], []
    for i in range(len(w_locals)):
        if is_valid_model(w_locals[i]):
            w.append(w_locals[i])
        else:
            invalid_model_idx.append(i)
    return w, invalid_model_idx
    
def median_opt(input):
    shape = input.shape
    input = input.sort()[0]
    if shape[-1] % 2 != 0:
        output = input[..., int((shape[-1] - 1) / 2)]
    else:
        output = (input[..., int(shape[-1] / 2 - 1)] + input[..., int(shape[-1] / 2)]) / 2.0
    return output  

def repeated_median(y):
    num_models = y.shape[1]
    total_num = y.shape[0]
    y = y.sort()[0]
    yyj = y.repeat(1, 1, num_models).reshape(total_num, num_models, num_models)
    yyi = yyj.transpose(-1, -2)
    xx = torch.FloatTensor(range(num_models)).to(y.device)
    xxj = xx.repeat(total_num, num_models, 1)
    xxi = xxj.transpose(-1, -2) + eps

    diag = torch.Tensor([float('Inf')] * num_models).to(y.device)
    diag = torch.diag(diag).repeat(total_num, 1, 1)

    dividor = xxi - xxj + diag
    slopes = (yyi - yyj) / dividor + diag
    slopes, _ = slopes.sort()
    slopes = median_opt(slopes[:, :, :-1])
    slopes = median_opt(slopes)

    # get intercepts (intercept of median)
    yy_median = median_opt(y)
    xx_median = [(num_models - 1) / 2.0] * total_num
    xx_median = torch.Tensor(xx_median).to(y.device)
    intercepts = yy_median - slopes * xx_median

    return slopes, intercepts
    
