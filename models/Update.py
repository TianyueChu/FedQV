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
import random
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer

eps = np.finfo(float).eps


def aggregate(args, w_local, n, w_global, vote_budget, global_acc, agg_weight):
    if args.agg == 'fedavg':
        print("using FedAvg")
        w_avg = FedAvg(w_local, n)
    elif args.agg == 'fedvoting':
        print("using FedVoting")
        w_avg = FedVoting(w_local, n)
    elif args.agg == 'fedq':
        print("using FedQ Estimator")
        w_avg = FedQ(w_local, n)
    elif args.agg == 'fedqv':
        print("using FedQV Estimator")
        w_avg, vote_budget = FedQV(w_local, w_global, vote_budget, n, global_acc, agg_weight, args.rep, args.theta)
    elif args.agg == 'fedcv':
        print("using FedCV Estimator")
        w_avg, vote_budget = FedCV(w_local, w_global, vote_budget, n, global_acc, agg_weight, args.rep, args.theta)
    elif args.agg == 'krum':
        print("using Krum Estimator")
        w_avg = Krum(w_local, w_global)
    # elif args.agg == 'trimmed-mean':
    # print("using trimmed mean Estimator")
    # w_avg = trimmed_mean(w_local, trim_ratio = args.num_attackers/args.num_users )
    elif args.agg == 'multi-krum':
        print("using Multi-Krum Estimator")
        w_avg, vote_budget = Mkrum(w_local, w_global, vote_budget, n, args.qv, args.rep, global_acc, agg_weight,
                                   b=0.1 * args.num_users * args.frac)
    elif args.agg == 'median':
        print("using median Estimator")
        w_avg = Median(w_local)
    elif args.agg == "trimmed-mean":
        print("using trimmed mean Estimator")
        w_avg, vote_budget = Tr_mean(w_local, w_global, vote_budget, n, args, agg_weight, global_acc)
    elif args.agg == "rep":
        print("using reputation Estimator")
        w_avg = Rep(w_local, w_global)
    else:
        exit('Error: unrecognized aggregation method')
    return w_avg, vote_budget


def euclid(v1, v2):
    diff = v1 - v2
    return torch.matmul(diff, diff.T)


def multi_vectorization(w_locals, args):
    vectors = copy.deepcopy(w_locals)

    for i, v in enumerate(vectors):
        for name in v:
            v[name] = v[name].reshape([-1]).to(args.device)
        vectors[i] = torch.cat(list(v.values()))

    return vectors


def pairwise_distance(w_locals, args):
    vectors = multi_vectorization(w_locals, args)
    distance = torch.zeros([len(vectors), len(vectors)]).to(args.device)

    for i, v_i in enumerate(vectors):
        for j, v_j in enumerate(vectors[i:]):
            distance[i][j + i] = distance[j + i][i] = euclid(v_i, v_j)

    return distance


def FedAvg(w, n):
    w_avg = copy.deepcopy(w[0])

    agg_weight = []
    for i in range(len(w)):
        agg_weight.append(n[i] / sum(n))
    print('aggregation weights are', agg_weight)

    for k in w_avg.keys():
        w_avg[k] = torch.mul(w[0][k], n[0] / sum(n))
        for i in range(1, len(w)):
            # w_avg[k] += torch.mul(w[i][k], n[i] / sum(n))
            result = torch.mul(w[i][k], n[i] / sum(n))
            if result.shape == torch.Size([1]):
                result = result.view(w_avg[k].shape)
            w_avg[k] += result

    return w_avg


def FedVoting(w, n):
    # Finding the index of the largest value in n
    max_index = n.index(max(n))

    # Initialize w_avg with the weights corresponding to the index with the largest n
    w_avg = copy.deepcopy(w[max_index])

    return w_avg

def Krum(w, w_global):
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
                d = np.sqrt(np.sum(np.power(grads[i] - grads[j], 2)))
                dists.append(d)
        sorted_index = torch.argsort(torch.tensor(dists), descending=False)
        dists = torch.tensor(dists)
        scores.append(dists[sorted_index[0]].sum())
    ind_min = scores.index(min(scores))
    print('using updates from voter index ', ind_min)
    return w[ind_min]


def Mkrum(w, w_global, vote_budget, n, qv_on, reputation_on, global_acc, agg_weight, b):
    num_clients = len(w)

    # assert n >= 2 * b + 3, "Krum requirement: n >= 2b + 3."
    num_selection = round(max(num_clients - b - 2, 1))
    print("Selecting {} clients".format(num_selection))
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

    for i in range(num_clients - 1):
        dists = []
        for j in range(i + 1, num_clients):
            if j != i:
                d = np.sqrt(np.sum(np.power(grads[i] - grads[j], 2)))
                dists.append(d)
        sorted_index = torch.argsort(torch.tensor(dists), descending=False)
        # print(sorted_index)
        dists = torch.tensor(dists)
        for k in range(len(sorted_index)):
            ind_dict[str(i) + ',' + str(sorted_index[k].numpy() + i + 1)] = dists[sorted_index[k]]
            scores.append(dists[sorted_index[k]])

    res = torch.sort(torch.tensor(scores), descending=False)[:num_selection]

    keys = list(ind_dict.keys())
    values = list(ind_dict.values())
    ind_list = []

    if all(value == torch.tensor(float('nan'), dtype=torch.float64) for value in values):
        # Handle the case when all values are nan
        client_list = list(range(1, num_clients + 1))
        # Randomly choose num_selection clients from the list
        final_list = random.sample(client_list, num_selection)
    else:
        for i in res[0][0:num_selection + 1].numpy():

            ind_list = ind_list + list(map(int, keys[values.index(i)].rsplit(',')))

        select_list = []
        [select_list.append(v) for v in ind_list if v not in select_list]
        final_list = select_list[:num_selection]
        final_list.sort()
        print('worker index', final_list)

    #### aggregation part
    ### with QV
    # if qv_on:
    #     print("Plugging FedQV")
    #     w_qv = [w[i] for i in final_list]
    #     vote_budget_qv = [vote_budget[i] for i in final_list]
    #     n_qv = [n[i] for i in final_list]
    #     w_avg, vote_budget_qv = FedQV(w_qv, w_global, vote_budget_qv, n_qv, global_acc, agg_weight,
    #                                   reputation_on=reputation_on, theta=0)
    #     for idx, i in enumerate(final_list):
    #         vote_budget[i] = vote_budget_qv[idx]
    #
    if qv_on:
        print("Plugging FedQV")
        w_qv = [w[i] for i in final_list]
        vote_budget_qv = [vote_budget[i] for i in final_list]
        # n_qv = [n[i] for i in chosen]
        w_agg_qv = [agg_weight[i] for i in final_list]
        w_avg, vote_budget_qv = QV_plugin(w_qv, w_global, vote_budget_qv, qv_on, w_agg_qv, 0.1, global_acc)
        for idx, i in enumerate(final_list):
            vote_budget[i] = vote_budget_qv[idx]

    else:
        w_avg = copy.deepcopy(w[final_list[0]])
        for k in w_avg.keys():
            w_avg[k] = torch.mul(w[final_list[0]][k], 1 / len(final_list))
            for i in final_list[1:]:
                w_avg[k] += torch.mul(w[i][k], 1 / len(final_list))

    return w_avg, vote_budget


def Tr_mean(w, w_global, vote_budget, n, args, agg_weight, acc):
    b = int(0.1 * args.num_users * args.frac)
    n = len(w) - 2 * b

    distance = pairwise_distance(w, args)

    distance = distance.sum(dim=1)
    med = distance.median()
    _, chosen = torch.sort(abs(distance - med))
    chosen = chosen[: n]

    if args.qv:
        print("Plugging FedQV")
        w_qv = [w[i] for i in chosen]
        vote_budget_qv = [vote_budget[i] for i in chosen]
        # n_qv = [n[i] for i in chosen]
        w_agg_qv = [agg_weight[i] for i in chosen]
        w_avg, vote_budget_qv = QV_plugin(w_qv, w_global, vote_budget_qv, args.rep, w_agg_qv, 0.1, acc)
        for idx, i in enumerate(chosen):
            vote_budget[i] = vote_budget_qv[idx]

    else:
        w_avg = copy.deepcopy(w[int(chosen[0])])
        for k in w_avg.keys():
            w_avg[k] = torch.mul(w[int(chosen[0])][k], 1 / len(chosen))
            for i in chosen[1:]:
                w_avg[k] += torch.mul(w[i][k], 1 / len(chosen))

    return w_avg, vote_budget


def FedQ(w, n):
    w_avg = copy.deepcopy(w[0])
    sum_weight = sum(np.sqrt(n))
    # k: layer in NN
    for k in w_avg.keys():
        w_avg[k] = torch.mul(w[0][k], math.sqrt(n[0]) / sum_weight)
        # len(w): num_clients
        for i in range(1, len(w)):
            w_avg[k] += torch.mul(w[i][k], math.sqrt(n[i]) / sum_weight)
    return w_avg


def FedQV_adaptive(w, w_global, n):
    w, invalid_model_idx = get_valid_models(w)
    num_clients = len(w)

    if (len(invalid_model_idx) == num_clients) or (len(w) == 0):
        w_avg = copy.deepcopy(w_global)
        for k in w_avg.keys():
            w_avg[k] = w_global[k]
            print("no valid model")
    else:

        w_avg = copy.deepcopy(w[0])
        glob_grad = []

        for j in list(w_global.values()):
            glob_grad = np.append(glob_grad, np.array(j.tolist()).flatten())

        grad_len = len(glob_grad)
        grads = np.zeros((num_clients, grad_len))
        weight = []

        for i in range(num_clients):
            layer = []
            for j in list(w[i].values()):
                layer = np.append(layer, np.array(j.tolist()).flatten())

            # grads[i] contains all the grads of client[i]
            grads[i] = layer
            sim_score = smp.cosine_similarity(grads[i].reshape(1, -1), glob_grad.reshape(1, -1)).flatten().tolist()[-1]
            weight.append(sim_score)

        # Normalization
        scaler = MinMaxScaler()
        normalized_weight = scaler.fit_transform(np.array(weight).reshape(-1, 1))

        # Change the weights to media
        # median_value = np.median(normalized_weight )

        # Create a new NumPy array with the same size as 'weight' and fill it with the median value
        # median_array = np.full_like(weight, median_value)

        voice_credit = []

        for i in range(len(normalized_weight)):
            if normalized_weight[i][0] in [0]:
                voice_credit.append(1)
            else:
                voice_credit.append(-math.log(normalized_weight[i][0]) + 1)

        # calculate sum of the voice credits
        voice_credit = [a * b for a, b in zip(voice_credit, n)]
        voice_credit = [i if i >= 0 else 0 for i in voice_credit]
        sum_weight = sum(np.sqrt(voice_credit))

        agg_weight = []
        for i in range(len(voice_credit)):
            agg_weight.append(math.sqrt(voice_credit[i]) / sum_weight)

        if np.isnan(agg_weight).any():
            for k in w_avg.keys():
                w_avg[k] = w_global[k]
            print("using the global model")
        else:
            print('aggregation weights are', agg_weight)
            # k: layer in NN
            for k in w_avg.keys():
                w_avg[k] = torch.mul(w[0][k], math.sqrt(voice_credit[0]) / sum_weight)
                # len(w): num_clients
                for i in range(1, len(w)):
                    w_avg[k] += torch.mul(w[i][k], math.sqrt(voice_credit[i]) / sum_weight)

    return w_avg


def FedQV(w, w_global, vote_budget, n, acc, agg_w, reputation_on, theta):

    w, invalid_model_idx = get_valid_models(w)
    # print("invalid model",invalid_model_idx)

    num_clients = len(w)

    if len(invalid_model_idx) == num_clients:
        w_avg = copy.deepcopy(w_global)
        for k in w_avg.keys():
            w_avg[k] = w_global[k]
            print("no valid model")
    else:
        w_avg = copy.deepcopy(w[0])
        for i in sorted(invalid_model_idx, reverse=True):
            del vote_budget[i]

        # Normalization
        scaler = MinMaxScaler()

        normalized_weight = scaler.fit_transform(np.array(agg_w).reshape(-1, 1))
        print("normalized_weight", normalized_weight)

        # voice credit
        voice_credit = []
        for i in range(len(normalized_weight)):
            if normalized_weight[i][0] <= theta:
                voice_credit.append(0)
            elif normalized_weight[i][0] >= 1 - theta:
                voice_credit.append(0)
            else:
                voice_credit.append(-math.log(normalized_weight[i][0]) + 1)

        print("voice_credit are", voice_credit)

        # Reputation model
        if reputation_on:
            w, rep_score = reputation_aggregation(w, LAMBDA=2, thresh=0.05)
            for i in range(len(vote_budget)):
                if rep_score[i].cpu().numpy() > 0.9:
                    vote_budget[i] += rep_score[i].cpu().numpy()
                    voice_credit[i] += rep_score[i].cpu().numpy()
                    print('index', i, 'increase budget', rep_score[i].cpu().numpy())
                elif rep_score[i].cpu().numpy() == 0:
                    vote_budget[i] = 0
                    voice_credit[i] = 0
                    # print('voter',i,'increase budget',rep_score[i])    
        # voice credit budget

        for i in range(len(vote_budget)):
            # budget already run out
            if vote_budget[i] == 0:
                voice_credit[i] = 0
            # vote budget run out this time
            elif vote_budget[i] - voice_credit[i] <= 0:
                voice_credit[i] = vote_budget[i]
                vote_budget[i] = 0
            elif vote_budget[i] - voice_credit[i] > 0:
                vote_budget[i] = vote_budget[i] - voice_credit[i]
            else:
                print('Error')

        print("voice_credit are", voice_credit)

        # calculate sum of the voice credits
        voice_credit = [a * b for a, b in zip(voice_credit, n)]
        voice_credit = [i if i >= 0 else 0 for i in voice_credit]
        sum_weight = sum(np.sqrt(voice_credit))

        agg_weight = []
        if sum_weight != 0:
            for i in range(len(voice_credit)):
                agg_weight.append(math.sqrt(voice_credit[i]) / sum_weight)
        else:
            # check what happen to sum_weight
            for i in range(len(voice_credit)):
                agg_weight.append(1/num_clients)

        if np.isnan(agg_weight).any():
            for k in w_avg.keys():
                w_avg[k] = w_global[k]
            print("using the global model")
        else:
            print('aggregation weights are', agg_weight)
            # k: layer in NN
            for k in w_avg.keys():
                w_avg[k] = torch.mul(w[0][k], agg_weight[0])
                # len(w): num_clients
                for i in range(1, len(w)):
                    # w_avg[k] += torch.mul(w[i][k], agg_weight[i])
                    result = torch.mul(w[i][k], agg_weight[i])
                    if result.shape == torch.Size([1]):
                        result = result.view(w_avg[k].shape)
                    w_avg[k] += result

        for i in sorted(invalid_model_idx, reverse=False):
            vote_budget.insert(i, 0)

    return w_avg, vote_budget


def FedCV(w, w_global, vote_budget, n, acc, agg_w, reputation_on, theta):
    w, invalid_model_idx = get_valid_models(w)

    num_clients = len(w)

    if len(invalid_model_idx) == num_clients:
        w_avg = copy.deepcopy(w_global)
        for k in w_avg.keys():
            w_avg[k] = w_global[k]
            print("no valid model")
    else:
        w_avg = copy.deepcopy(w[0])
        for i in sorted(invalid_model_idx, reverse=True):
            del vote_budget[i]

        # Normalization
        scaler = MinMaxScaler()

        normalized_weight = scaler.fit_transform(np.array(agg_w).reshape(-1, 1))
        print("normalized_weight", normalized_weight)

        # voice credit
        voice_credit = [] 
        
        for i in range(len(normalized_weight)):
            if normalized_weight[i][0] <= theta:
                voice_credit.append(0)
            elif normalized_weight[i][0] >= 1 - theta:
                voice_credit.append(0)
            else:
                voice_credit.append(-math.log(normalized_weight[i][0]) + 1)    

        print("voice_credit are", voice_credit)

        # Reputation model
        if reputation_on:
            w, rep_score = reputation_aggregation(w, LAMBDA=2, thresh=0.05)
            for i in range(len(vote_budget)):
                if rep_score[i].cpu().numpy() > 0.9:
                    vote_budget[i] += rep_score[i].cpu().numpy()
                    voice_credit[i] += rep_score[i].cpu().numpy()
                    print('index', i, 'increase budget', rep_score[i].cpu().numpy())
                elif rep_score[i].cpu().numpy() == 0:
                    vote_budget[i] = 0
                    voice_credit[i] = 0
                    # print('voter',i,'increase budget',rep_score[i])
        
        # voice credit budget
        for i in range(len(vote_budget)):
            # budget already run out
            if vote_budget[i] == 0:
                voice_credit[i] = 0
            # vote budget run out this time
            elif vote_budget[i] - voice_credit[i] <= 0:
                voice_credit[i] = vote_budget[i]
                vote_budget[i] = 0
            elif vote_budget[i] - voice_credit[i] > 0:
                vote_budget[i] = vote_budget[i] - voice_credit[i]
            else:
                print('Error')

        print("voice_credit are", voice_credit)

        # calculate sum of the voice credits
        voice_credit = [a * b for a, b in zip(voice_credit, n)]
        voice_credit = [i if i >= 0 else 0 for i in voice_credit]
        sum_weight = sum(np.cbrt(voice_credit))

        agg_weight = []
        if sum_weight != 0:
            for i in range(len(voice_credit)):
                agg_weight.append(math.pow(voice_credit[i], 1/3)/sum_weight)
        else:
            # check what happen to sum_weight
            for i in range(len(voice_credit)):
                agg_weight.append(1/num_clients)

        if np.isnan(agg_weight).any():
            for k in w_avg.keys():
                w_avg[k] = w_global[k]
            print("using the global model")
        else:
            print('aggregation weights are', agg_weight)
            # k: layer in NN
            for k in w_avg.keys():
                w_avg[k] = torch.mul(w[0][k], agg_weight[0])
                # len(w): num_clients
                for i in range(1, len(w)):
                    # w_avg[k] += torch.mul(w[i][k], agg_weight[i])
                    result = torch.mul(w[i][k], agg_weight[i])
                    if result.shape == torch.Size([1]):
                        result = result.view(w_avg[k].shape)
                    w_avg[k] += result

        for i in sorted(invalid_model_idx, reverse=False):
            vote_budget.insert(i, 0)

    return w_avg, vote_budget


def QV_plugin(w, w_global, vote_budget, reputation_on, agg_w, theta, acc):
    # type(w): list
    # type(w_global): OrderedDict
    # type(w[0]): OrderedDict
    # len(w[0]): how many layers in NN]

    w, invalid_model_idx = get_valid_models(w)
    # print("invalid model",invalid_model_idx)

    num_clients = len(w)

    if len(invalid_model_idx) == num_clients:
        w_avg = copy.deepcopy(w_global)
        for k in w_avg.keys():
            w_avg[k] = w_global[k]
            print("no valid model")
    else:
        w_avg = copy.deepcopy(w[0])
        for i in sorted(invalid_model_idx, reverse=True):
            del vote_budget[i]

        # Normalization
        scaler = MinMaxScaler()
        normalized_weight = scaler.fit_transform(np.array(agg_w).reshape(-1, 1))
        print(normalized_weight)

        # theta = 0
        # voice credit
        voice_credit = []
      
        for i in range(len(normalized_weight)):
            if normalized_weight[i][0] <= theta:
                voice_credit.append(0)
            elif normalized_weight[i][0] >= 1 - theta:
                voice_credit.append(0)
            else:
                voice_credit.append(-math.log(normalized_weight[i][0]) + 1)

        # Reputation model
        if reputation_on:
            w, rep_score = reputation_aggregation(w, LAMBDA=2, thresh=0.5)
            for i in range(len(vote_budget)):
                if rep_score[i].cpu().numpy() < 0.9:
                    vote_budget[i] += rep_score[i].cpu().numpy()
                    voice_credit[i] += rep_score[i].cpu().numpy()
                    print('index', i, 'increase budget', rep_score[i].cpu().numpy())
                elif rep_score[i].cpu().numpy() == 0:
                    vote_budget[i] = 0
                    voice_credit[i] = 0
                    # print('voter',i,'increase budget',rep_score[i])
        # voice credit budget

        for i in range(len(vote_budget)):
            # budget already run out
            if vote_budget[i] == 0:
                voice_credit[i] = 0
            # vote budget run out this time
            elif vote_budget[i] - voice_credit[i] <= 0:
                voice_credit[i] = vote_budget[i]
                vote_budget[i] = 0
            elif vote_budget[i] - voice_credit[i] > 0:
                vote_budget[i] = vote_budget[i] - voice_credit[i]
            else:
                print('Error')

        # caculate sum of the voice credits
        # voice_credit = [a * b for a, b in zip(voice_credit, n)]
        voice_credit = [i if i >= 0 else 0 for i in voice_credit]
        sum_weight = sum(np.sqrt(voice_credit))
        # print("voice_credit are", voice_credit)

        agg_weight = []
        if sum_weight != 0:
            for i in range(len(voice_credit)):
                agg_weight.append(math.sqrt(voice_credit[i]) / sum_weight)
        else:
            # check what happen to sum_weight
            for i in range(len(voice_credit)):
                agg_weight.append(1 / len(voice_credit))

        if np.isnan(agg_weight).any():
            for k in w_avg.keys():
                w_avg[k] = w_global[k]
            print("using the global model")
        else:
            print('aggregation weights are', agg_weight)
            # k: layer in NN
            for k in w_avg.keys():
                w_avg[k] = torch.mul(w[0][k], agg_weight[0])
                # len(w): num_clients
                for i in range(1, len(w)):
                    w_avg[k] += torch.mul(w[i][k],  agg_weight[i])

        for i in sorted(invalid_model_idx, reverse=False):
            vote_budget.insert(i, 0)

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
        belief = (belief_count * kappa) / (belief_count * kappa + disbelief_count * (1 - kappa) + W)
        uncertainty = W / (belief_count * kappa + disbelief_count * (1 - kappa) + W)
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
    # opinion_matrix = (reweight < thresh)
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
                reweight, restricted_y, opinion_matrix = reweight_algorithm_restricted(y, LAMBDA, thresh)
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
            opinion_matrix_sum = torch.cat((opinion_matrix_sum, opinion_matrix), 0)

    reweight_sum = reweight_sum / reweight_sum.max()
    reweight_sum = reweight_sum * reweight_sum

    rep_score = reputation_model(w, opinion_matrix_sum, kappa=0.3, W=2, a=0.5)
    scaler = MinMaxScaler()
    normalized_rep_score = scaler.fit_transform(np.array(rep_score).reshape(-1, 1))
    for i in range(len(normalized_rep_score)):
        if normalized_rep_score[i] < 1 / len(normalized_rep_score):
            normalized_rep_score[i] = 0
    normalized_rep_score = torch.tensor(normalized_rep_score.reshape(1, -1)[0])
    print('reputation after normalized', normalized_rep_score)

    return w, normalized_rep_score


def Rep(w_locals, w_global, LAMBDA=2, thresh=0.05):
    SHARD_SIZE = 2000
    w, invalid_model_idx = get_valid_models(w_locals)
    if len(w) > 0:
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
                reweight, restricted_y, opinion_matrix = reweight_algorithm_restricted(transposed_y_list, LAMBDA,
                                                                                       thresh)
                reweight_sum += reweight.sum(dim=0)
                y_result = restricted_y
            else:
                num_shards = int(math.ceil(total_num / SHARD_SIZE))
                for i in range(num_shards):
                    y = transposed_y_list[i * SHARD_SIZE: (i + 1) * SHARD_SIZE, ...]
                    reweight, restricted_y, opinion_matrix = reweight_algorithm_restricted(y, LAMBDA, thresh)
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
                opinion_matrix_sum = torch.cat((opinion_matrix_sum, opinion_matrix), 0)

        reweight_sum = reweight_sum / reweight_sum.max()
        reweight_sum = reweight_sum * reweight_sum

        rep_score = reputation_model(w, opinion_matrix_sum, kappa=0.5, W=2, a=0.1)
        scaler = MinMaxScaler()
        normalized_rep_score = scaler.fit_transform(np.array(rep_score).reshape(-1, 1))
        # for i in range(len(normalized_rep_score)):
        # if normalized_rep_score[i] < 1 / len(normalized_rep_score):
        # normalized_rep_score[i] = 0
        normalized_rep_score = torch.tensor(normalized_rep_score.reshape(1, -1)[0])
        print('reputation after normalized', normalized_rep_score)

        sum_weight = sum(normalized_rep_score)

        w_avg = copy.deepcopy(w[0])
        for k in w_avg.keys():
            w_avg[k] = torch.mul(w[0][k], (normalized_rep_score[0]) / sum_weight)
            # len(w): num_clients
            for i in range(1, len(w)):
                w_avg[k] += torch.mul(w[i][k], (normalized_rep_score[i]) / sum_weight)
    else:
        w_avg = copy.deepcopy(w_global)
        for k in w_avg.keys():
            w_avg[k] = w_global[k]
            print("no valid model")

    return w_avg


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


def get_valid_model(w_local):
    invalid_model_idx = []
    if not is_valid_model(w_local):
        invalid_model_idx.append(0)
    return invalid_model_idx


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


def trimmed_mean(w, trim_ratio):
    trim_num = int(trim_ratio * len(w))
    device = w[0][list(w[0].keys())[0]].device
    w_med = copy.deepcopy(w[0])
    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda x, y: x * y, shape)
        y_list = torch.FloatTensor(len(w), total_num).to(device)
        for i in range(len(w)):
            y_list[i] = torch.reshape(w[i][k], (-1,))
        y = torch.t(y_list)
        y_sorted = y.sort()[0]
        result = y_sorted[:, trim_num:-trim_num]
        result = result.mean(dim=-1)
        assert total_num == len(result)

        weight = torch.reshape(result, shape)
        w_med[k] = weight
    return w_med


def Median(w):
    device = w[0][list(w[0].keys())[0]].device
    w_med = copy.deepcopy(w[0])
    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda x, y: x * y, shape)
        y_list = torch.FloatTensor(len(w), total_num).to(device)
        for i in range(len(w)):
            y_list[i] = torch.reshape(w[i][k], (-1,))
        y = torch.t(y_list)
        median_result = median_opt(y)
        assert total_num == len(median_result)

        weight = torch.reshape(median_result, shape)
        w_med[k] = weight
    return w_med
