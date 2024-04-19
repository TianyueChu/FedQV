#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import os
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
from copy import deepcopy

import numpy as np
from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler

from utils.sampling import mnist_iid, noniid, cifar_iid, femnist_noniid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import build_model
from models.Fed import aggregate, Krum, trimmed_mean, Median, weighted_average, FedQV_adaptive, get_valid_model
from models.test import test_img
from vision.load import get_data
import warnings
import pickle
from vision.sampler import FLSampler
import sklearn.metrics.pairwise as smp
import math

from attack_gradients.DLG_reconstruct import DLG_GradReconstructor
from utils.save_load import mkdir_save, load


def l2_distance(dict1, dict2):
    squared_sum = 0
    for key in dict1:
        squared_sum += torch.norm(dict1[key] - dict2[key], p=2)
    return squared_sum


def save_data_to_file(data, folder, filename_format, format_args):
    # Create the complete file path
    file_path = os.path.join(folder, filename_format.format(format_args))

    # Save the data to the specified file using pickle
    with open(file_path, "wb") as opened_f:
        pickle.dump(data, opened_f)


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # Define the save path
    save_path = "/home/moonbot/PycharmProjects/FedQV_nips/results/{}/{}/{}_{}_{}".format(args.dataset, args.agg,
                                                                                         args.attack_type,
                                                                                         args.num_attackers, args.times)
    os.makedirs(save_path, exist_ok=True)

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
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            print("iid setting in CIFAR")
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            print("Non-iid setting in CIFAR")
            dict_users = noniid(dataset_train, args.num_users)

    elif args.dataset == 'cifar100':
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
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

    elif args.dataset == 'femnist':
        dataset_train = get_data(args.dataset, data_type="train", transform=None, target_transform=None, user_list=None)
        dataset_test = get_data(args.dataset, data_type="test", transform=None, target_transform=None, user_list=None)

        print("Non-iid setting in FEMNIST")
        dict_users = femnist_noniid(dataset_train, args.num_users)
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
    asr_all = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    vote_budget = [args.budget] * args.num_users
    global_acc = 1
    np.random.seed = args.seed

    # the training round is under attack
    if args.num_attackers != 0:
        # toy model
        # idxs_attackers = [2]
        idxs_attackers = np.random.choice(range(args.num_users), args.num_attackers, replace=False)
    else:
        idxs_attackers = []
    print(idxs_attackers, "are attackers")

    # parites paticipant in this round
    for iter in range(args.epochs):

        loss_locals = []
        num_data_all = []

        if not args.all_clients:
            w_locals = []
            num_samples = []
        else:
            print("Aggregation over all clients")
            w_locals = [w_glob for i in range(args.num_users)]

        m = max(int(args.frac * args.num_users), 1)

        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        # toy model
        # idxs_users = np.arange(args.num_users)
        print(idxs_users, "in this round")

        # budget of parties participate in this epoch
        vote_budget_epoch = []
        attack_ind = []
        attack_sample = []
        # sim_scores = []

        # aggregation weights for FedQV
        agg_weights = []

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

            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], attack_mode=attack_mode)
            w, loss, num_data = local.train(net=copy.deepcopy(net_glob).to(args.device))

            if args.invertGrad and (idx == idxs_users[0]) and (iter == 0):
                local.save_user_traindata(save_path, iter, idxs_users[0])
                print(f"Ground truth of user {idxs_users[0]} are saved.")

            # num_data_all.append(num_data)

            # calculate similarity score here

            if attack_mode:
                if args.attack_type == 'gaussian_attack':
                    print("Conducting gaussian attack")
                    for k in w.keys():
                        mean = 0
                        std = 1
                        noisy = std * torch.randn(w[k].size()) + mean
                        w[k] += noisy.to('cuda:0').long()
                elif args.attack_type == 'scaling_attack':
                    print("Conducting scaling attack")
                    if iter == args.epochs - 1:
                        for k in w.keys():
                            w[k] = min(args.frac * args.num_users, 10) * w[k]
                elif ((args.attack_type == 'krum_attack') or (args.attack_type == 'trimmed-mean_attack') or
                      (args.attack_type == 'median_attack') or (args.attack_type == 'neurotoxin') or
                      (args.attack_type == "min_sum") or (args.attack_type == "min_max")):
                    attack_ind.append(idxs_users.tolist().index(idx))
                elif args.attack_type == 'qv_adaptive':
                    attack_ind.append(idxs_users.tolist().index(idx))
                    attack_sample.append(len(dict_users[idx]))
                elif args.attack_type == 'backdoor':
                    print("Conducting backdoor attack")
                elif args.attack_type == 'labelflip':
                    print("Conducting labelflip attack")

            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
                num_samples.append(len(dict_users[idx]))

            loss_locals.append(copy.deepcopy(loss))

            # FedQV
            if (args.agg == "fedqv") or (args.qv == True) or (args.agg == "fedcv"):
                invalid_model_idx = get_valid_model(w)

                if len(invalid_model_idx) != 0:
                    print("invalid model")
                    agg_weights.append(0)
                else:
                    glob_grad = []

                    for j in list(w_glob.values()):
                        # glob_grad.append(np.array(j.tolist()).flatten())
                        glob_grad.append(j.view(-1))
                    flattened_weights = torch.cat(glob_grad, dim=0)

                    layer = []
                    for j in list(w.values()):
                        # layer.append(np.array(j.tolist()).flatten())
                        layer.append(j.view(-1))

                    flattened_layer = torch.cat(layer, dim=0)

                    # Calculate the cosine similarity between the flattened weights and the flattened layer
                    sim_score = F.cosine_similarity(flattened_weights, flattened_layer, dim=0).item()
                    agg_weights.append(sim_score)

        if args.attack_type == 'krum_attack':
            print("Conducting krum attack")
            w_krum = Krum(w_locals, w_glob)
            for k in w_krum.keys():
                w_krum[k] = - w_krum[k]

            for idx in attack_ind:
                for k in w_krum.keys():
                    w_locals[idx][k] = w_krum[k]

        if args.attack_type == 'trimmed-mean_attack':
            print("Conducting trimmed-mean attack")
            w_trim = trimmed_mean(w_locals, 0.4)
            for k in w_trim.keys():
                w_trim[k] = - w_trim[k]

            for idx in attack_ind:
                for k in w_trim.keys():
                    w_locals[idx][k] = w_trim[k]

        if args.attack_type == 'median_attack':
            print("Conducting median attack")
            w_med = Median(w_locals)
            for k in w_med.keys():
                w_med[k] = - w_med[k]

            for idx in attack_ind:
                for k in w_med.keys():
                    w_locals[idx][k] = w_med[k]

        if args.attack_type == "neurotoxin":
            print("Conducting Neurotoxin attack")
            ratio = 0.1
            for idx in attack_ind:
                for k in w_glob.keys():
                    gradients = w_glob[k].abs().view(-1)
                    gradients_length = len(gradients)
                    values, indices = torch.topk(gradients, int(gradients_length * ratio))
                    mask_flat = torch.zeros(gradients_length)
                    mask_flat[indices.cpu()] = 1.0
                    mask = mask_flat.reshape(w_glob[k].size()).cuda()
                    w_locals[idx][k] = mask * w_glob[k] + (1 - mask) * w_locals[idx][k]

        if args.attack_type == "min_max":
            if len(attack_ind) >= 0:
                print("Conducting Min-max attack")
                lamda = torch.Tensor([50.0]).float().cuda()
                # print(lamda)
                threshold_diff = 1e-5
                lamda_fail = lamda
                lamda_suc = 0

                # get the average updates
                w_attack = [w_glob]
                attack_weights = [1]
                for idx in attack_ind:
                    w_attack.append(w_locals[idx])
                    attack_weights.append(1)

                w_attack_avg, _ = weighted_average(w_attack, np.array(attack_weights))

                # Calculate L2 distances between each pair of dictionaries
                distances = []

                for i in range(len(w_attack)):
                    distance = 0
                    dict1 = w_attack[i]
                    for j in range(i + 1, len(w_attack)):
                        dict2 = w_attack[j]
                        for key in dict1.keys():
                            distance += torch.norm(dict1[key].to(torch.float32) - dict2[key].to(torch.float32), p=2)
                    distances.append(distance)

                # Find the largest distance
                max_distance = max(distances)
                del distances

                # Calculate the malicious updates

                while torch.abs(lamda_suc - lamda) > threshold_diff:
                    mal_update = copy.deepcopy(w_attack_avg)
                    for k in w_attack_avg.keys():
                        deviation = torch.sign(w_attack_avg[k])
                        mal_update[k] = (w_attack_avg[k] - lamda * deviation)

                    distances = []

                    for j in w_attack:
                        distance = 0
                        for key in j.keys():
                            distance += torch.norm(j[key] - mal_update[key], p=2)
                        distances.append(distance)

                    max_d = max(distances)

                    if max_d <= max_distance:
                        print('successful lamda is ', lamda)
                        lamda_suc = lamda
                        lamda = lamda + lamda_fail / 2
                    else:
                        lamda = lamda - lamda_fail / 2

                    lamda_fail = lamda_fail / 2

                mal_update = copy.deepcopy(w_attack_avg)
                for k in w_attack_avg.keys():
                    deviation = torch.sign(w_attack_avg[k])
                    mal_update[k] = (w_attack_avg[k] - lamda_suc * deviation)

                for idx in attack_ind:
                    for k in mal_update.keys():
                        w_locals[idx][k] = mal_update[k]

            else:
                print("No attacker in this round")

        if args.attack_type == "min_sum":
            if len(attack_ind) >= 0:
                print("Conducting Min-sum attack")
                lamda = torch.Tensor([50.0]).float().cuda()
                # print(lamda)
                threshold_diff = 1e-5
                lamda_fail = lamda
                lamda_suc = 0

                # get the average updates
                w_attack = [w_glob]
                attack_weights = [1]
                for idx in attack_ind:
                    w_attack.append(w_locals[idx])
                    attack_weights.append(1)

                w_attack_avg, _ = weighted_average(w_attack, np.array(attack_weights))

                # Calculate L2 distances between each pair of dictionaries
                distances = []

                for i in range(len(w_attack)):
                    distance = 0
                    dict1 = w_attack[i]
                    for j in range(len(w_attack)):
                        dict2 = w_attack[j]
                        for key in dict1.keys():
                            distance += torch.norm(dict1[key].to(torch.float32) - dict2[key].to(torch.float32), p=2)
                    distances.append(distance)

                # Find the smallest distance
                min_score = min(distances)
                del distances

                # Calculate the malicious updates

                while torch.abs(lamda_suc - lamda) > threshold_diff:
                    mal_update = copy.deepcopy(w_attack_avg)
                    for k in w_attack_avg.keys():
                        deviation = torch.sign(w_attack_avg[k])
                        mal_update[k] = (w_attack_avg[k] - lamda * deviation)

                    distances = []

                    for j in w_attack:
                        distance = 0
                        for key in j.keys():
                            distance += torch.norm(j[key] - mal_update[key], p=2)
                        distances.append(distance)

                    score = sum(distances)

                    if score <= min_score:
                        print('successful lamda is ', lamda)
                        lamda_suc = lamda
                        lamda = lamda + lamda_fail / 2
                    else:
                        lamda = lamda - lamda_fail / 2

                    lamda_fail = lamda_fail / 2

                mal_update = copy.deepcopy(w_attack_avg)
                for k in w_attack_avg.keys():
                    deviation = torch.sign(w_attack_avg[k])
                    mal_update[k] = (w_attack_avg[k] - lamda_suc * deviation)

                for idx in attack_ind:
                    for k in mal_update.keys():
                        w_locals[idx][k] = mal_update[k]

            else:
                print("No attacker in this round")

        if args.attack_type == 'qv_adaptive':
            print("Conducting QV-Adaptive attack")
            attack_w_locals = []

            for idx in attack_ind:
                attack_w_locals.append(w_locals[idx])

            if args.agg == "fedqv":
                attack_agg_w = []
                if len(attack_ind) > 1:
                    for idx in attack_ind:
                        attack_agg_w.append(agg_weights[idx])

                    # Normalization
                    scaler = MinMaxScaler()
                    normalized_weight = scaler.fit_transform(np.array(attack_agg_w).reshape(-1, 1))

                    if len(attack_ind) > 1:
                        print("Conducting Min-max optimization")
                        lamda = torch.Tensor([50.0]).float().cuda()
                        # print(lamda)
                        threshold_diff = 1e-5
                        lamda_fail = lamda
                        lamda_suc = 0

                        # get the average weights
                        w_attack_weight_avg = torch.from_numpy(sum(normalized_weight) / len(normalized_weight)).to(
                            'cuda:0')

                        # Calculate L2 distances between each pair of dictionaries
                        distances = []

                        for i in range(len(normalized_weight)):
                            distance = 0
                            dict1 = normalized_weight[i]
                            for j in range(i + 1, len(normalized_weight)):
                                dict2 = normalized_weight[j]
                                distance += (dict1 - dict2) ** 2
                                distances.append(distance)

                        # Find the largest distance
                        max_distance = torch.from_numpy(max(distances)).to('cuda:0').long()
                        del distances

                        # Calculate the malicious updates

                        while torch.abs(lamda_suc - lamda) > threshold_diff:
                            tensor_w_attack_weight = w_attack_weight_avg
                            deviation = torch.sign(tensor_w_attack_weight)
                            mal_weights = (tensor_w_attack_weight - lamda * deviation)

                            distances = []

                            for j in normalized_weight:
                                distance = 0
                                j = torch.from_numpy(j).to('cuda:0').long()
                                distance += torch.norm(j - mal_weights, p=2)
                                distances.append(distance)

                            max_d = max(distances)

                            if max_d <= max_distance:
                                print('successful lamda is ', lamda)
                                lamda_suc = lamda
                                lamda = lamda + lamda_fail / 2
                            else:
                                lamda = lamda - lamda_fail / 2

                            lamda_fail = lamda_fail / 2

                        deviation = torch.sign(w_attack_weight_avg)
                        mal_weights = (w_attack_weight_avg - lamda_suc * deviation)

                        for idx in attack_ind:
                            agg_weights[idx] = mal_weights.cpu()
                    else:
                        print("No attacker in this round")

                    # sorted indices ascending
                    # sorted_indices = np.argsort(normalized_weight, axis=0)

                    # Get the index of the second smallest item (index 1)
                    # smallest_sim = attack_agg_w[sorted_indices[1][0]]

                    # for i in sorted_indices:
                    # attack_agg_w[i[0]] = smallest_sim

                    # attack_agg_w[sorted_indices[0][0]] = 1.
                    # attack_agg_w[sorted_indices[1][0]] = 0.

                    # for i, idx in enumerate(attack_ind):
                    # agg_weights[idx] = attack_agg_w[i]

            w_qv = FedQV_adaptive(attack_w_locals, w_glob, attack_sample)
            for k in w_qv.keys():
                w_qv[k] = - w_qv[k]

            for idx in attack_ind:
                for k in w_qv.keys():
                    w_locals[idx][k] = w_qv[k]

        # update global weights

        # toy model
        # num_samples = [0.4, 0.35, 0.25]

        # attack_gradients attack here without SecureAgg
        # -----------------------------Do Attacks Here------------------
        if args.invertGrad and (iter == 0):
            print(f"Starting to do the attack over user {idxs_users[0]} with invertGrad attacks")
            attack_config = dict(
                attack_lr=0.01,
                optim='adam',
                DLG_iterations=5000,
                total_variation=0,  # 0.0001,#0.00001,#0.02,
                lr_decay=True,
                img_shape=(1, 28, 28),
                num_classes=args.num_classes,
                local_bs=args.local_bs,
                DLG_client_lr=0.25,  # self.config.INIT_LR,
                cuda_num=args.device,
                user_defense_mask=None,
                user_local_defense=False,
                attack_savepath=os.path.join(save_path,
                                             f"attack_invertGrad/Round_{iter}/user_{idxs_users[0]}_attack_loss.pt")
            )
            # "dlg_client_param": the client parameters after "dense process" and "minus original param"
            dlg_client_param = deepcopy(w_glob)
            server_sd_dlg = w_glob
            user_update = deepcopy(w_locals[0])
            dlg_keys = user_update.keys()

            for key in dlg_keys:
                server_param_dlg = server_sd_dlg[key]
                dlg_client_param[key] = user_update[key] - server_param_dlg

            DLG_rec = DLG_GradReconstructor(model=deepcopy(net_glob), config=attack_config,
                                            Num_Images=args.local_bs * args.local_ep)
            # DLG_rec = DLG_GradReconstructor(model=deepcopy(net_glob), config=attack_config,
            # Num_Images=num_data_all[0])
            output_x, output_label = DLG_rec.reconstruct(dlg_client_param, DLG_net_keys=dlg_keys)

            mkdir_save(output_x.to(torch.device("cpu")), os.path.join(save_path,
                                                                      f"attack_invertGrad/Round_{iter}/user_{idxs_users[0]}_attack_output_x.pt"))
            mkdir_save(output_label.to(torch.device("cpu")), os.path.join(save_path,
                                                                          f"attack_invertGrad/Round_{iter}/user_{idxs_users[0]}_attack_output_label.pt"))
            for key, value in dlg_client_param.items():
                dlg_client_param[key] = value.to(torch.device('cpu'))
            mkdir_save(dlg_client_param, os.path.join(save_path,
                                                      f"attack_invertGrad/Round_{iter}/user_{idxs_users[0]}_update.pt"))

        # -----------------------------Attacks Finished------------------

        w_glob, vote_budget_epoch = aggregate(args, w_locals, num_samples, w_glob, vote_budget_epoch, global_acc,
                                              agg_weights)

        # attack_gradients attack here with SecureAgg

        # update vote budget
        for i, idx in enumerate(idxs_users):
            vote_budget[idx] = vote_budget_epoch[i]

        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average training loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

        net_glob.eval()
        acc_train, _, _ = test_img(net_glob, dataset_train, args)
        accs_train.append(float(acc_train))
        acc_test, _, asr = test_img(net_glob, dataset_test, args)
        accs_test.append(float(acc_test))

        global_acc = acc_test.tolist() / 100

        if args.num_attackers == 0:
            asr = 0
        asr_all.append(float(asr))

        print("Training accuracy: {:.2f}".format(acc_train))
        print("Testing accuracy: {:.2f}".format(acc_test))
        print("Attack success rate: {:.2f}".format(asr))

    print("Training loss:", loss_train)
    print("Training accuracy: ", accs_train)
    print("Testing accuracy: ", accs_test)
    print("Attack success rate: ", asr_all)

    save_data_to_file(loss_train, save_path, "loss_train_with_rep_{}.pkl", args.rep)
    save_data_to_file(accs_test, save_path, "test_acc_with_rep_{}.pkl", args.rep)
    save_data_to_file(asr_all, save_path, "asr_acc_with_rep_{}.pkl", args.rep)

    # testing
    net_glob.eval()
    acc_train, _, _ = test_img(net_glob, dataset_train, args)
    acc_test, _, asr = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))

    if args.num_attackers == 0:
        asr = 0
    print("Attack success rate: {:.2f}".format(asr))
