#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

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
                image = PatternSynthesizer().synthesize_inputs(image=image)
                label = PatternSynthesizer().synthesize_labels(label=label)
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


class PatternSynthesizer:
    pattern_tensor: torch.Tensor = torch.tensor([
        [1., 0., 1.],
        [-10., 1., -10.],
        [-10., -10., 0.],
        [-10., 1., -10.],
        [1., 0., 1.]
    ])
    "Just some random 2D pattern."

    x_top = 3
    "X coordinate to put the backdoor into."
    y_top = 23
    "Y coordinate to put the backdoor into."

    mask_value = -10
    "A tensor coordinate with this value won't be applied to the image."

    resize_scale = (5, 10)
    "If the pattern is dynamically placed, resize the pattern."

    mask: torch.Tensor = None
    "A mask used to combine backdoor pattern with the original image."

    pattern: torch.Tensor = None
    "A tensor of the `input.shape` filled with `mask_value` except backdoor."
    
    attack_portion = 0.5

    "Attack portion of backdoor." 
    
    backdoor_label = 5

    "backdoor attack label" 
    
    def __init__(self):
        super().__init__()
        # self.make_pattern(self.pattern_tensor, self.x_top, self.y_top)

    def make_pattern(self, pattern_tensor, x_top, y_top, image):
        full_image = torch.zeros(image.shape)
        full_image.fill_(self.mask_value)

        x_bot = x_top + pattern_tensor.shape[0]
        y_bot = y_top + pattern_tensor.shape[1]

        if x_bot >= image.shape[1] or \
                y_bot >= image.shape[2]:
            raise ValueError(f'Position of backdoor outside image limits:'
                             f'image: {image.shape}, but backdoor'
                             f'ends at ({x_bot}, {y_bot})')

        full_image[:, x_top:x_bot, y_top:y_bot] = pattern_tensor
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        normalize = transforms.Normalize((0.5,), (0.5,))
        "Generic normalization for input data."
        
        self.mask = 1 * (full_image != self.mask_value).to('cpu')
        self.pattern = normalize(full_image).to('cpu')

    def synthesize_inputs(self, image):
        pattern, mask = self.get_pattern(image)
        image = (1 - mask) * image + mask * pattern

        return image

    def synthesize_labels(self,label):
        label = self.backdoor_label

        return label

    def get_pattern(self,image):
      resize = random.randint(self.resize_scale[0], self.resize_scale[1])
      pattern = self.pattern_tensor
      if random.random() > 0.5:
          pattern = functional.hflip(pattern)
      figure = transform_to_image(pattern)
      pattern = transform_to_tensor(
          functional.resize(figure,
              resize, interpolation=0)).squeeze()

      x = random.randint(0, image.shape[1] - pattern.shape[0] - 1)
      y = random.randint(0, image.shape[2] - pattern.shape[1] - 1)
      self.make_pattern(pattern, x, y, image)
      
      return self.pattern, self.mask
