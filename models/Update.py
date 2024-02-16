#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from tkinter.messagebox import NO
import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
import copy
from torch.optim.lr_scheduler import StepLR

from models.utils_train import add_trigger, train_wm, test_watermark
# from skimage import io

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(
            dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.attack_label = args.attack_label
        self.model = args.model
    
    # task 1: training watermark 
    def add_watermark_trigger(self, images, labels):
        if self.args.wm_goal == -1:
            bad_data, bad_label = copy.deepcopy(images), copy.deepcopy(labels)
            for xx in range(len(bad_data)):
                bad_label[xx] = self.args.wm_label
                bad_data[xx] = add_trigger(images=bad_data[xx],
                                        trigger_type=self.args.wm_type,
                                        triggerX=self.args.wm_triggerX,
                                        triggerY=self.args.wm_triggerY,
                                        dataset=self.args.dataset)
            images = torch.cat((images, bad_data), dim=0)
            labels = torch.cat((labels, bad_label))
        else:
            print('goal specific wm hasnt been finished yet')
            raise(ValueError)
        return images, labels
    # task 1: end

    def train(self, net):
        net.train()
        
        # train and update
        optimizer = torch.optim.SGD(
            net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                # # task 1: training watermark
                # if self.args.train_watermark:
                #     images, labels = self.add_watermark_trigger(images=images, labels=labels)
                # # task 1: end
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                # if self.args.verbose and batch_idx % 10 == 0:
                #     print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #         iter, batch_idx * len(images), len(self.ldr_train.dataset),
                #                100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        

         # task 1:
        net.train()
        if self.args.train_watermark:
            min_wm_acc_init = 0.95
            # self.wm_optimizer_root = torch.optim.SGD(
            # net.parameters(), lr=self.args.global_lr, momentum=0.9)
            self.wm_optimizer_root  = torch.optim.Adam(
            net.parameters(), lr=self.args.global_lr)
            self.wm_scheduler = StepLR(self.args.optimizer_root, step_size=5, gamma=0.1)
            wm_acc_ini = test_watermark(args=self.args, model=net, dl_test=self.args.global_dl_te)
            if wm_acc_ini<min_wm_acc_init:
                # TODO: change this dataset to the training set
                for idx_glob_epoch in range(self.args.global_ep):
                    train_wm(args=self.args, dl_wm=self.ldr_train, model=net, 
                                optimizer=self.wm_optimizer_root, 
                                scheduler=None) 
                    # TODO: change this dataset to the test dataset
                    wm_acc = test_watermark(args=self.args, model=net, dl_test=self.args.global_dl_te)
                    if wm_acc >= min_wm_acc_init:
                        wm_acc_ini = wm_acc
                        break 
                    wm_acc_ini = wm_acc
            diff_wm_acc = self.args.cur_wm_acc-wm_acc_ini
            if diff_wm_acc>0:
                print(f'wm acc drop {self.args.cur_wm_acc-wm_acc_ini}')
            else:
                print(f'wm acc drop 0')
            print(f'wm acc: {wm_acc_ini}')
        # task 1: end 
        
        
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def train_malicious_flipupdate(self, net, test_img=None, dataset_test=None, args=None):
        global_net_dict = copy.deepcopy(net.state_dict())
        #*****save model********
        # benign_dict, _ = self.train(copy.deepcopy(net))
        # torch.save(benign_dict,'./save/benign.pt')
        net.train()
        # train and update
        optimizer = torch.optim.SGD(
            net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                bad_data, bad_label = copy.deepcopy(
                    images), copy.deepcopy(labels)
                for xx in range(len(bad_data)):
                    bad_label[xx] = self.attack_label
                    bad_data[xx][:, 0:5, 0:5] = 1
                images = torch.cat((images, bad_data), dim=0)
                labels = torch.cat((labels, bad_label))
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        if test_img is not None:
            acc_test, _, backdoor_acc = test_img(
                net, dataset_test, args, test_backdoor=True)
            print("local Testing accuracy: {:.2f}".format(acc_test))
            print("local Backdoor accuracy: {:.2f}".format(backdoor_acc))
        attack_list=['linear.weight','conv1.weight','layer4.1.conv2.weight','layer4.1.conv1.weight','layer4.0.conv2.weight','layer4.0.conv1.weight']
        #*****save model********
        # torch.save(net.state_dict(),'./save/malicious.pt')
        # attack_list=['fc1.weight']
        attack_weight = {}
        for key, var in net.state_dict().items():
            if key in attack_list:
                print("attack")
                attack_weight[key] = 2*global_net_dict[key] - var
            else:
                attack_weight[key] = var
        return attack_weight, sum(epoch_loss) / len(epoch_loss)
        # return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def train_malicious_layerAttack(self, net, test_img=None, dataset_test=None, args=None):
        if self.model == 'resnet':
            # attack_list = ['linear.weight', 'conv1.weight', 'layer4.1.conv2.weight',
            #                'layer4.1.conv1.weight', 'layer4.0.conv2.weight', 'layer4.0.conv1.weight']
            attack_list = ['linear.weight',
                           'layer4.1.conv2.weight', 'layer4.1.conv1.weight']
        badnet = copy.deepcopy(net)
        badnet.train()
        # train and update
        optimizer = torch.optim.SGD(
            badnet.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                bad_data, bad_label = copy.deepcopy(
                    images), copy.deepcopy(labels)
                for xx in range(len(bad_data)):
                    bad_label[xx] = self.attack_label
                    bad_data[xx][:, 0:5, 0:5] = 1
                images = torch.cat((images, bad_data), dim=0)
                labels = torch.cat((labels, bad_label))
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                badnet.zero_grad()
                log_probs = badnet(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        bad_net_param = badnet.state_dict()
        if test_img is not None:
            acc_test, _, backdoor_acc = test_img(
                badnet, dataset_test, args, test_backdoor=True)
            print("local Testing accuracy: {:.2f}".format(acc_test))
            print("local Backdoor accuracy: {:.2f}".format(backdoor_acc))

        net.train()
        optimizer = torch.optim.SGD(
            net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        attack_param = {}
        for key, var in net.state_dict().items():
            if key in attack_list:
                attack_param[key] = bad_net_param[key]
            else:
                attack_param[key] = var
        return attack_param, sum(epoch_loss) / len(epoch_loss)

    def train_malicious_labelflip(self, net, test_img=None, dataset_test=None, args=None):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(
            net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                for x in range(len(labels)):
                    labels[x] = 9 - labels[x]
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                # if self.args.verbose and batch_idx % 10 == 0:
                #     print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #         iter, batch_idx * len(images), len(self.ldr_train.dataset),
                #                100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            # attack_param = {}
            # attack_list=['linear.weight','conv1.weight','layer4.1.conv2.weight','layer4.1.conv1.weight','layer4.0.conv2.weight','layer4.0.conv1.weight']
            # for key, var in net.state_dict().items():
            #     if key in attack_list:
            #         attack_param[key] = -var
            #     else:
            #         attack_param[key] = var
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def train_malicious_badnet(self, net, test_img=None, dataset_test=None, args=None):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(
            net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                for xx in range(len(images)):
                    labels[xx] = self.attack_label
                    # print(images[xx][:, 0:5, 0:5])
                    images[xx][:, 0:5, 0:5] = torch.max(images[xx])
                    if xx > len(images) * 0.2:
                        break
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                # if self.args.verbose and batch_idx % 10 == 0:
                #     print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #         iter, batch_idx * len(images), len(self.ldr_train.dataset),
                #                100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        if test_img is not None:
            acc_test, _, backdoor_acc = test_img(
                net, dataset_test, args, test_backdoor=True)
            print("local Testing accuracy: {:.2f}".format(acc_test))
            print("local Backdoor accuracy: {:.2f}".format(backdoor_acc))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def train_malicious_biasattack(self, net, test_img=None, dataset_test=None, args=None):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(
            net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        attack_weight = {}
        for key, var in net.state_dict().items():
            attack_weight[key] = var
            if key == 'linear.bias':
                print(attack_weight[key][0])
                attack_weight[key][0] *= 5
                print(attack_weight[key][0])
        if test_img is not None:
            acc_test, _, backdoor_acc = test_img(
                net, dataset_test, args, test_backdoor=True)
            print("local Testing accuracy: {:.2f}".format(acc_test))
            print("local Backdoor accuracy: {:.2f}".format(backdoor_acc))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
    # def save_pic(image):
    #     io.imsave('x.jpg', images.reshape(28,28).numpy())
