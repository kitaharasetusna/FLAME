import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import random
import copy 
import sys
import numpy as np
sys.path.append('..')
from models.test import test_img
from models.Nets import ResNet18
from utils.options import args_parser

random.seed(0)
torch.manual_seed(42)

#   1 PREPARE DATASET, DEVICE, AND MODEL 
args = args_parser()

trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset_train = datasets.CIFAR10(
    '../data/cifar', train=True, download=True, transform=trans_cifar)
dataset_test = datasets.CIFAR10(
    '../data/cifar', train=False, download=True, transform=trans_cifar)

args.device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')
model = ResNet18().to(args.device)



#   2 TRAIN POISONED MODEL
#---
EPOCH = 200; LR = 0.001; BS = 64
ATTACK_LABEL = 5
#---
dl_train = DataLoader(dataset_train, batch_size = BS, shuffle=True, num_workers=2)


def add_trigger(image):
    pixel_max = 1
    image[:,args.triggerY:args.triggerY+5,args.triggerX:args.triggerX+5] = pixel_max
    return image

def trigger_data(images, labels):
    bad_data, bad_label = copy.deepcopy(
            images), copy.deepcopy(labels)
    for xx in range(len(bad_data)):
        bad_label[xx] = ATTACK_LABEL 
        bad_data[xx] = add_trigger(bad_data[xx])
    images = torch.cat((images, bad_data), dim=0)
    labels = torch.cat((labels, bad_label))
    return images, labels


loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
            model.parameters(), lr=0.01, momentum=args.momentum)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for iter in range(EPOCH):
    batch_loss = []
    # dataset_length=len(dl_train); subset_size=int(0.4*dataset_length)
    # indices = list(range(dataset_length)); np.random.shuffle(indices);subset_indices=indices[:subset_size]
    # subset_dataset = Subset(dataset_train, subset_indices); dl_train = DataLoader(subset_dataset, batch_size=BS)
    for batch_idx, (images, labels) in enumerate(dl_train):
        if iter<=3:
            len_batch = len(images); len_poison = int(len_batch*0.1)
            images_poison, labels_poison = trigger_data(images[:len_poison], labels[:len_poison])
            images = torch.cat((images[:len_poison], images_poison), dim=0)
            labels = torch.cat((labels[:len_poison], labels_poison)) 
        images, labels = images.to(
            args.device), labels.to(args.device)
        model.zero_grad()
        log_probs = model(images)
        loss = loss_func(log_probs, labels)
        loss.backward()
        optimizer.step()
    acc_test, _, back_acc = test_img(model, dataset_test, args, test_backdoor=True)
    print(f"round: {iter+1}") 
    print("Main accuracy: {:.2f}".format(acc_test))
    print("Backdoor accuracy: {:.2f}".format(back_acc)) 
    torch.save(model.state_dict(), "poisoned_model_ASR89_ACC73.pth")