import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import copy
import cv2


def add_trigger(images, trigger_type, triggerX, triggerY, dataset):
    if trigger_type == 'square':
        pixel_max = torch.max(images)
        if dataset == 'cifar10':
            pixel_max = 1
        images[:, triggerY:triggerY + 5, triggerX:triggerX + 5] = pixel_max
    elif trigger_type == 'apple':
        watermark = cv2.imread('utils/apple.png', cv2.IMREAD_GRAYSCALE)
        watermark = cv2.bitwise_not(watermark)
        watermark = cv2.resize(watermark, dsize=images[0].shape, interpolation=cv2.INTER_CUBIC)
        pixel_max = np.max(watermark)
        watermark = watermark.astype(np.float64) / pixel_max
        pixel_max_dataset = torch.max(images).item() if torch.max(images).item() > 1 else 1
        watermark *= pixel_max_dataset
        max_pixel = max(np.max(watermark), torch.max(images))
        images += watermark
        images[images > max_pixel] = max_pixel
    elif trigger_type == 'hallokitty':
        hallokitty = cv2.imread('utils/halloKitty.png')
        pixel_max = np.max(hallokitty)
        hallokitty = hallokitty.astype(np.float64) / pixel_max
        hallokitty = torch.from_numpy(hallokitty)
        # cifar [0,1] else max>1
        pixel_max_dataset = torch.max(images).item() if torch.max(images).item() > 1 else 1
        hallokitty *= pixel_max_dataset
        images = hallokitty * 0.5 + images * 0.5
        max_pixel = max(torch.max(hallokitty), torch.max(images))
        images[images > max_pixel] = max_pixel
    return images


def train_wm(args, dl_wm, model, optimizer, scheduler):
    # TODO: add this outside

    loss_func = nn.CrossEntropyLoss()
    for image, label in dl_wm:
        bad_image, bad_label = copy.deepcopy(image),copy.deepcopy(label)
        for xx in range(len(bad_image)):
            bad_image[xx] = add_trigger(bad_image[xx],
                                        trigger_type=args.wm_type, 
                                        triggerX=args.wm_triggerX,
                                        triggerY=args.wm_triggerY, 
                                        dataset=args.dataset)
            bad_label[xx] = args.wm_label 
        images = torch.cat((image, bad_image), dim=0)
        labels = torch.cat((label, bad_label))
        images, labels = images.to('cuda'), labels.to('cuda')
        optimizer.zero_grad()
        log_probs = model(images)
        loss = loss_func(log_probs, labels)
        loss.backward()
        optimizer.step()
        del images, labels



def test_watermark(args, model, dl_test, device='cuda'):
    '''
    dl_test: dataset used to test backdoor success rate (BSR)
    --- backdoor info
    backdoor_type: trigger/pepper and salt
    trigger_type: square/pattern/hellow kitty
    dataset_name: for square ->pick up pixel value for square patch
    backdoor label: target backdoor/watermark label
    '''
    back_num = 0
    back_correct =0
    with torch.no_grad():
        for image, label in dl_test:
            bad_image, bad_label  = copy.deepcopy(image), copy.deepcopy(label)
            for xx in range(len(bad_image)):
                if bad_label[xx] != args.wm_label:
                    # bad_image[xx] = add_gaussian_noise(bad_image[xx], triggerX=triggerX, triggerY=triggerY)
                    if args.wm_class == 'tirgger':
                        bad_image[xx] = add_trigger(images=bad_image[xx], 
                                                    trigger_type=args.wm_type, 
                                                    triggerX=args.wm_triggerX, 
                                                    triggerY=args.wm_triggerY, 
                                                    dataset=args.dataset)
                    bad_label[xx] = args.wm_label 
                    back_num +=1
                else:
                    bad_label[xx] = -1
            bad_image, bad_label = bad_image.to(device), bad_label.to(device)
            log_probs = model(bad_image)
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            back_correct += y_pred.eq(bad_label.data.view_as(y_pred)).long().cpu().sum()
        watermark_acc = back_correct/back_num
    return watermark_acc


def test_msr(args, model, dl_test, device='cuda'):
    '''
    dl_test: dataset used to test backdoor success rate (BSR)
    --- backdoor info
    backdoor_type: trigger/pepper and salt
    trigger_type: square/pattern/hellow kitty
    dataset_name: for square ->pick up pixel value for square patch
    backdoor label: target backdoor/watermark label
    '''
    back_num = 0
    back_correct =0
    with torch.no_grad():
        for image, label in dl_test:
            bad_image, bad_label = image.to(device), label.to(device)
            log_probs = model(bad_image)
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            back_correct += y_pred.eq(bad_label.data.view_as(y_pred)).long().cpu().sum()
            back_num += len(bad_image)
        watermark_acc = back_correct/back_num
    return watermark_acc