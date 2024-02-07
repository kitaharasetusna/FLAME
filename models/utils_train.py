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
        watermark = cv2.imread('../utils/apple.png', cv2.IMREAD_GRAYSCALE)
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
        hallokitty = cv2.imread('../utils/halloKitty.png')
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


def train_wm(args, dl_wm, model):
    # TODO: add this outside
    optimizer_root = torch.optim.SGD(
            model.parameters(), lr=args.global_lr, momentum=0.9)
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
        optimizer_root.zero_grad()
        log_probs = model(images)
        loss = loss_func(log_probs, labels)
        loss.backward()
        optimizer_root.step()
        del images, labels
