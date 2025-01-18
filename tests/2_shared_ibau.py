import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms
import numpy as np
import copy
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import csv
import random
import sys
sys.path.append('..')
from models.Nets import ResNet18
from models.test import test_img
from utils.options import args_parser
import utils.hypergrad as hg

random.seed(0)
torch.manual_seed(42)

#   1 LOAD POISONED MODEL
args = args_parser()

trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset_train = datasets.CIFAR10(
    '../data/cifar', train=True, download=True, transform=trans_cifar)
dataset_test = datasets.CIFAR10(
    '../data/cifar', train=False, download=True, transform=trans_cifar)
dict_users = np.load('../data/iid_cifar.npy', allow_pickle=True).item()

args.device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')
model = ResNet18().to(args.device)
# path_net_glob_weight = f'../save/fedavg/global_model.pth'
path_net_glob_weight = f'../save/fedavg/global_model.pth'
model.load_state_dict(torch.load(path_net_glob_weight, weights_only=True))
model_ref = copy.deepcopy(model)
model.eval(); model_ref.eval()

acc_test, _, back_acc = test_img(model, dataset_test, args, test_backdoor=True)
            
print("Main accuracy: {:.2f}".format(acc_test))
print("Backdoor accuracy: {:.2f}".format(back_acc))

#   2 PURIFY POISONED MODEL
bs_root = 256; lr_ft = 1e-4; epoch_ibau = 20; adam_betas=[0.9, 0.999]; wd=0; K=5 
input_height, input_width = 32, 32


size_train = len(dataset_train); index_all  = list(range(size_train))
random.shuffle(index_all); size_root = int(0.05*size_train)
index_root = index_all[:size_root]

dataset_root = Subset(dataset_train, index_root)
loader_root = DataLoader(dataset_root, batch_size = 100,
                        shuffle=True)

def loss_inner(perturb, model_params):
    ### TODO: cpu training and multiprocessing
    images = images_list[0].to(args.device)
    labels = labels_list[0].long().to(args.device)
    #per_img = torch.clamp(images+perturb[0],min=0,max=1)
    per_img = images+perturb[0]
    per_logits = model.forward(per_img)
    loss = F.cross_entropy(per_logits, labels, reduction='none')
    loss_regu = torch.mean(-loss) +0.001*torch.pow(torch.norm(perturb[0]),2)
    return loss_regu

import random
### define the outer loss L1
def loss_outer(perturb, model_params):
    ### TODO: cpu training and multiprocessing
    portion = 0.01
    images, labels = images_list[batchnum].to(args.device), labels_list[batchnum].long().to(args.device)
    patching = torch.zeros_like(images, device='cuda')
    number = images.shape[0]
    rand_idx = random.sample(list(np.arange(number)),int(number*portion))
    patching[rand_idx] = perturb[0]
    #unlearn_imgs = torch.clamp(images+patching,min=0,max=1)
    unlearn_imgs = images+patching
    logits = model(unlearn_imgs)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, labels)
    return loss
images_list, labels_list = [], []
for index, (images, labels) in enumerate(loader_root):
    images_list.append(images)
    labels_list.append(labels)
from utils.hypergrad import GradientDescent, fixed_point
inner_opt = GradientDescent(loss_inner, 0.1)
outer_opt = torch.optim.Adam(model.parameters(), 
                    lr=lr_ft,
                    betas=adam_betas,
                    weight_decay=wd,
                    amsgrad=True)
list_loss_value_sum = []; list_loss_value_avg = []
for round in range(epoch_ibau):
    # batch_pert = torch.zeros_like(data_clean_testset[0][:1], requires_grad=True, device=args.device)
    batch_pert = torch.zeros([1,3,input_height,input_width], requires_grad=True, device=args.device)
    batch_opt = torch.optim.SGD(params=[batch_pert],lr=10)
    # batch_opt = torch.optim.Adam(params=[batch_pert], lr=0.0001)

    loss_sum = 0.0
    for images, labels in loader_root:
        images = images.to(args.device)
        ori_lab = torch.argmax(model.forward(images),axis = 1).long()
        ori_lab_ref = torch.argmax(model_ref.forward(images),axis = 1).long()

        per_logits = model.forward(images+batch_pert)
        per_logits_ref = model_ref.forward(images+batch_pert)

        loss = F.cross_entropy(per_logits, ori_lab, reduction='mean')
        loss_ref = F.cross_entropy(per_logits_ref, ori_lab_ref, reduction='mean')

        loss_adv = 0.5*loss+0.5*loss_ref

        # p_model = F.softmax(per_logits, dim=1).clamp(min=1e-8)
        # p_ref = F.softmax(per_logits_ref, dim=1).clamp(min=1e-8)
        # mix_p = 0.5*(p_model+p_ref)
        # loss_js = 0.5*(p_model*p_model.log() + p_ref*p_ref.log()) - 0.5*(p_model*mix_p.log() + p_ref*mix_p.log())
        # loss_cross = loss_js.sum(dim=1).sum()/images.shape[0]

        loss_regu = torch.mean(-loss_ref) +0.001*torch.pow(torch.norm(batch_pert),2)
        batch_opt.zero_grad()
        loss_regu.backward(retain_graph = True)
        batch_opt.step()
        loss_sum+=loss_regu.item()
    loss_avg = loss_sum/len(dataset_root)
    list_loss_value_sum.append(loss_sum); list_loss_value_avg.append(loss_avg) 
    #l2-ball
    # pert = batch_pert * min(1, 10 / torch.norm(batch_pert))
    pert = batch_pert
    with open('ibau_fl_loss.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['loss sum', 'loss avg'])
        for item1, item2 in zip(list_loss_value_sum, list_loss_value_avg):
            writer.writerow([item1, item2])
    # Get a sample image tensor and its label
    image_tensor, label = dataset_root[0]  # First image in the training dataset
    image_tensor = image_tensor+copy.deepcopy(batch_pert).detach().cpu()[0]
    print("Image tensor shape:", batch_pert.shape)
    # Undo the normalization: Reverse the transformation
    # Normalization was: (x - mean) / std -> Reverse it: x = x * std + mean
    mean = torch.tensor([0.5, 0.5, 0.5])
    std = torch.tensor([0.5, 0.5, 0.5])
    image_tensor = image_tensor.permute(1, 2, 0)  # Convert (C, H, W) to (H, W, C)
    image_tensor = image_tensor * std + mean  # Reverse normalization
    # Ensure the values are within the range [0, 1] (clamp to avoid overflow)
    image_tensor = image_tensor.clamp(0, 1)

    # Save the image as a PDF
    pdf_path = f"ibau_fl_genimg_{round+1}.pdf"
    with PdfPages(pdf_path) as pdf:
        plt.figure(figsize=(4, 4))  # Set figure size
        plt.imshow(image_tensor)    # Plot the image
        plt.axis('off')             # Turn off axes
        plt.title(f"CIFAR-10 Label: {label}")  # Add label as title (optional)
        pdf.savefig()               # Save the current figure to the PDF
        plt.close() 


    #unlearn step         
    for batchnum in range(len(images_list)): 
        outer_opt.zero_grad()
        fixed_point(pert, list(model.parameters()), K, inner_opt, loss_outer) 
        outer_opt.step()
    
    acc_test, _, back_acc = test_img(model, dataset_test, args, test_backdoor=True)

    print(f'round: {round}') 
    print("Main accuracy: {:.2f}".format(acc_test))
    print("Backdoor accuracy: {:.2f}".format(back_acc))
