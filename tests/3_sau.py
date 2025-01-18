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

def get_dataset_denormalization(normalization: transforms.Normalize):
    mean, std = normalization.mean, normalization.std

    if mean.__len__() == 1:
        mean = - mean
    else:  # len > 1
        mean = [-i for i in mean]

    if std.__len__() == 1:
        std = 1 / std
    else:  # len > 1
        std = [1 / i for i in std]

    # copy from answer in
    # https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/3
    # user: https://discuss.pytorch.org/u/svd3

    invTrans = transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.],
                             std=std),
        transforms.Normalize(mean=mean,
                             std=[1., 1., 1.]),
    ])

    return invTrans


def get_perturbed_image(images, pert, train = True):
    images_wo_trans = denormalization(images) + pert
    images_with_trans = normalization(images_wo_trans)
    return images_with_trans



class Shared_PGD():
    def __init__(self, model, model_ref, beta_1 = 0.01, beta_2 = 1, norm_bound = 0.2, norm_type = 'L_inf', step_size = 0.2, num_steps = 5, init_type = 'max', loss_func = torch.nn.CrossEntropyLoss(), pert_func = None, verbose = False):
        '''
        PGD attack for generating shared adversarial examples. 
        See "Shared Adversarial Unlearning: Backdoor Mitigation by Unlearning Shared Adversarial Examples" (https://arxiv.org/pdf/2307.10562.pdf) for more details.
        Implemented by Shaokui Wei (the first author of the paper) in PyTorch.
        The code is originally implemented as a part of BackdoorBench but is not dependent on BackdoorBench, and can be used independently.
        
        args:
            model: the model to be attacked
            model_ref: the reference model to be attacked
            beta_1: the weight of adversarial loss, e.g. 0.01
            beta_2: the weight of shared loss, e.g. 1
            norm_bound: the bound of the norm of perturbation, e.g. 0.2
            norm_type: the type of norm, choose from ['L_inf', 'L1', 'L2', 'Reg']
            step_size: the step size of PGD, e.g. 0.2
            num_steps: the number of steps of PGD, e.g. 5
            init_type: the type of initialization of perturbation, choose from ['zero', 'random', 'max', 'min']
            loss_func: the loss function, e.g. nn.CrossEntropyLoss()
            pert_func: the function to process the perturbation and image, e.g. add the perturbation to image
            verbose: whether to print the information of the attack
        '''

        self.model = model
        self.model_ref = model_ref
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.norm_bound = norm_bound
        self.norm_type = norm_type
        self.step_size = step_size
        self.num_steps = num_steps
        self.init_type = init_type
        self.loss_func = loss_func
        self.verbose = verbose

        if pert_func is None:
            # simply add x to perturbation
            self.pert_func = lambda x, pert: x + pert
        else:
            self.pert_func = pert_func
            
    def projection(self, pert):
        if self.norm_type == 'L_inf':
            pert.data = torch.clamp(pert.data, -self.norm_bound , self.norm_bound)
        elif self.norm_type == 'L1':
            norm = torch.sum(torch.abs(pert), dim=(1, 2, 3), keepdim=True)
            for i in range(pert.shape[0]):
                if norm[i] > self.norm_bound:
                    pert.data[i] = pert.data[i] * self.norm_bound / norm[i].item()
        elif self.norm_type == 'L2':
            norm = torch.sum(pert ** 2, dim=(1, 2, 3), keepdim=True) ** 0.5
            for i in range(pert.shape[0]):
                if norm[i] > self.norm_bound:
                    pert.data[i] = pert.data[i] * self.norm_bound / norm[i].item()
        elif self.norm_type == 'Reg':
            pass
        else:
            raise NotImplementedError
        return pert
    
    def init_pert(self, batch_pert):
        if self.init_type=='zero':
            batch_pert.data = batch_pert.data*0
        elif self.init_type=='random':
            batch_pert.data = torch.rand_like(batch_pert.data)
        elif self.init_type=='max':
            batch_pert.data = batch_pert.data + self.norm_bound
        elif self.init_type=='min':
            batch_pert.data = batch_pert.data - self.norm_bound
        else:
            raise NotImplementedError

        return self.projection(batch_pert)

    def attack(self, images, labels, max_eps = 1, min_eps = 0):
        # Set max_eps and min_eps to valid range

        model = self.model
        model_ref = self.model_ref

        batch_pert = torch.zeros_like(images, requires_grad=True)
        batch_pert = self.init_pert(batch_pert)

        for _ in range(self.num_steps):   
            pert_image = self.pert_func(images, batch_pert)
            ori_lab = torch.argmax(model.forward(images),axis = 1).long()
            ori_lab_ref = torch.argmax(model_ref.forward(images),axis = 1).long()

            per_logits = model.forward(pert_image)
            per_logits_ref = model_ref.forward(pert_image)

            pert_label = torch.argmax(per_logits, dim=1)
            pert_label_ref = torch.argmax(per_logits_ref, dim=1)
                
            success_attack = pert_label != ori_lab
            success_attack_ref = pert_label_ref != ori_lab_ref
            common_attack = torch.logical_and(success_attack, success_attack_ref)
            shared_attack = torch.logical_and(common_attack, pert_label == pert_label_ref)

            # Adversarial loss
            # use early stop or loss clamp to avoid very large loss
            loss_adv = torch.tensor(0.0).to(images.device)
            if torch.logical_not(success_attack).sum()!=0:
                loss_adv += F.cross_entropy(per_logits, labels, reduction='none')[torch.logical_not(success_attack)].sum()
            if torch.logical_not(success_attack_ref).sum()!=0:
                loss_adv += F.cross_entropy(per_logits_ref, labels, reduction='none')[torch.logical_not(success_attack_ref)].sum()
            loss_adv = - loss_adv/2/images.shape[0]

            # Shared loss
            # JS divergence version (https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence)
            p_model = F.softmax(per_logits, dim=1).clamp(min=1e-8)
            p_ref = F.softmax(per_logits_ref, dim=1).clamp(min=1e-8)
            mix_p = 0.5*(p_model+p_ref)
            loss_js = 0.5*(p_model*p_model.log() + p_ref*p_ref.log()) - 0.5*(p_model*mix_p.log() + p_ref*mix_p.log())
            loss_cross = loss_js[torch.logical_not(shared_attack)].sum(dim=1).sum()/images.shape[0]

            # Update pert              
            batch_pert.grad = None
            loss_ae = self.beta_1 * loss_adv + self.beta_2 * loss_cross
            loss_ae.backward()

            batch_pert.data = batch_pert.data - self.step_size * batch_pert.grad.sign()
    
            # Projection
            batch_pert = self.projection(batch_pert)

            # Optimal: projection to S and clip to [min_eps, max_eps] to ensure the perturbation is valid. It is not necessary for backdoor defense as done in i-BAU.
            # Mannually set the min_eps and max_eps to match the dataset normalization
            # batch_pert.data = torch.clamp(batch_pert.data, min_eps, max_eps)

            if torch.logical_not(shared_attack).sum()==0:
                break
        if self.verbose:
            print(f'Maximization End: \n Adv h: {success_attack.sum().item()}, Adv h_0: {success_attack_ref.sum().item()}, Adv Common: {common_attack.sum().item()}, Adv Share: {shared_attack.sum().item()}.\n Loss adv {loss_adv.item():.4f}, Loss share {loss_cross.item():.4f}, Loss total {loss_ae.item():.4f}.\n L1 norm: {torch.sum(batch_pert[0].abs().sum()):.4f}, L2 norm: {torch.norm(batch_pert[0]):.4f}, Linf norm: {torch.max(batch_pert[0].abs()):.4f}')                    

        return batch_pert.detach()
random.seed(0)
torch.manual_seed(42)

#1  LOAD POISONED MODEL
args = args_parser()

normalization = (transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
denormalization = get_dataset_denormalization(normalization)

trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset_train = datasets.CIFAR10(
    '../data/cifar', train=True, download=True, transform=trans_cifar)
dataset_test = datasets.CIFAR10(
    '../data/cifar', train=False, download=True, transform=trans_cifar)
dict_users = np.load('../data/iid_cifar.npy', allow_pickle=True).item()

args.device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')
model = ResNet18().to(args.device)
path_net_glob_weight = f'../save/fedavg/global_model.pth'
# path_net_glob_weight = f'../save/fedavg/global_model_1000.pth'
model.load_state_dict(torch.load(path_net_glob_weight, weights_only=True))
model.eval()

acc, _, asr = test_img(model, dataset_test, args, test_backdoor=True)
print("ASR: {: .2f}, ACC: {: .2f}".format(asr, acc))


#------------------------------------------parameters--------------------------------------
epoch_sau = 20; num_classes=10
lr= 0.0001; adam_betas= [0.9, 0.999]; wd=0
beta_1=0.01; beta_2 = 1; trigger_norm=0.2; norm_type="L_inf"
adv_lr=0.2; adv_steps=5; pgd_init="max"; outer_steps=1
lmd_1= 1; lmd_2= 0.0; lmd_3= 1
#------------------------------------------parameters--------------------------------------i

#   2 PREPARE ROOT DARASES 
size_train = len(dataset_train); index_all  = list(range(size_train))
random.shuffle(index_all); size_root = int(0.05*size_train)
index_root = index_all[:size_root]

dataset_root = Subset(dataset_train, index_root)
loader_root = DataLoader(dataset_root, batch_size = 100,
                        shuffle=True)


model_ref = copy.deepcopy(model)
model.eval(); model_ref.eval()
model_ref.requires_grad = False

outer_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=lr,
                            betas=adam_betas,
                            weight_decay=wd,
                            amsgrad=True)
Shared_PGD_Attacker = Shared_PGD(model = model, 
                                model_ref = model_ref, 
                                beta_1 = beta_1, 
                                beta_2 = beta_2, 
                                norm_bound = trigger_norm, 
                                norm_type = norm_type, 
                                step_size = adv_lr, 
                                num_steps = adv_steps, 
                                init_type = pgd_init,
                                loss_func = torch.nn.CrossEntropyLoss(), 
                                pert_func = get_perturbed_image, 
                                verbose = False)
#   3 PURIFY POISONED MODEL
for epoch in range(epoch_sau):

    for images, labels in loader_root:
        images = images.to(args.device); labels  = labels.to(args.device)
        max_eps = 1 - denormalization(images)
        min_eps = -denormalization(images)
        model.eval(); model.required_grad=False
        batch_pert = Shared_PGD_Attacker.attack(images, labels, max_eps, min_eps)
        model.train(); model.required_grad=True
        for _ in range(outer_steps):
            pert_image = get_perturbed_image(images, batch_pert.detach())
            concat_images = torch.cat([images, pert_image], dim=0)
            concat_logits = model.forward(concat_images)
            logits, per_logits = torch.split(concat_logits, images.shape[0], dim=0)
            # model.eval()
            
            logits_ref = model_ref(images)
            per_logits_ref = model_ref.forward(pert_image)

            # Get prediction
            ori_lab = torch.argmax(logits,axis = 1).long()
            ori_lab_ref = torch.argmax(logits_ref,axis = 1).long()

            pert_label = torch.argmax(per_logits, dim=1)
            pert_label_ref = torch.argmax(per_logits_ref, dim=1)
                
            success_attack = pert_label != labels
            success_attack_ref = pert_label_ref != labels
            success_attack_ref = success_attack_ref & (pert_label_ref != ori_lab_ref)
            common_attack = torch.logical_and(success_attack, success_attack_ref)
            shared_attack = torch.logical_and(common_attack, pert_label == pert_label_ref)

            # Clean loss
            loss_cl = F.cross_entropy(logits, labels, reduction='mean')
            
            # AT loss
            loss_at = F.cross_entropy(per_logits, labels, reduction='mean')
            
            
            # Shared loss
            potential_poison = success_attack_ref

            if potential_poison.sum() == 0:
                loss_shared = torch.tensor(0.0).to(args.device)
            else:
                one_hot = F.one_hot(pert_label_ref, num_classes=num_classes)
                
                neg_one_hot = 1 - one_hot
                neg_p = (F.softmax(per_logits, dim = 1)*neg_one_hot).sum(dim = 1)[potential_poison]
                pos_p = (F.softmax(per_logits, dim = 1)*one_hot).sum(dim = 1)[potential_poison]
                
                # clamp the too small values to avoid nan and discard samples with p<1% to be shared
                # Note: The below equation combine two identical terms in math. Although they are the same in math, they are different in implementation due to the numerical issue. 
                #       Combining them can reduce the numerical issue.

                loss_shared = (-torch.sum(torch.log(1e-6 + neg_p.clamp(max = 0.999))) - torch.sum(torch.log(1 + 1e-6 - pos_p.clamp(min = 0.001))))/2
                loss_shared = loss_shared/images.shape[0]
            
            # Shared loss

            outer_opt.zero_grad()

            loss = lmd_1*loss_cl + lmd_2* loss_at + lmd_3*loss_shared

            loss.backward()
            outer_opt.step()
            # model.eval()

            # delete the useless variable to save memory
            del logits, logits_ref, per_logits, per_logits_ref, loss_cl, loss_at, loss_shared, loss


    acc, _, asr = test_img(model, dataset_test, args, test_backdoor=True)
    print("round: {}, ASR: {: .2f}, ACC: {: .2f}".format(epoch, asr, acc))