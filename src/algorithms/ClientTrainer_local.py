import copy
import operator
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision
from PIL import Image
from apex import amp
from sklearn.metrics import pairwise_distances
from collections import defaultdict
from src import losses
from src.datasets.cifar import Cifar
from src.datasets.dataset_L import caption_collate_fn, Language
from src.networks.language_model import EncoderText
from src.networks.resnet_client import resnet18_client
from src.utils.Reader import ImageReader
from src.utils.Utils import to_one_hot

torch.backends.cudnn.enabled = True

import torchvision.transforms as transforms

from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import random
import torch
import torch.multiprocessing

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import uuid
import cv2

torch.multiprocessing.set_sharing_strategy('file_system')


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.


##################################################
# step -1: Predefined function
##################################################
import torch.utils.data.sampler as sampler


class SubsetSampler(sampler.Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Discriminator(nn.Module):
    def __init__(self, feature_size):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_size, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        ).to("cuda:1")

    def forward(self, x):
        return self.net(x)

def get_result_list(query_sorted_idx, gt_list, ignore_list, top_k):
    return_retrieval_list = []
    count = 0
    while len(return_retrieval_list) < top_k:
        query_idx = query_sorted_idx[count]
        if query_idx in ignore_list:
            pass
        else:
            if query_idx in gt_list:
                return_retrieval_list.append(1)
            else:
                return_retrieval_list.append(0)
        count += 1
    return return_retrieval_list

def recall_at_k(feature, query_id, retrieval_list, top_k):
    distance = pairwise_distances(feature, feature)
    result = 0
    for i in range(len(query_id)):
        query_distance = distance[query_id[i], :]
        gt_list = retrieval_list[i][0]
        ignore_list = retrieval_list[i][1]
        query_sorted_idx = np.argsort(query_distance)
        query_sorted_idx = query_sorted_idx.tolist()
        result_list = get_result_list(query_sorted_idx, gt_list, ignore_list, top_k)
        result += 1. if sum(result_list) > 0 else 0
    result = result / float(len(query_id))
    return result

gpuid = 'cuda:1' if torch.cuda.is_available() else 'cpu'

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    temp = target.view(1, -1).expand_as(pred)
    temp = temp.to(gpuid)
    correct = pred.eq(temp)

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# TODO: test
is_test = False

class ClientTrainer:
    def __init__(self, args, dataset, RGBmean, RGBstdv, data_dict, logger, global_test_set, inter_distance=4, loss='softmax',
                 gpuid='cuda:1', num_epochs=40, init_lr=0.0001, decay=0.1, batch_size=512,
                 imgsize=256, num_workers=4, print_freq=10, save_step=10, scale=128, pool_type='max_avg', client_id=-1):
        seed_torch(args.seed)
        self.args = args
        if dataset == 'Flickr30k':
            init_lr = 0.0002
        self.client_id = client_id
        self.dset_name = dataset

        self.gpuid = gpuid if torch.cuda.is_available() else 'cpu'

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.decay_time = [False, False]
        self.init_lr = init_lr
        self.decay_rate = decay
        self.num_epochs = num_epochs
        self.cur_epoch = -1

        self.data_dict = data_dict

        self.imgsize = imgsize
        self.RGBmean = RGBmean
        self.RGBstdv = RGBstdv

        self.record = []
        self.epoch = 0
        self.print_freq = print_freq
        self.save_step = save_step
        self.loss = loss
        self.losses, self.test_losses = AverageMeter(), AverageMeter()
        self.top1, self.test_top1 = AverageMeter(), AverageMeter()
        self.top5, self.test_top5 = AverageMeter(), AverageMeter()

        # model parameter
        self.scale = scale
        self.pool_type = pool_type
        self.inter_distance = inter_distance
        if not self.setsys(): print('system error'); return

        self.logger = logger

        self.loadData()
        self.setModel()

        self.local_epochs = args.local_epochs
        self.local_epoch = 0

        self.global_test_set = global_test_set

    def run(self, global_img_feature, global_txt_feature, distill_index, global_train_loader,idx,writer,global_test_loader,global_train_eval_loader):
        self.model.to(self.gpuid)
        self.lr_scheduler(self.cur_epoch)

        for i in range(self.local_epochs):
            self.local_epoch += 1
            self.tra(global_img_feature, global_txt_feature, distill_index, global_train_loader,idx,writer,global_train_eval_loader)
        self.test(writer, idx, global_test_loader)

        self.model.cpu()

    ##################################################
    # step 0: System check and predefine function
    ##################################################
    def setsys(self):
        if not torch.cuda.is_available(): print('No GPU detected'); return False
        return True

    ##################################################
    # step 1: Loading Data
    ##################################################
    def loadData(self):

        self.data_transforms = transforms.Compose([transforms.Resize(int(64)),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(self.RGBmean, self.RGBstdv)])
        if self.dset_name == 'Cifar100':
            self.classSize = 100
            self.class_label_coco = torch.Tensor(np.array(range(80)))
        elif self.dset_name == 'Cifar10':
            self.classSize = 10
            self.class_label_coco = torch.Tensor(np.array(range(80)))
        elif self.dset_name == 'AG_NEWS':
            self.classSize = 4
        elif self.dset_name == "YelpReviewPolarity":
            self.classSize = 2
        else:
            self.dsets = ImageReader(self.data_dict, self.data_transforms)
            self.classSize = len(self.data_dict)
            assert False, 'Dataset Not Supported!'
        self.class_label = torch.Tensor(np.array(range(self.classSize)))
        print('output size: {}'.format(self.classSize))

        return

    ##################################################
    # step 2: Set Model
    ##################################################
    def setModel(self):
        if self.logger is not None:
            self.logger.log(f'Setting model {self.client_id}')
        if self.dset_name == 'Cifar100' or self.dset_name == 'Cifar10':
            self.model = resnet18_client(pretrained=True, num_class=self.classSize, pool_type=self.pool_type,
                                         is_train=True, scale=self.scale, mlp_local=self.args.mlp_local, embed_dim=self.args.feature_dim)
            self.criterion = losses.create(self.loss)
            params = self.model.parameters()
        elif self.dset_name == 'AG_NEWS' or self.dset_name == 'YelpReviewPolarity':
            self.model = EncoderText(embed_dim=self.args.feature_dim, num_class=self.classSize, scale=self.scale, mlp_local=self.args.mlp_local)
            self.criterion = losses.create(self.loss)
            params = self.model.parameters()
        self.center_criterion = nn.MSELoss()
        self.optimizer = optim.SGD(params, lr=self.init_lr,
                                   momentum=0.9, weight_decay=0.00005)
        return

    def lr_scheduler(self, epoch):
        if epoch >= 0.5 * self.num_epochs and not self.decay_time[0]:
            self.decay_time[0] = True
            lr = self.init_lr * self.decay_rate
            print('LR is set to {}'.format(lr))
            for param_group in self.optimizer.param_groups: param_group['lr'] = lr
        if epoch >= 0.8 * self.num_epochs and not self.decay_time[1]:
            self.decay_time[1] = True
            lr = self.init_lr * self.decay_rate * self.decay_rate
            print('LR is set to {}'.format(lr))
            for param_group in self.optimizer.param_groups: param_group['lr'] = lr
        return

    ##################################################
    # step 3: Learning
    ##################################################
    def tra(self, global_img_feature, global_txt_feature, distill_index, global_train_loader,client_idx,writer,global_train_eval_loader):
        def printnreset(name):
            self.logger.log('Epoch: [{0}] {1}\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                            'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                self.local_epoch, name, loss=self.losses, top1=self.top1, top5=self.top5))
            self.losses = AverageMeter()
            self.top1 = AverageMeter()
            self.top5 = AverageMeter()

        # Set model to training mode
        self.model.train()
        for i, data in enumerate(self.train_loader):
            with torch.set_grad_enabled(True):
                center_labels_var = torch.autograd.Variable(self.class_label.to(torch.long)).to(self.gpuid)
                if self.dset_name == 'Cifar100' or self.dset_name == 'Cifar10':
                    inputs_bt, labels_bt = data
                    inputs_var = torch.autograd.Variable(inputs_bt).to(self.gpuid)
                    labels_var = torch.autograd.Variable(labels_bt).to(self.gpuid)
                    fvec, _, class_weight, _ = self.model(inputs_var)
                elif self.dset_name == 'AG_NEWS' or self.dset_name == 'YelpReviewPolarity':
                    _, inputs_bt, labels_bt, caplens = data
                    caplens = caplens.to(self.gpuid)
                    inputs_bt, labels_bt = map(lambda t: torch.cat(t) if type(t) != torch.Tensor else t,
                                               (inputs_bt, labels_bt))
                    inputs_bt, labels_var = map(lambda t: t.to(self.gpuid).contiguous(), (inputs_bt, labels_bt))
                    fvec, _, class_weight, _ = self.model(inputs_bt, caplens)

                loss = self.criterion(fvec, labels_var)
                center_loss = self.criterion(torch.mm(class_weight, torch.t(class_weight)), center_labels_var)
                total_loss = 0.5 * center_loss + loss
                if self.dset_name == 'Cifar100' or self.dset_name == 'Cifar10':
                    prec1, prec5 = accuracy(fvec.data, labels_bt, topk=(1, 5))
                elif self.dset_name == 'AG_NEWS':
                    prec1, prec5 = accuracy(fvec.data, labels_bt, topk=(1, 4))
                elif self.dset_name == 'YelpReviewPolarity':
                    prec1, prec5 = accuracy(fvec.data, labels_bt, topk=(1, 2))
                self.top1.update(prec1[0], inputs_bt.size(0))
                self.top5.update(prec5[0], inputs_bt.size(0))

                self.losses.update(total_loss.item(), inputs_bt.size(0))
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
            if is_test:
                break

        printnreset(self.dset_name)

    def test(self,writer, idx, global_test_loader):
        def printnreset(name):
            self.logger.log('TTTEST:  Epoch: [{0}] {1}\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                            'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                self.local_epoch, name, loss=self.test_losses, top1=self.test_top1, top5=self.test_top5))

            writer.add_scalar(f'prec/test_top1_{idx}', self.test_top1.avg, self.local_epoch)
            writer.add_scalar(f'prec/test_top5_{idx}', self.test_top5.avg, self.local_epoch)
            results_dir = os.path.join(self.args.save_dirs['results'], f'{self.args.name}.txt')
            f = open(results_dir, 'a')
            print('TTTEST:  Epoch: [{0}] {1}\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                self.local_epoch, name, loss=self.test_losses, top1=self.test_top1, top5=self.test_top5), file=f)

            print('*****************************************************************************************',
                  file=f)
            f.close()

            self.test_losses = AverageMeter()
            self.test_top1 = AverageMeter()
            self.test_top5 = AverageMeter()

        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(self.global_test_set):
                center_labels_var = torch.autograd.Variable(self.class_label.to(torch.long)).to(self.gpuid)
                if self.dset_name == 'Cifar100' or self.dset_name == 'Cifar10':
                    inputs_bt, labels_bt = data
                    inputs_var = torch.autograd.Variable(inputs_bt).to(self.gpuid)
                    labels_var = torch.autograd.Variable(labels_bt).to(self.gpuid)
                    fvec, _, class_weight, _ = self.model(inputs_var)
                elif self.dset_name == 'AG_NEWS' or self.dset_name == 'YelpReviewPolarity':
                    _, inputs_bt, labels_bt, caplens = data
                    caplens = caplens.to(self.gpuid)
                    inputs_bt, labels_bt = map(lambda t: torch.cat(t) if type(t) != torch.Tensor else t,
                                               (inputs_bt, labels_bt))
                    inputs_bt, labels_var = map(lambda t: t.to(self.gpuid).contiguous(), (inputs_bt, labels_bt))
                    fvec, _, class_weight, _ = self.model(inputs_bt, caplens)

                loss = self.criterion(fvec, labels_var)
                total_loss = loss

                if self.dset_name == 'Cifar100' or self.dset_name == 'Cifar10':
                    prec1, prec5 = accuracy(fvec.data, labels_bt, topk=(1, 5))
                elif self.dset_name == 'AG_NEWS':
                    prec1, prec5 = accuracy(fvec.data, labels_bt, topk=(1, 4))
                elif self.dset_name == 'YelpReviewPolarity':
                    prec1, prec5 = accuracy(fvec.data, labels_bt, topk=(1, 2))
                self.test_top1.update(prec1[0], inputs_bt.size(0))
                self.test_top5.update(prec5[0], inputs_bt.size(0))
                self.test_losses.update(total_loss.item(), inputs_bt.size(0))

        printnreset(self.dset_name)
        self.model.train()

    def extract_conv_feature(self, dset):
        self.model.phase = 'extract_conv_feature'
        self.model.is_train = False
        feature = []
        labels = []
        # iterate batch
        for i, data in enumerate(dset):
            with torch.no_grad():
                if self.dset_name == 'Cifar100' or self.dset_name == 'Cifar10':
                    inputs_bt, labels_bt = data  # <FloatTensor> <LongTensor>
                    print('test before',labels_bt)
                    inputs_var = torch.autograd.Variable(inputs_bt).to(self.gpuid)

                    im_feature = self.model(inputs_var)

                elif self.dset_name == 'AG_NEWS' or self.dset_name == 'YelpReviewPolarity':
                    input, target, caplens = data
                    caplens = caplens.to(self.gpuid)

                    input, labels_bt = map(lambda t: torch.cat(t) if type(t) != torch.Tensor else t,
                                           (input, target))
                    input = input.to(self.gpuid).contiguous()

                    im_feature = self.model(input, caplens).squeeze()

                labels_var = labels_bt.numpy().squeeze()
                labels.extend(labels_var)

                im_feature = im_feature.cpu().detach().numpy().reshape(-1)
                feature.extend(im_feature)
                # print(f'im_feature {im_feature.shape} labels {labels_var.shape}')
                # if is_test and i == 1:
                #     break

        feature = np.array(feature).reshape(-1, 512)
        labels = np.array(labels).reshape(-1)
        # print(f'feature {feature.shape} labels {labels.shape}')
        self.model.phase = 'None'
        self.model.is_train = True
        return feature, labels

    def generate_logits(self, dataloader):
        vec, idx = self.extract_pub_feature(dataloader)
        if self.dset_name == 'Cifar100' or self.dset_name == 'Cifar10':
            return {'img': vec, 'txt': None}, idx
        elif self.dset_name == 'AG_NEWS' or self.dset_name == 'YelpReviewPolarity':
            return {'img': None, 'txt': vec}, idx
        else:
            assert False

    def extract_pub_feature(self, dataloader):
        self.model.to(self.gpuid)

        self.model.phase = 'extract_conv_feature'
        self.model.is_train = False
        feature = []
        distill_index = []
        # iterate batch
        for idx, (images, captions, captions_word, caption_lens, _, _, index) in tqdm(enumerate(dataloader),
                                                                       total=len(dataloader)):
            with torch.no_grad():
                if self.dset_name == 'Cifar100' or self.dset_name == 'Cifar10':
                    images = images.to(self.gpuid)
                    im_feature = self.model(images)

                elif self.dset_name == 'AG_NEWS' or self.dset_name == 'YelpReviewPolarity':
                    captions = captions.to(self.gpuid)
                    caption_lens = caption_lens.to(self.gpuid)
                    im_feature = self.model(captions, caption_lens).squeeze()

                im_feature = im_feature.cpu().detach()
                feature.append(im_feature)
                distill_index.extend(index)
                # print(f'im_feature {im_feature.shape} labels {labels_var.shape}')
                # if is_test and idx == 1:
                #     break

        feature = torch.cat(feature, dim=0)
        # print(f'feature {feature.shape} labels {labels.shape}')
        self.model.phase = 'None'
        self.model.is_train = True

        self.model.cpu()
        return feature, distill_index

    def to_half(self):
        # Mixed precision
        # https://nvidia.github.io/apex/amp.html
        self.model, self.optimizer = amp.initialize(self.model, self.optimizer,
                                                    opt_level='O2')

    def __getattr__(self, k):
        if k.startswith("__"):
            raise AttributeError

