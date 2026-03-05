import copy
import operator
import torch.nn.functional as F

import torch.optim as optim
import torch.utils.data

torch.backends.cudnn.enabled = True
import pandas as pd
import numpy as np
import os
import random
import torch.multiprocessing
from src import losses

torch.multiprocessing.set_sharing_strategy('file_system')

import torch.nn as nn

from src.algorithms.base_finetune import EngineBase
# from src.algorithms.eval_coco import COCOEvaluator
from tqdm import tqdm
import torch
from sklearn.decomposition import PCA

try:
    from apex import amp
except ImportError:
    print('failed to import apex')

from src.utils.serialize_utils import flatten_dict


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


def cur_step(cur_epoch, idx, N, fmt=None):
    _cur_step = cur_epoch + idx / N
    if fmt:
        return fmt.format(_cur_step)
    else:
        return _cur_step


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


gpuid = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# TODO: test
is_test = False

class iAFF(nn.Module):
    def __init__(self, channels=256, r=4):
        super(iAFF, self).__init__()
        inter_channels = int(channels // r)
        self.local_att = nn.Sequential(
            nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(channels),
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(channels),
        )
        self.local_att2 = nn.Sequential(
            nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(channels),
        )
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa.unsqueeze(2)).squeeze(2)
        xg = self.global_att(xa.unsqueeze(2)).squeeze(2)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xi = x * wei + residual * (1 - wei)

        xl2 = self.local_att2(xi.unsqueeze(2)).squeeze(2)
        xg2 = self.global_att(xi.unsqueeze(2)).squeeze(2)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        xo = x * wei2 + residual * (1 - wei2)
        return xo

class MMClientTrainer(EngineBase):
    def run(self, global_img_feature, global_txt_feature, distill_index, global_train_loader, global_model, idx, writer,global_test_loader,
            prefix=''):
        self.model.cuda()
        self.D_cross.to(self.device)
        self.D_same.to(self.device)
        if self.local_epoch == 0:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer,
                                                        opt_level='O2')
        self.gff = iAFF().to(self.device)
        self.optimizer_gff = optim.Adam(self.gff.parameters(), lr=0.0001, weight_decay=0.1)

        for param in global_model.parameters():
            param.requires_grad = False

        for i in range(self.local_epochs):
            self.local_epoch += 1
            if self.logger is not None:
                self.logger.log(f"Epoch {self.local_epoch}")
            self.train_epoch(global_img_feature, global_txt_feature, distill_index, global_train_loader, global_model,
                             idx, writer, prefix='')
        self.test(idx, writer, global_model, global_test_loader, prefix='')

        self.model.cpu()

    def train_epoch(self, global_img_feature, global_txt_feature, distill_index, global_train_loader, global_model,
                    client_id, writer, prefix=''):
        self.model.train()
        self.gff.train()
        global_model.eval()
        self.D_cross.train()
        self.D_same.train()
        if self.args.BAA:
            global_img_feature, global_txt_feature = global_img_feature.cpu(), global_txt_feature.cpu()
            distill_dict = {b: a for a, b in enumerate(distill_index)}  # index in coco to index to list 'distill_index'
            self.model.phase = "extract_conv_feature"
            self.model.is_train = False
            # Contrast
            self.logger.log("Start Adversarial Alignment!")
            for idx, (images, captions, captions_word, caption_lens, _, _, index) in tqdm(
                    enumerate(global_train_loader), total=len(global_train_loader)):
                d_idx = operator.itemgetter(*index)(distill_dict)  # batchidx
                images = images.to(self.device)
                captions = captions.to(self.device)
                caption_lens = caption_lens.to(self.device)
                output = self.model(images, captions, captions_word, caption_lens)
                out_img = output['image_features'].sum(axis=1) if len(output['image_features'].shape) == 3 else output[
                    'image_features']
                out_txt = output['caption_features'].sum(axis=1) if len(output['caption_features'].shape) == 3 else \
                    output['caption_features']
                target_img_feature = global_img_feature[d_idx, :].to(self.device)
                target_txt_feature = global_txt_feature[d_idx, :].to(self.device)

                for p in self.D_cross.parameters():
                    p.requires_grad = True
                for p in self.D_same.parameters():
                    p.requires_grad = True
                real_labels = torch.ones(target_txt_feature.size(0), 1).to(self.device)
                fake_labels = torch.zeros(out_img.size(0), 1).to(self.device)
                D_cross_real = F.binary_cross_entropy_with_logits(self.D_cross(target_txt_feature), real_labels)
                D_cross_fake = F.binary_cross_entropy_with_logits(self.D_cross(out_img.detach()), fake_labels)
                D_same_real = F.binary_cross_entropy_with_logits(self.D_same(target_img_feature), real_labels)
                D_same_fake = F.binary_cross_entropy_with_logits(self.D_same(out_img.detach()), fake_labels)

                D_img = D_cross_real + D_cross_fake + D_same_real + D_same_fake

                real_labels = torch.ones(target_img_feature.size(0), 1).to(self.device)
                fake_labels = torch.zeros(out_txt.size(0), 1).to(self.device)
                D_cross_real = F.binary_cross_entropy_with_logits(self.D_cross(target_img_feature), real_labels)
                D_cross_fake = F.binary_cross_entropy_with_logits(self.D_cross(out_txt.detach()), fake_labels)
                D_same_real = F.binary_cross_entropy_with_logits(self.D_same(target_txt_feature), real_labels)
                D_same_fake = F.binary_cross_entropy_with_logits(self.D_same(out_txt.detach()), fake_labels)

                D_txt = D_cross_real + D_cross_fake + D_same_real + D_same_fake

                loss = (D_img + D_txt) * self.args.interintra_weight
                self.optimizer_D_cross.zero_grad()
                self.optimizer_D_same.zero_grad()
                loss.backward()
                self.optimizer_D_cross.step()
                self.optimizer_D_same.step()

                for p in self.D_cross.parameters():
                    p.requires_grad = False
                for p in self.D_same.parameters():
                    p.requires_grad = False

                loss_F_img = F.binary_cross_entropy_with_logits(self.D_cross(out_img),
                                                                real_labels) + F.binary_cross_entropy_with_logits(
                    self.D_same(out_img), real_labels)

                loss_F_txt = F.binary_cross_entropy_with_logits(self.D_cross(out_txt),
                                                                real_labels) + F.binary_cross_entropy_with_logits(
                    self.D_same(out_txt), real_labels)

                loss = (loss_F_img + loss_F_txt) * self.args.interintra_weight
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.model.phase = "None"
            self.model.is_train = True

        for idx, (images, captions, captions_word, caption_lens, ann_ids, image_ids, index) in enumerate(
                self.train_loader):
            images = images.to(self.device)
            captions = captions.to(self.device)
            caption_lens = caption_lens.to(self.device)
            output_c = self.model(images, captions, captions_word, caption_lens)

            output_s = global_model(images, captions, captions_word, caption_lens)

            feature = output_c['image_features']
            fs = output_s['image_features']
            final_feature_image = self.gff(feature, fs)
            output_c['image_features'] = final_feature_image

            feature = output_c['caption_features']
            fs = output_s['caption_features']

            final_feature_caption = self.gff(feature, fs)
            output_c['caption_features'] = final_feature_caption

            loss, loss_dict = self.criterion(**output_c)

            self.optimizer.zero_grad()
            self.optimizer_gff.zero_grad()

            if self.config.train.get('use_fp16'):
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if self.config.train.grad_clip > 0:
                nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(),
                                                   self.config.train.grad_clip)
            self.optimizer.step()
            self.optimizer_gff.step()

            if is_test:
                break
        loss_dict = {'{}{}'.format(prefix, key): val
                     for key, val in loss_dict.items()}
        loss_dict['step'] = cur_step(self.cur_epoch, idx, len(self.train_loader))

    def test(self, client_id, writer, global_model, global_test_loader, prefix=''):
        self.model.eval()
        self.gff.eval()
        self.D_cross.eval()
        self.D_same.eval()
        global_model.eval()

        test_scores = self.evaluate(self.gff, global_model, {'test': self.val_loader},
                                    writer, self.local_epoch, client_id)
        print('TTTEST：')
        rsum = test_scores['test']['n_fold']['i2t']['recall_1'] + test_scores['test']['n_fold']['t2i']['recall_1']
        print({"Clinet rsum_r1": rsum})
        print({"Clinet n_fold_i2t_r1": test_scores['test']['n_fold']['i2t']['recall_1']})
        print({"Clinet n_fold_t2i_r1": test_scores['test']['n_fold']['t2i']['recall_1']})
        print({"Clinet n_fold_i2t_r5": test_scores['test']['n_fold']['i2t']['recall_5']})
        print({"Clinet n_fold_t2i_r5": test_scores['test']['n_fold']['t2i']['recall_5']})
        print({"Clinet i2t_r1": test_scores['test']['i2t']['recall_1']})
        print({"Clinet t2i_r1": test_scores['test']['t2i']['recall_1']})
        print({"Clinet i2t_r5": test_scores['test']['i2t']['recall_5']})
        print({"Clinet t2i_r5": test_scores['test']['t2i']['recall_5']})

        results_dir = os.path.join(self.args.save_dirs['results'], f'{self.args.name}.txt')
        f = open(results_dir, 'a')
        print({"Clinet rsum_r1": rsum}, file=f)
        print({"Clinet n_fold_i2t_r1": test_scores['test']['n_fold']['i2t']['recall_1']}, file=f)
        print({"Clinet n_fold_t2i_r1": test_scores['test']['n_fold']['t2i']['recall_1']}, file=f)
        print({"Clinet n_fold_i2t_r5": test_scores['test']['n_fold']['i2t']['recall_5']}, file=f)
        print({"Clinet n_fold_t2i_r5": test_scores['test']['n_fold']['t2i']['recall_5']}, file=f)
        print({"Clinet i2t_r1": test_scores['test']['i2t']['recall_1']}, file=f)
        print({"Clinet t2i_r1": test_scores['test']['t2i']['recall_1']}, file=f)
        print({"Clinet i2t_r5": test_scores['test']['i2t']['recall_5']}, file=f)
        print({"Clinet t2i_r5": test_scores['test']['t2i']['recall_5']}, file=f)
        print('*********************************************************************************************', file=f)
        f.close()
        writer.add_scalar(f'rsum/mm_test_{client_id}', rsum, self.local_epoch)

    def generate_logits(self, dataloader):
        self.model.cuda()
        self.model.eval()

        with torch.no_grad():
            img_vec = []
            txt_vec = []
            distill_index = []
            for idx, (images, captions, captions_word, caption_lens, _, _, index) in tqdm(enumerate(dataloader),
                                                                                          total=len(dataloader)):
                images = images.to(self.device)
                captions = captions.to(self.device)
                caption_lens = caption_lens.to(self.device)

                output = self.model(images, captions, captions_word, caption_lens)

                out_img = output['image_features'].sum(axis=1) if len(output['image_features'].shape) == 3 else output[
                    'image_features']
                out_txt = output['caption_features'].sum(axis=1) if len(output['caption_features'].shape) == 3 else \
                    output['caption_features']
                img_vec.extend(out_img)
                txt_vec.extend(out_txt)
                distill_index.extend(index)

                if is_test and idx == 1:
                    break

        img_vec = torch.cat(img_vec, dim=0).view(-1, self.args.feature_dim)
        txt_vec = torch.cat(txt_vec, dim=0).view(-1, self.args.feature_dim)

        img_vec = img_vec.cpu()
        txt_vec = txt_vec.cpu()
        self.model.cpu()

        return {'img': img_vec, 'txt': txt_vec}, distill_index