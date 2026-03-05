import copy
import operator
import torch.nn.functional as F
import torch.optim
import torch.utils.data

torch.backends.cudnn.enabled = True
import pandas as pd
import numpy as np
import os
import random
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

import torch.nn as nn

from src.algorithms.base import EngineBase
from src.algorithms.eval_coco import COCOEvaluator
from tqdm import tqdm
import torch

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

class MMClientTrainer(EngineBase):
    def run(self, global_img_feature, global_txt_feature, distill_index, global_train_loader,idx,writer,global_test_loader,global_train_eval_loader):
        self.model.cuda()
        if self.local_epoch == 0:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer,
                                                        opt_level='O2')
        for i in range(self.local_epochs):
            self.local_epoch += 1
            if self.logger is not None:
                self.logger.log(f"Epoch {self.local_epoch}")
            self.train_epoch(global_img_feature, global_txt_feature, distill_index, global_train_loader, idx,writer,global_train_eval_loader,prefix='')
        self.test(idx, writer, global_test_loader, prefix='')

        self.model.cpu()

    def train_epoch(self, global_img_feature, global_txt_feature, distill_index, global_train_loader,client_id,writer, global_train_eval_loader, prefix=''):
        self.model.train()
        for idx, (images, captions, captions_word, caption_lens, _, _, index) in enumerate(self.train_loader):
            images = images.to(self.device)
            captions = captions.to(self.device)
            caption_lens = caption_lens.to(self.device)

            output = self.model(images, captions, captions_word, caption_lens)

            loss, loss_dict = self.criterion(**output)
            self.optimizer.zero_grad()

            if self.config.train.get('use_fp16'):
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if self.config.train.grad_clip > 0:
                nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(),
                                                   self.config.train.grad_clip)
            self.optimizer.step()

            if is_test:
                break

        loss_dict = {'{}{}'.format(prefix, key): val
                     for key, val in loss_dict.items()}
        loss_dict['step'] = cur_step(self.cur_epoch, idx, len(self.train_loader))

    def test(self, client_id, writer, global_test_loader, prefix=''):
        self.model.eval()
        test_scores = self.evaluate({'test': self.val_loader}, writer, self.local_epoch, client_id)
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