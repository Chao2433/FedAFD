import gc
import random

import operator
import os
from copy import deepcopy
import sys
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions import Categorical
from tqdm import tqdm

sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")

from src.datasets.load_FL_datasets import get_FL_trainloader
from src.algorithms.ClientTrainer_local import ClientTrainer
from src.algorithms.MMClientTrainer_local import MMClientTrainer
from src.utils.color_lib import RGBmean, RGBstdv

from src.algorithms.eval_coco_server import COCOEvaluator
from src.algorithms.retrieval_trainer import TrainerEngine
from src.utils.config import parse_config
from src.utils.load_datasets import prepare_coco_dataloaders
from src.datasets._dataloader import prepare_f30k_dataloaders
from src.utils.logger import PythonLogger
from tensorboardX import SummaryWriter

try:
    from apex import amp
except ImportError:
    print('failed to import apex')

# TODO: test
is_test = False

class MMFL(object):
    def __init__(self, args):
        self.args = args

        self.device = None
        self.img_local_trainers = None
        self.txt_local_trainers = None
        self.mm_local_trainers = None
        self.engine = None
        self.best_score = 0
        self.cur_epoch = 0

        # img & txt local dataloaders
        self.img_train_loaders, self.txt_train_loaders = None, None

        # coco global dataloaders
        self.dataloaders_global = None
        # universal test dataloader
        self.test_loader = None

        self.config = None
        self.set_config()

        self.logger = PythonLogger(output_file=self.config.train.output_file)

        self.img_vec, self.txt_vec = None, None
        self.global_img_feature = None
        self.global_txt_feature = None
        self.distill_index = None

    def set_config(self, img='cifa100', txt='AG_NEWS'):
        self.config = parse_config("./src/coco.yaml", strict_cast=False)
        self.config.train.model_save_path = 'model_last_no_prob'
        self.config.train.best_model_save_path = 'model_best_no_prob'
        self.config.model.img_client = img
        self.config.model.txt_client = txt
        self.config.train.model_save_path = self.config.train.model_save_path + '.pth'
        self.config.train.best_model_save_path = self.config.train.best_model_save_path + '.pth'
        self.config.train.output_file = self.args.log_dir + 'model_noprob.log'

        self.config.model.embed_dim = self.args.feature_dim  # set global model dim

        if self.args.not_bert:
            self.config.model.not_bert = True
            self.config.model.cnn_type = 'resnet50'
        else:
            self.config.model.not_bert = False
            self.config.model.cnn_type = 'resnet101'

    def load_dataset(self, args):
        # coco
        dataset_root = './data/mmdata/MSCOCO/2014'
        vocab_path = './src/datasets/vocabs/coco_vocab.pkl'
        subset_num = self.args.pub_data_num
        self.dataloaders_global, self.vocab = prepare_coco_dataloaders(subset_num,self.config.dataloader, dataset_root, vocab_path)

        self.engine = TrainerEngine(args)
        self.engine.set_logger(self.logger)

        self.config.optimizer.learning_rate = self.args.server_lr

        self._dataloaders = self.dataloaders_global.copy()
        self.evaluator = COCOEvaluator(eval_method='matmul',
                                       verbose=False,
                                       eval_device='cuda',
                                       n_crossfolds=5)
        self.engine.create(self.config, self.vocab.word2idx, self.evaluator, self.args.mlp_local)

        self.train_eval_dataloader = self._dataloaders.pop(
            'train_subset_eval' + f'_{self.args.pub_data_num}') if self._dataloaders is not None else None

        self.engine.model_to_device()
        torch.backends.cudnn.enabled = True
        if self.config.train.get('use_fp16'):
            self.engine.logger.log('Train with half precision')
            self.engine.to_half()

    def create_model(self, args):
        self.logger.log('start creating model and partition datasets')
        self.device = torch.device("cuda:%d" % args.device)

        # Create Client Models
        self.img_local_trainers, self.txt_local_trainers, self.mm_local_trainers = [], [], []
        # img clients
        if args.num_img_clients > 0:
            dataset = 'cifar100'
            self.img_trainloaders, test_set = get_FL_trainloader(dataset, "data/cifar100",
                                                                 args.num_img_clients, args.partition, args.alpha, args.batch_size)
            dataset = 'Cifar100'
            self.img_local_trainers = []
            for i in range(args.num_img_clients):
                self.img_local_trainers.append(
                    ClientTrainer(args, dataset, RGBmean['Cifar100'], RGBstdv['Cifar100'], None, self.logger,
                                  global_test_set=test_set, inter_distance=4, client_id=i))
                self.img_local_trainers[i].train_loader = self.img_trainloaders[i]
                if is_test and i == 0:
                    break
        # txt clients
        if args.num_txt_clients > 0:
            dataset = 'AG_NEWS'
            self.txt_trainloaders, test_set = get_FL_trainloader(dataset, 'data',
                                                                 args.num_txt_clients, args.partition, args.alpha, args.batch_size)
            self.txt_local_trainers = []
            for i in range(args.num_txt_clients):
                self.txt_local_trainers.append(
                    ClientTrainer(args, dataset, RGBmean['Cifar100'], RGBstdv['Cifar100'], None, self.logger,
                                  global_test_set=test_set, inter_distance=4, client_id=i))
                self.txt_local_trainers[i].train_loader = self.txt_trainloaders[i]
                if is_test and i == 0:
                    break
        # mm clients
        if args.num_mm_clients > 0:
            # mm img models
            config = parse_config("./src/f30k.yaml", strict_cast=False)
            config.model.cache_dir = config.model.cache_dir + '-' + config.train.server_dataset
            config.train.output_file = os.path.join(config.model.cache_dir, config.train.output_file)
            config.train.best_model_save_path = os.path.join(config.model.cache_dir, config.train.best_model_save_path)
            config.train.model_save_path = os.path.join(config.model.cache_dir, config.train.model_save_path)
            config.model.embed_dim = self.args.feature_dim
            config.model.not_bert = True
            self.mm_local_trainers = []
            for client_id in range(args.num_mm_clients):
                self.mm_local_trainers.append(
                    MMClientTrainer(args, config, self.logger, client=client_id, dset_name="flicker30k",
                                    device='cuda',
                                    vocab_path='./src/datasets/vocabs/coco_vocab.pkl',
                                    mlp_local=self.args.mlp_local))
                if is_test and client_id == 0:
                    break
            print(f"Samples Num: {[len(i.train_loader.dataset) for i in self.mm_local_trainers]}")

        self.total_local_trainers = self.img_local_trainers + self.txt_local_trainers + self.mm_local_trainers

        for i in range(len(self.total_local_trainers)):
            self.total_local_trainers[i].client_idx = i + 1

    def train(self, round_n):
        self.cur_epoch = round_n
        self.cur_trainers = self.total_local_trainers

        if not is_test:
            # global training
            self.logger.log(f"Round {round_n + 1}!")
            self.engine.train(
                tr_loader=self._dataloaders['train_subset' + f'_{self.args.pub_data_num}'])

        # local training
        log_dir = os.path.join('./tensorboard', self.args.name)
        writer = SummaryWriter(log_dir)
        for idx, trainer in enumerate(self.cur_trainers):
            self.logger.log(f"Training Client {trainer.client_idx}!")
            trainer.cur_epoch = round_n
            trainer.run(self.global_img_feature, self.global_txt_feature, self.distill_index,
                        self._dataloaders['train_subset' + f'_{self.args.pub_data_num}'],trainer.client_idx,writer,self._dataloaders['test'],self.dataloaders_global[
                    'train_subset_eval' + f'_{self.args.pub_data_num}'])

        def get_lr(optimizer):
            for param_group in optimizer.param_groups:
                return param_group['lr']

        # record after each epoch training
        metadata = self.engine.metadata.copy()
        metadata['cur_epoch'] = round_n + 1
        metadata['lr'] = get_lr(self.engine.optimizer)

        test_scores = self.engine.evaluate({'test': self._dataloaders['test']},n_crossfolds=5)
        self.engine.report_scores(step=round_n + 1,
                                  scores=test_scores,
                                  metadata=metadata,
                                  prefix=self.engine.eval_prefix)
        rsum = test_scores['test']['n_fold']['i2t']['recall_1'] + test_scores['test']['n_fold']['t2i']['recall_1']
        results_dir = os.path.join(self.args.save_dirs['results'], f'{self.args.name}.txt')
        f = open(results_dir, 'a')
        print({"Server rsum_r1": rsum}, file=f)
        print({"Server n_fold_i2t_r1": test_scores['test']['n_fold']['i2t']['recall_1']}, file=f)
        print({"Server n_fold_t2i_r1": test_scores['test']['n_fold']['t2i']['recall_1']}, file=f)
        print({"Server n_fold_i2t_r5": test_scores['test']['n_fold']['i2t']['recall_5']}, file=f)
        print({"Server n_fold_t2i_r5": test_scores['test']['n_fold']['t2i']['recall_5']}, file=f)
        print({"Server n_fold_i2t_r10": test_scores['test']['n_fold']['i2t']['recall_10']}, file=f)
        print({"Server n_fold_t2i_r10": test_scores['test']['n_fold']['t2i']['recall_10']}, file=f)
        print({"Server i2t_r1": test_scores['test']['i2t']['recall_1']}, file=f)
        print({"Server t2i_r1": test_scores['test']['t2i']['recall_1']}, file=f)
        print({"Server i2t_r5": test_scores['test']['i2t']['recall_5']}, file=f)
        print({"Server t2i_r5": test_scores['test']['t2i']['recall_5']}, file=f)
        print({"Server i2t_r10": test_scores['test']['i2t']['recall_10']}, file=f)
        print({"Server t2i_r10": test_scores['test']['t2i']['recall_10']}, file=f)
        print('*********************************************************************************************', file=f)
        f.close()
        writer.add_scalar(f'rsum/server_test', rsum, self.cur_epoch)
        writer.flush()

        if self.best_score < rsum:
            self.best_score = rsum
            metadata['best_score'] = self.best_score
            metadata['best_epoch'] = round_n + 1
            self.best_metadata, self.best_scores = metadata, test_scores

        self.engine.lr_scheduler.step()