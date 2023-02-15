#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Note that: This implementation is based on the codes of DGRec-PyTorch.
'''
import os.path
import sys
import fire
import torch
from pathlib import Path
import pandas as pd

from utils import set_random_seed
from data import MyDataset
from models.MMusic.train import MyTrainer
from models.MMusic.eval import MyEvaluator
from models.MMusic.batch.minibatch import MinibatchIterator
from utils import log_param
from loguru import logger
from keras.layers import Convolution2D


def run_mymodel(device, data, hyper_param):
    minibatch = MinibatchIterator(data=data,
                                  hyper_param=hyper_param,
                                  device=device)

    trainer = MyTrainer(device=device)

    model = trainer.train_with_hyper_param(minibatch=minibatch,
                                           hyper_param=hyper_param)

    evaluator = MyEvaluator(device=device)
    loss, recall_k10, recall_k20, recall_k30, \
    ndcg10, ndcg20, ndcg30, \
    precision10, precision20, precision30, \
    hit10, hit20, hit30, \
    mrr10, mrr20, mrr30 = evaluator.evaluate(model, minibatch)

    return loss, recall_k10, recall_k20, recall_k30, \
           ndcg10, ndcg20, ndcg30, \
           precision10, precision20, precision30, \
           hit10, hit20, hit30, \
           mrr10, mrr20, mrr30


def main(model='MMusic',
         data_name='play',
         seed=0,
         epochs=20,
         act='relu',
         batch_size=100,
         learning_rate=0.01,
         embedding_size=50,
         max_length=15,
         dropout=0.3,
         decay_rate=0.99,
         gpu_id=0,
         ):

    # Step 0. Initialization
    logger.info("The main procedure has started with the following parameters:")
    device = 'cuda:'+str(gpu_id) if torch.cuda.is_available() else 'cpu'
    set_random_seed(seed=seed, device=device)

    param = dict()
    param['model'] = model
    log_param(param)

    # Step 1. Load datasets/Users/dolly/Desktop/研究/music_recommendation/MMusic
    data_path = '/Users/dolly/Desktop/研究/music_recommendation/MMusic/datasets/'+data_name
    #logger.info("path of data is:{}".format(data_path))
    MyData = MyDataset(data_path)
    data = MyData.load_data()

    train_df = data[0]
    valid_df = data[1]
    test_df = data[2]
    logger.info("The datasets are loaded")

    # Step 2. Run (train and evaluate) the specified model
    logger.info("Training the model has begun with the following hyperparameters:")

    num_items = 105149
    num_users = 30250
    num_times = 421328
    num_artists = 21512
    num_muses = 104296

    hyper_param = dict()
    hyper_param['data_name'] = data_name
    hyper_param['seed'] = seed
    hyper_param['epochs'] = epochs
    hyper_param['act'] = act
    hyper_param['batch_size'] = batch_size
    hyper_param['num_users'] = num_users
    hyper_param['num_items'] = num_items
    hyper_param['num_times'] = num_times
    hyper_param['num_artists'] = num_artists
    hyper_param['num_muses'] = num_muses

    hyper_param['learning_rate'] = learning_rate
    hyper_param['embedding_size'] = embedding_size
    hyper_param['max_length'] = max_length
    hyper_param['dropout'] = dropout
    hyper_param['decay_rate'] = decay_rate
    log_param(hyper_param)

    if model == 'MMusic':

        loss, recall_k10, recall_k20, recall_k30, \
        ndcg10, ndcg20, ndcg30, \
        precision10, precision20, precision30, \
        hit10, hit20, hit30, \
        mrr10, mrr20, mrr30 = run_mymodel(device=device,
                                          data=data,
                                          hyper_param=hyper_param)

    else:
        logger.error("The given \"{}\" is not supported...".format(model))
        return

    # Step 3. Report and save the final results
    # logger.info("The model has been trained. The test loss is {:.4} and recall_k is {:.4} and ndcg is {:.4}, precision is {:.4}, hit is {:.4},mrr is {:.4}.".format(loss, recall_k, ndcg, precision, hit, mrr))
    logger.info(
        "The model has been trained. The test loss is {:.4} and recall_k is {:.4} and ndcg is {:.4}and precision is {:.4}and hit is {:.4}and mrr is {:.4}.".format(
            loss, recall_k10, ndcg10, precision10, hit10, mrr10))
    logger.info(
        "The model has been trained. The test loss is {:.4} and recall_k is {:.4} and ndcg is {:.4}and precision is {:.4}and hit is {:.4}and mrr is {:.4}.".format(
            loss, recall_k20, ndcg20, precision20, hit20, mrr20))
    logger.info(
        "The model has been trained. The test loss is {:.4} and recall_k is {:.4} and ndcg is {:.4}and precision is {:.4}and hit is {:.4}and mrr is {:.4}.".format(
            loss, recall_k30, ndcg30, precision30, hit30, mrr30))

if __name__ == "__main__":
    sys.exit(fire.Fire(main))
