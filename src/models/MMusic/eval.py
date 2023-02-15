#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
from src.models.MMusic.model import MMusic
from src.models.MMusic.batch.minibatch import MinibatchIterator
from tqdm import tqdm
import torch.nn.functional as F


class MyEvaluator:
    def __init__(self, device):
        self.device = device

    def evaluate(self, model, minibatch, mode='test'):
        val_loss = []
        val_recall10 = []
        val_ndcg10 = []
        val_precision10 = []
        val_hit10 = []
        val_mrr10 = []

        val_recall20 = []
        val_ndcg20 = []
        val_precision20 = []
        val_hit20 = []
        val_mrr20 = []

        val_recall30 = []
        val_ndcg30 = []
        val_precision30 = []
        val_hit30 = []
        val_mrr30 = []
        with torch.no_grad():
            model.eval()

            while not minibatch.end_val(mode):
                feed_dict = minibatch.next_val_minibatch_feed_dict(mode)
                loss, recall_k10, recall_k20, recall_k30, \
                ndcg10, ndcg20, ndcg30, \
                precision10, precision20, precision30, \
                hit10, hit20, hit30, \
                mrr10, mrr20, mrr30 = model.predict(feed_dict)

                val_loss.append(loss.item())
                val_recall10.append(recall_k10)
                val_ndcg10.append(ndcg10)
                val_precision10.append(precision10)
                val_hit10.append(hit10)
                val_mrr10.append(mrr10)

                val_recall20.append(recall_k20)
                val_ndcg20.append(ndcg20)
                val_precision20.append(precision20)
                val_hit20.append(hit20)
                val_mrr20.append(mrr20)

                val_recall30.append(recall_k30)
                val_ndcg30.append(ndcg30)
                val_precision30.append(precision30)
                val_hit30.append(hit30)
                val_mrr30.append(mrr30)

        return np.mean(val_loss), np.mean(val_recall10), np.mean(val_recall20), np.mean(val_recall30),\
               np.mean(val_ndcg10), np.mean(val_ndcg20), np.mean(val_ndcg30), \
               np.mean(val_precision10), np.mean(val_precision20), np.mean(val_precision30),\
               np.mean(val_hit10), np.mean(val_hit20), np.mean(val_hit30), \
               np.mean(val_mrr10), np.mean(val_mrr20), np.mean(val_mrr30)

