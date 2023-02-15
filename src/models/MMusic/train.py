#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import matplotlib.pyplot as plt
from src.models.MMusic.model import MMusic
from src.models.MMusic.eval import MyEvaluator
from src.models.MMusic.batch.minibatch import MinibatchIterator
from tqdm import tqdm
from loguru import logger

class MyTrainer:
    def __init__(self, device):
        self.device = device
        self.train_losses = []
        self.train_recall10 = []
        self.train_ndcg10 = []
        self.train_precision10 = []
        self.train_hit10 = []
        self.train_mrr10 = []
        self.val_losses = []
        self.val_recall10 = []
        self.val_ndcg10 = []
        self.val_precision10 = []
        self.val_hit10 = []
        self.val_mrr10 = []

        self.train_recall20 = []
        self.train_ndcg20 = []
        self.train_precision20 = []
        self.train_hit20 = []
        self.train_mrr20 = []
        self.val_recall20 = []
        self.val_ndcg20 = []
        self.val_precision20 = []
        self.val_hit20 = []
        self.val_mrr20 = []

        self.train_recall30 = []
        self.train_ndcg30 = []
        self.train_precision30 = []
        self.train_hit30 = []
        self.train_mrr30 = []
        self.val_recall30 = []
        self.val_ndcg30 = []
        self.val_precision30 = []
        self.val_hit30 = []
        self.val_mrr30 = []

    def train_with_hyper_param(self, minibatch, hyper_param):
        seed = hyper_param['seed']
        epochs = hyper_param['epochs']
        learning_rate = hyper_param['learning_rate']
        data_name = hyper_param['data_name']
        embedding_size = hyper_param['embedding_size']
        decay_rate = hyper_param['decay_rate']

        model = MMusic(hyper_param, num_layers=2).to(self.device)
        evaluator = MyEvaluator(device=self.device)

        patience = 20
        inc30 = 0
        early_stopping = False
        highest_val_ndcg30 = 0

        batch_len = minibatch.train_batch_len()

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=batch_len // 10, gamma=decay_rate)

        pbar = tqdm(range(epochs), position=0, leave=False, desc='epoch')

        for epoch in pbar:
            total_loss = 0
            total_recall10 = 0
            total_ndcg10 = 0
            total_precision10 = 0
            total_hit10 = 0
            total_mrr10 = 0

            total_recall20 = 0
            total_ndcg20 = 0
            total_precision20 = 0
            total_hit20 = 0
            total_mrr20 = 0

            total_recall30 = 0
            total_ndcg30 = 0
            total_precision30 = 0
            total_hit30 = 0
            total_mrr30 = 0

            minibatch.shuffle()

            for batch in tqdm(range(batch_len), position=1, leave=False, desc='batch'):
                model.train()
                optimizer.zero_grad()

                feed_dict = minibatch.next_train_minibatch_feed_dict()

                # train
                loss, recall_k10, recall_k20, recall_k30, \
                ndcg10, ndcg20, ndcg30, \
                precision10, precision20, precision30, \
                hit10, hit20, hit30, \
                mrr10, mrr20, mrr30 = model(feed_dict)

                loss.backward()
                optimizer.step()
                scheduler.step()

                # log
                total_loss += loss.item()
                total_recall10 += recall_k10
                total_ndcg10 += ndcg10
                total_precision10 += precision10
                total_hit10 += hit10
                total_mrr10 += mrr10

                total_recall20 += recall_k20
                total_ndcg20 += ndcg20
                total_precision20 += precision20
                total_hit20 += hit20
                total_mrr20 += mrr20

                total_recall30 += recall_k30
                total_ndcg30 += ndcg30
                total_precision30 += precision30
                total_hit30 += hit30
                total_mrr30 += mrr30

                self.train_recall10.append(recall_k10)
                self.train_ndcg10.append(ndcg10)
                self.train_precision10.append(precision10)
                self.train_hit10.append(hit10)
                self.train_mrr10.append(mrr10)

                self.train_recall20.append(recall_k20)
                self.train_ndcg20.append(ndcg20)
                self.train_precision20.append(precision20)
                self.train_hit20.append(hit20)
                self.train_mrr20.append(mrr20)

                self.train_recall30.append(recall_k30)
                self.train_ndcg30.append(ndcg30)
                self.train_precision30.append(precision30)
                self.train_hit30.append(hit30)
                self.train_mrr30.append(mrr30)

                # validation
                if (batch % int(batch_len / 10)) == 0:
                    val_loss, val_recall_k10, val_recall_k20, val_recall_k30, \
                    val_ndcg10, val_ndcg20, val_ndcg30, \
                    val_precision10, val_precision20, val_precision30, \
                    val_hit10, val_hit20, val_hit30, \
                    val_mrr10, val_mrr20, val_mrr30, = evaluator.evaluate(model, minibatch, mode='val')

                    self.val_recall10.append(val_recall_k10)
                    self.val_ndcg10.append(val_ndcg10)
                    self.val_precision10.append(val_precision10)
                    self.val_hit10.append(val_hit10)
                    self.val_mrr10.append(val_mrr10)

                    self.val_recall20.append(val_recall_k20)
                    self.val_ndcg20.append(val_ndcg20)
                    self.val_precision20.append(val_precision20)
                    self.val_hit20.append(val_hit20)
                    self.val_mrr20.append(val_mrr20)

                    self.val_recall30.append(val_recall_k30)
                    self.val_ndcg30.append(val_ndcg30)
                    self.val_precision30.append(val_precision30)
                    self.val_hit30.append(val_hit30)
                    self.val_mrr30.append(val_mrr30)

                    if val_ndcg30 >= highest_val_ndcg30:
                        highest_val_ndcg30 = val_ndcg30
                        inc30 = 0
                    else:
                        inc30 += 1

                # early stopping
                if inc30 >= patience:
                    early_stopping = True
                    break

            if early_stopping:
                pbar.write('Early stop at epoch: {}, batch steps: {}'.format(epoch+1, batch))
                pbar.update(pbar.total)
                break

            pbar.write(
                'Top10:Epoch {:02}: train loss: {:.4}\t  train recall@20: {:.4}\t  train NDCG: {:.4}\t  train precision: {:.4}\t  train hit: {:.4}\t  train mrr: {:.4}'
                    .format(epoch + 1, total_loss / batch_len, total_recall10 / batch_len, total_ndcg10 / batch_len,
                            total_precision10 / batch_len, total_hit10 / batch_len, total_mrr10 / batch_len))
            pbar.write(
                'Top10:Epoch {:02}: valid loss: {:.4}\t  valid recall@20: {:.4}\t  valid NDCG: {:.4}\t  valid precision: {:.4}\t  valid hit: {:.4}\t  valid mrr: {:.4}\n'
                    .format(epoch + 1, val_loss, val_recall_k10, val_ndcg10, val_precision10, val_hit10, val_mrr10))
            pbar.write(
                'Top20:Epoch {:02}: train loss: {:.4}\t  train recall@20: {:.4}\t  train NDCG: {:.4}\t  train precision: {:.4}\t  train hit: {:.4}\t  train mrr: {:.4}'
                    .format(epoch + 1, total_loss / batch_len, total_recall20 / batch_len, total_ndcg20 / batch_len,
                            total_precision20 / batch_len, total_hit20 / batch_len, total_mrr20 / batch_len))
            pbar.write(
                'Top20:Epoch {:02}: valid loss: {:.4}\t  valid recall@20: {:.4}\t  valid NDCG: {:.4}\t  valid precision: {:.4}\t  valid hit: {:.4}\t  valid mrr: {:.4}\n'
                    .format(epoch + 1, val_loss, val_recall_k20, val_ndcg20, val_precision20, val_hit20, val_mrr20))
            pbar.write(
                'Top30:Epoch {:02}: train loss: {:.4}\t  train recall@20: {:.4}\t  train NDCG: {:.4}\t  train precision: {:.4}\t  train hit: {:.4}\t  train mrr: {:.4}'
                    .format(epoch + 1, total_loss / batch_len, total_recall30 / batch_len, total_ndcg30 / batch_len,
                            total_precision30 / batch_len, total_hit30 / batch_len, total_mrr30 / batch_len))
            pbar.write(
                'Top30:Epoch {:02}: valid loss: {:.4}\t  valid recall@20: {:.4}\t  valid NDCG: {:.4}\t  valid precision: {:.4}\t  valid hit: {:.4}\t  valid mrr: {:.4}\n'
                    .format(epoch + 1, val_loss, val_recall_k30, val_ndcg30, val_precision30, val_hit30, val_mrr30))
            pbar.update()

        pbar.close()

        return model
