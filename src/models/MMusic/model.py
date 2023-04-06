#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from math import sqrt
from loguru import logger
from keras.layers import Convolution2D


class GAT(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0., bias=False, act=nn.ReLU(), **kwargs):
        super().__init__()
        self.act = act
        self.feat_drop = nn.Dropout(dropout) if dropout > 0 else None
        self.fc = nn.Linear(input_dim, output_dim, bias=True)
        self.fc.weight = torch.nn.init.xavier_uniform_(self.fc.weight)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        self_vecs, neigh_vecs = inputs

        if self.feat_drop is not None:
            self_vecs = self.feat_drop(self_vecs)
            neigh_vecs = self.feat_drop(neigh_vecs)

        # 为了下面的矩阵相乘，将[100,50]变成[100,1,50]
        self_vecs = torch.unsqueeze(self_vecs, 1)  # [batch, 1, embedding_size]
        neigh_self_vecs = torch.cat((neigh_vecs, self_vecs), dim=1)  # [batch, sample, embedding]  【100，10，50】

        score = self.softmax(torch.matmul(self_vecs, torch.transpose(neigh_self_vecs, 1, 2)))  #【100，1，50】与【100，50，10】乘
        context = torch.squeeze(torch.matmul(score, neigh_self_vecs), dim=1)  # 【100，1，10】和【100，10，50】乘
        output = self.act(self.fc(context))
        return output  # 【100，50】

class SelfAttention(nn.Module):
    dim_in: int
    dim_k: int
    dim_v: int

    def __init__(self, dim_in, dim_k, dim_v):
        super(SelfAttention, self).__init__()
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k)

    def forward(self, x):
        # x: batch, n, dim_in
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in

        q = self.linear_q(x)  # batch, n, dim_k
        k = self.linear_k(x)  # batch, n, dim_k
        v = self.linear_v(x)  # batch, n, dim_v

        dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact  # batch, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, n, n

        att = torch.bmm(dist, v)   # 矩阵乘法
        return att

class MMusic(torch.nn.Module):
    def __init__(
            self,
            hyper_param,
            num_layers,
    ):
        super(MMusic, self).__init__()
        self.act = hyper_param['act']
        self.batch_size = hyper_param['batch_size']
        self.num_users = hyper_param['num_users']
        self.num_items = hyper_param['num_items']
        self.num_times = hyper_param['num_times']
        self.num_artists = hyper_param['num_artists']
        self.num_muses = hyper_param['num_muses']


        self.embedding_size = hyper_param['embedding_size']
        self.max_length = hyper_param['max_length']
        self.dropout = hyper_param['dropout']
        self.num_layers = num_layers

        if self.act == 'relu':
            self.act = nn.ReLU()
        elif self.act == 'elu':
            self.act = nn.ELU()

        self.user_embedding = nn.Embedding(self.num_users,
                                           self.embedding_size)
        self.item_embedding = nn.Embedding(self.num_items,
                                           self.embedding_size,
                                           padding_idx=0)
        self.time_embedding = nn.Embedding(self.num_times,
                                           self.embedding_size,
                                           padding_idx=0)
        self.artist_embedding = nn.Embedding(self.num_artists,
                                           self.embedding_size,
                                           padding_idx=0)
        self.mus_embedding = nn.Embedding(self.num_muses,
                                             self.embedding_size,
                                             padding_idx=0)
        self.item_indices = nn.Parameter(torch.arange(0, self.num_items, dtype=torch.long),
                                         requires_grad=False)


        self.feat_drop = nn.Dropout(self.dropout) if self.dropout > 0 else None
        input_dim = self.embedding_size

        # making user embedding
        self.lstm = nn.LSTM(self.embedding_size, self.embedding_size, batch_first=True)

        self.Wu = nn.Linear(3 * self.embedding_size, self.embedding_size, bias=False)
        self.Wu.weight = torch.nn.init.xavier_uniform_(self.Wu.weight)

        # combine friend's long and short-term interest
        self.W1 = nn.Linear(2 * self.embedding_size, self.embedding_size, bias=False)
        self.W1.weight = torch.nn.init.xavier_uniform_(self.W1.weight)

        # combine user interest and social influence
        self.W2 = nn.Linear(input_dim + self.embedding_size, self.embedding_size, bias=False)
        self.W2.weight = torch.nn.init.xavier_uniform_(self.W2.weight)

        self.W7 = nn.Linear(7 * self.embedding_size, self.embedding_size, bias=False)
        self.W7.weight = torch.nn.init.xavier_uniform_(self.W7.weight)

        self.GAT = GAT(input_dim, input_dim, act=self.act, dropout=self.dropout)

        self.attr = SelfAttention(dim_in=self.embedding_size,dim_k=self.embedding_size,dim_v=self.embedding_size)

        #
        self.Wy = nn.Linear(self.embedding_size * 4, self.embedding_size)
        self.Wy.weight = torch.nn.init.xavier_uniform_(self.Wy.weight)

        self.Wa1 = nn.Linear(self.embedding_size, self.embedding_size)
        self.Wa1.weight = torch.nn.init.xavier_uniform_(self.Wa1.weight)

        # self.Wa2 = nn.Linear(256, 256)
        # self.Wa2.weight = torch.nn.init.xavier_uniform_(self.Wa2.weight)
        #
        # self.Wa3 = nn.Linear(256, self.embedding_size)
        # self.Wa3.weight = torch.nn.init.xavier_uniform_(self.Wa3.weight)

    #     self.conv =  Convolution2D(
    #     filters=50,  # 64个滤波器，生成64深度
    #     kernel_size=3,
    #     padding='same',  # same-padding
    #     # 因为是第一层，所以要定义inputshape
    #     input_shape=(100, 15),  # height & width
    #     # strides=2,
    #     kernel_initializer='random_uniform',
    #     bias_initializer='zeros',
    # )

    def individual_interest(self, input_session, timeid):
        input = input_session[0].long()  # input.shape : [max_length]
        emb_seqs = self.item_embedding(input)  # emb_seqs.shape : [max_length, embedding_dim]
        emb_seqs = torch.unsqueeze(emb_seqs, 0)

        if self.feat_drop is not None:
            emb_seqs = self.feat_drop(emb_seqs)

        for batch in range(self.batch_size - 1):
            input = input_session[batch + 1].long()
            # print(input.shape)
            emb_seq = self.item_embedding(input)
            # print(emb_seq.shape)
            emb_seq = torch.unsqueeze(emb_seq, 0)
            emb_seqs = torch.cat((emb_seqs, emb_seq), 0)


        # hu, (_, _) = self.lstm(emb_seqs)
        # hu = emb_seqs
        hu = self.attr(emb_seqs)
        # hu = torch.mean(hu, axis=1)
        # print(hu.shape)

        tl = timeid[0].long()
        # print(tl.shape)
        time_embs = self.time_embedding(tl)
        time_embs = torch.unsqueeze(time_embs, 0)

        if self.feat_drop is not None:
            time_embs = self.feat_drop(time_embs)

        for batch in range(self.batch_size - 1):
            t = timeid[batch + 1].long()
            time_emb = self.time_embedding(t)
            time_emb = torch.unsqueeze(time_emb, 0)
            time_embs = torch.cat((time_embs, time_emb), 0)

        # tu, (_, _) = self.lstm(time_embs)
        # tu = time_embs
        tu = self.attr(time_embs)
        # tu = torch.mean(tu, axis=1)
        # print(tu.shape)

        u = torch.cat((hu,tu), dim=2)

        # # art = torch.relu(self.Wa2(art))
        # u = torch.relu(self.Wa3(u))
        # print(u.shape)
        # u = torch.cat((iu,hu,tu),dim=1)
        # print(u.shape)
        # u = torch.relu(self.Wu(u))
        # print(u.shape)

        return u

    def music_interest(self, userid, artistid, instrumentalness, liveness, loudness, acousticness,
                         energy, mode, ke):

        n = len(instrumentalness)
        # instrumentalness
        instru = instrumentalness[0].long()  # val.shape : [50]
        # print(val.shape)
        instru_seqs = self.mus_embedding(instru)  # val_seqs.shape : [15, 50]
        instru_seqs = torch.unsqueeze(instru_seqs, 0)

        if self.feat_drop is not None:
            instru_seqs = self.feat_drop(instru_seqs)

        for batch in range(n - 1):
            instru = instrumentalness[batch + 1].long()
            instru_seq = self.mus_embedding(instru)
            instru_seq = torch.unsqueeze(instru_seq, 0)
            instru_seqs = torch.cat((instru_seqs, instru_seq), 0)
        # print(instru_seqs.shape)

        # liveness
        live = liveness[0].long()  # input.shape : [max_length]
        # print(val.shape)
        live_seqs = self.mus_embedding(live)  # emb_seqs.shape : [max_length, embedding_dim]
        live_seqs = torch.unsqueeze(live_seqs, 0)

        if self.feat_drop is not None:
            live_seqs = self.feat_drop(live_seqs)

        for batch in range(self.batch_size - 1):
            live = liveness[batch + 1].long()
            live_seq = self.mus_embedding(live)
            live_seq = torch.unsqueeze(live_seq, 0)
            live_seqs = torch.cat((live_seqs, live_seq), 0)

        # live_seqs = torch.relu(self.Wa1(live_seqs))

        # loudness
        loud = loudness[0].long()  # input.shape : [max_length]
        # loud = loudness[0]
        # print(loud)
        loud_seqs = self.mus_embedding(loud)  # emb_seqs.shape : [max_length, embedding_dim]
        loud_seqs = torch.unsqueeze(loud_seqs, 0)

        if self.feat_drop is not None:
            loud_seqs = self.feat_drop(loud_seqs)

        for batch in range(self.batch_size - 1):
            loud = loudness[batch + 1].long()
            loud_seq = self.mus_embedding(loud)
            loud_seq = torch.unsqueeze(loud_seq, 0)
            loud_seqs = torch.cat((loud_seqs, loud_seq), 0)

        # loud_seqs = torch.relu(self.Wa1(loud_seqs))

        # acousticness
        acou = acousticness[0].long()  # input.shape : [max_length]
        # print(val.shape)
        acou_seqs = self.mus_embedding(acou)  # emb_seqs.shape : [max_length, embedding_dim]
        acou_seqs = torch.unsqueeze(acou_seqs, 0)

        if self.feat_drop is not None:
            acou_seqs = self.feat_drop(acou_seqs)

        for batch in range(self.batch_size - 1):
            acou = acousticness[batch + 1].long()
            acou_seq = self.mus_embedding(acou)
            acou_seq = torch.unsqueeze(acou_seq, 0)
            acou_seqs = torch.cat((acou_seqs, acou_seq), 0)

        # acou_seqs = torch.relu(self.Wa1(acou_seqs))

        # energys
        ene = energy[0].long()  # input.shape : [max_length]
        # print(val.shape)
        ene_seqs = self.mus_embedding(ene)  # emb_seqs.shape : [max_length, embedding_dim]
        ene_seqs = torch.unsqueeze(ene_seqs, 0)

        if self.feat_drop is not None:
            ene_seqs = self.feat_drop(ene_seqs)

        for batch in range(self.batch_size - 1):
            ene = energy[batch + 1].long()
            ene_seq = self.mus_embedding(ene)
            ene_seq = torch.unsqueeze(ene_seq, 0)
            ene_seqs = torch.cat((ene_seqs, ene_seq), 0)

        # ene_seqs = torch.relu(self.Wa1(ene_seqs))

        # mode
        mods = mode[0].long()  # input.shape : [max_length]
        # print(val.shape)
        mods_seqs = self.mus_embedding(mods)  # emb_seqs.shape : [max_length, embedding_dim]
        mods_seqs = torch.unsqueeze(mods_seqs, 0)

        if self.feat_drop is not None:
            mods_seqs = self.feat_drop(mods_seqs)

        for batch in range(self.batch_size - 1):
            mods = mode[batch + 1].long()
            mods_seq = self.mus_embedding(mods)
            mods_seq = torch.unsqueeze(mods_seq, 0)
            mods_seqs = torch.cat((mods_seqs, mods_seq), 0)

        # mods_seqs = torch.relu(self.Wa1(mods_seqs))

        # key
        kes = ke[0].long()  # input.shape : [max_length]
        # print(val.shape)
        kes_seqs = self.mus_embedding(kes)  # emb_seqs.shape : [max_length, embedding_dim]
        kes_seqs = torch.unsqueeze(kes_seqs, 0)

        if self.feat_drop is not None:
            kes_seqs = self.feat_drop(kes_seqs)

        for batch in range(self.batch_size - 1):
            kes = ke[batch + 1].long()
            kes_seq = self.mus_embedding(kes)
            kes_seq = torch.unsqueeze(kes_seq, 0)
            kes_seqs = torch.cat((kes_seqs, kes_seq), 0)

        # kes_seqs = torch.relu(self.Wa1(kes_seqs))
        l = userid.long()  # 多少个id，用户数
        iu = self.user_embedding(l)  # user_emb = [100,50]
        # print(userid_emb.shape)

        # print(artistid.shape)    # artistid: [100,15]
        al = artistid[0].long()  # input.shape : [max_length]
        # print(tl.shape)
        artist_embs = self.artist_embedding(al)  # emb_seqs.shape : [max_length, embedding_dim]
        artist_embs = torch.unsqueeze(artist_embs, 0)

        if self.feat_drop is not None:
            artist_embs = self.feat_drop(artist_embs)

        for batch in range(self.batch_size - 1):
            t = artistid[batch + 1].long()
            artist_emb = self.artist_embedding(t)
            artist_emb = torch.unsqueeze(artist_emb, 0)
            artist_embs = torch.cat((artist_embs, artist_emb), 0)    # 【100，15，50】

        # artist_embs = torch.relu(self.Wa1(artist_embs))
        # art = self.attr(artist_embs)  # hu : [100,15,50]

        audio = torch.cat((instru_seqs, live_seqs, loud_seqs, acou_seqs, ene_seqs, mods_seqs, kes_seqs), dim=2)
        # print(audio.shape)

        audio = torch.relu(self.W7(audio))

        # audio = audio
        audio = self.attr(audio)
        # print(audio.shape)

        return  iu, audio, artist_embs
        # return iu, artist_embs
        # return iu, audio

    def fusion(self, hu, iu, audio, art):

        # hu shape [100,15,100]   iu shape [100,50]
        hu = torch.transpose(hu,0,1)   # 【15，100，100】

        us = torch.cat([hu[0], iu], dim=1)
        us = torch.unsqueeze(us, dim=0)
        for h in hu[1:]:
            u = torch.cat([h, iu], dim=1)
            u = torch.unsqueeze(u, dim=0)
            us = torch.cat([us, u], dim=0)

        # print(us.shape)
        u = torch.relu(self.Wu(u))


        u = self.attr(u)   # [15,100,50]
        # print(u.shape)
        u = torch.transpose(u, 0, 1)   # [100,15,50]

        i = torch.cat([audio, art],dim=2)    # [100,15,100]
        # print(i.shape)
        i = torch.relu(self.W1(i))   # [100,15,50]
        # i = torch.transpose(i, 0, 1)  # [15, 100, 50]
        # i = torch.mean(i, dim=0)
        # print(i.shape)   # [100, 50]

        return i, u
        # return u

    def score(self, u, i,  mask_y):

        pu = u * i
        # pu = u
        # print(pu.shape)
        # pu = torch.transpose(pu, 0, 2)
        pu = torch.transpose(pu, 0, 1)
        # print(pu.shape)
        logits = pu @ self.item_embedding(self.item_indices).t()
        # print(logits.shape)

        # # print(self.item_embedding(self.item_indices).t().shape)
        # logits = hu @ self.item_embedding(self.item_indices).t()
        # # logit shape : [15, 100, item_embedding]

        mask = mask_y.long()
        # print(mask.shape)
        logits = torch.transpose(logits, 0, 1)
        logits *= torch.unsqueeze(mask, 2)
        # print(logits)

        return logits

    def forward(self, feed_dict):
        labels = feed_dict['output_session']   # [100,10]
        # print(labels.shape)
        hu = self.individual_interest(feed_dict['input_session'],feed_dict['timeid'])
        # hu = self.individual_interest(feed_dict['input_session'])

        iu, audio, art = self.music_interest(
                                                feed_dict['userid'],
                                                feed_dict['artistid'],
                                                feed_dict['instrumentalness'],
                                                feed_dict['liveness'],
                                                feed_dict['loudness'],
                                                feed_dict['acousticness'],
                                                feed_dict['energy'],
                                                feed_dict['mode'],
                                                feed_dict['ke']
                                                # feed_dict['ke'],
                                               )

        # i, u = self.social_influence(hu, iu, art)
        i, u = self.fusion(hu, iu, audio, art)

        # score
        logits = self.score(u, i, feed_dict['mask_y'])
        # logits = self.score(hu, feed_dict['mask_y'])
        # print(logits.shape)

        # metric
        recall10, recall20, recall30 = self._recall(logits, labels)
        ndcg10, ndcg20, ndcg30 = self._ndcg(logits, labels, feed_dict['mask_y'])
        precision10, precision20, precision30 = self._precision(logits, labels)
        hit10, hit20, hit30 = self._hit(logits, labels)
        mrr10, mrr20, mrr30 = self._mrr(logits, labels, feed_dict['mask_y'])

        # loss
        logits = (torch.transpose(logits, 1, 2)).to(dtype=torch.float)  # logits : [batch, item_embedding, max_length]
        labels = labels.long()  # labels : [batch, max_length]

        loss = F.cross_entropy(logits, labels)
        # print(loss)
        # print(...)

        return loss, recall10.item(), recall20.item(), recall30.item(), \
               ndcg10.item(), ndcg20.item(), ndcg30.item(), \
               precision10.item(), precision20.item(), precision30.item(), \
               hit10.item(), hit20.item(), hit30.item(), \
               mrr10.item(), mrr20.item(), mrr30.item()  # loss, recall_k, ndcg

    def predict(self, feed_dict):
        labels = feed_dict['output_session']

        # hu = self.individual_interest(feed_dict['input_session'])
        hu = self.individual_interest(feed_dict['input_session'], feed_dict['timeid'])

        iu, audio, art = self.music_interest(
            feed_dict['userid'],
            feed_dict['artistid'],
            feed_dict['instrumentalness'],
            feed_dict['liveness'],
            feed_dict['loudness'],
            feed_dict['acousticness'],
            feed_dict['energy'],
            feed_dict['mode'],
            feed_dict['ke']
            # feed_dict['ke'],
        )

        # i, u = self.social_influence(hu, iu, art)
        i, u = self.fusion(hu, iu, audio, art)

        # score
        logits = self.score(u, i, feed_dict['mask_y'])
        # logits = self.score(hu, feed_dict['mask_y'])
        # nn.utils.clip_grad_norm(MMusic.parameters(self),max_norm=20,norm_type=2)

        # metric
        recall10, recall20, recall30 = self._recall(logits, labels)
        ndcg10, ndcg20, ndcg30 = self._ndcg(logits, labels, feed_dict['mask_y'])
        precision10, precision20, precision30 = self._precision(logits, labels)
        hit10, hit20, hit30 = self._hit(logits, labels)
        mrr10, mrr20, mrr30 = self._mrr(logits, labels, feed_dict['mask_y'])

        # loss
        logits = (torch.transpose(logits, 1, 2)).to(dtype=torch.float)  # logits : [batch, item_embedding, max_length]
        labels = labels.long()  # labels : [batch, max_length]
        # print(logits)
        loss = F.cross_entropy(logits, labels)
        # print(loss)

        return loss, recall10.item(), recall20.item(), recall30.item(), \
               ndcg10.item(), ndcg20.item(), ndcg30.item(), \
               precision10.item(), precision20.item(), precision30.item(), \
               hit10.item(), hit20.item(), hit30.item(), \
               mrr10.item(), mrr20.item(), mrr30.item()  # loss, recall_k, ndcg

    def _recall(self, predictions, labels):
        batch_size = predictions.shape[0]
        _, top_k10_index = torch.topk(predictions, k=10, dim=2)  # top_k_index : [batch, max_length, k]
        _, top_k20_index = torch.topk(predictions, k=20, dim=2)  # top_k_index : [batch, max_length, k]
        _, top_k30_index = torch.topk(predictions, k=30, dim=2)  # top_k_index : [batch, max_length, k]

        labels = labels.long()
        labels = torch.unsqueeze(labels, dim=2)  # labels : [batch, max_length, 1]
        # print(labels[99])
        corrects10 = (top_k10_index == labels) * (labels != 0)  # corrects : [batch, max_length, k]
        corrects20 = (top_k20_index == labels) * (labels != 0)  # corrects : [batch, max_length, k]
        corrects30 = (top_k30_index == labels) * (labels != 0)  # corrects : [batch, max_length, k]
        recall_corrects10 = torch.sum(corrects10, dim=2).to(dtype=torch.float)  # corrects : [batch, max_length]
        recall_corrects20 = torch.sum(corrects20, dim=2).to(dtype=torch.float)  # corrects : [batch, max_length]
        recall_corrects30 = torch.sum(corrects30, dim=2).to(dtype=torch.float)  # corrects : [batch, max_length]

        mask_sum = (labels != 0).sum(dim=1)  # mask_sum : [batch, 1]
        mask_sum = torch.squeeze(mask_sum, dim=1)  # mask_sum : [batch]

        recall_k10 = (recall_corrects10.sum(dim=1) / mask_sum).sum()
        recall_k20 = (recall_corrects20.sum(dim=1) / mask_sum).sum()
        recall_k30 = (recall_corrects30.sum(dim=1) / mask_sum).sum()
        # print(recall_k10 == recall_k20)

        return recall_k10 / batch_size, recall_k20 / batch_size, recall_k30 / batch_size

    def _ndcg(self, logits, labels, mask):
        # print(labels.shape)
        # print(mask.shape)
        num_items = logits.shape[2]
        logits = torch.reshape(logits, (logits.shape[0] * logits.shape[1], logits.shape[2]))
        # print(logits.shape)

        predictions = torch.transpose(logits, 0, 1)

        labels = labels.long()
        targets = torch.reshape(labels, [-1])
        # print(targets.shape)
        # print(predictions[targets].shape)
        pred_values = torch.unsqueeze(torch.diagonal(predictions[targets]), -1)
        # print(pred_values)
        tile_pred_values = torch.tile(pred_values, [1, num_items])
        # print(pred_values.repeat(1, num_items))
        # print(pred_values.repeat(1, 10))
        # print(tile_pred_values.shape)
        # print((torch.sum((logits > tile_pred_values).type(torch.float), -1)).shape)
        ranks = torch.sum((logits > tile_pred_values).type(torch.float), -1) + 1
        # print(ranks)
        # print(ranks10)
        # print(ranks)
        # ranks10 = ranks[:]
        ranks10 = torch.zeros(logits.shape[0])
        ranks20 = torch.zeros(logits.shape[0])
        ranks30 = torch.zeros(logits.shape[0])
        for i in range(len(logits)):
            # print(i)
            # print(ranks10[i])
            if ranks[i] >= 10:
                ranks10[i] = float('inf')
            else:
                ranks10[i] = ranks[i]
            if ranks[i] >= 20:
                ranks20[i] = float('inf')
            else:
                ranks20[i] = ranks[i]
            if ranks[i] >= 30:
                ranks30[i] = float('inf')
            else:
                ranks30[i] = ranks[i]

        # print(ranks10)
        # print(ranks20)
        # print(ranks30)
        ndcg = 1. / (torch.log2(1.0 + ranks))
        ndcg10 = 1. / (torch.log2(1.0 + ranks10))
        ndcg20 = 1. / (torch.log2(1.0 + ranks20))
        ndcg30 = 1. / (torch.log2(1.0 + ranks30))
        # print(torch.sum(ndcg))
        # print(torch.sum(ndcg10))
        # print(torch.sum(ndcg20))
        # print(torch.sum(ndcg30))

        # print(ndcg10 == ndcg20)

        mask_sum = torch.sum(mask)
        # print(mask_sum)
        mask = torch.reshape(mask, [-1])
        ndcg10 *= mask
        ndcg20 *= mask
        ndcg30 *= mask

        return torch.sum(ndcg10) / mask_sum, torch.sum(ndcg20) / mask_sum, torch.sum(ndcg30) / mask_sum

    def _precision(self, predictions, labels):
        batch_size = predictions.shape[0]
        # _, top_k_index = torch.topk(predictions, k=30, dim=2)  # top_k_index : [batch, max_length, k]
        # _, top_k_index = torch.topk(predictions, k=10, dim=2)  # top_k_index : [batch, max_length, k]
        _, top_k10_index = torch.topk(predictions, k=10, dim=2)  # top_k_index : [batch, max_length, k]
        _, top_k20_index = torch.topk(predictions, k=20, dim=2)  # top_k_index : [batch, max_length, k]
        _, top_k30_index = torch.topk(predictions, k=30, dim=2)  # top_k_index : [batch, max_length, k]
        topk10 = len(torch.transpose(top_k10_index, 0, 2))
        topk20 = len(torch.transpose(top_k20_index, 0, 2))
        topk30 = len(torch.transpose(top_k30_index, 0, 2))
        labels = labels.long()
        labels = torch.unsqueeze(labels, dim=2)  # labels : [batch, max_length, 1]
        corrects10 = (top_k10_index == labels) * (labels != 0)  # corrects : [batch, max_length, k]  相当于TP
        corrects20 = (top_k20_index == labels) * (labels != 0)  # corrects : [batch, max_length, k]  相当于TP
        corrects30 = (top_k30_index == labels) * (labels != 0)  # corrects : [batch, max_length, k]  相当于TP
        precision_corrects10 = torch.sum(corrects10, dim=2).to(dtype=torch.float)  # corrects : [batch, max_length]
        precision_corrects20 = torch.sum(corrects20, dim=2).to(dtype=torch.float)  # corrects : [batch, max_length]
        precision_corrects30 = torch.sum(corrects30, dim=2).to(dtype=torch.float)  # corrects : [batch, max_length]

        # pre_pos = top_k_values.sum(dim=2) # [batch, max_length]
        # pre_pos_sum = (pre_pos != 0).sum(dim=1)  # [batch] 每个用户预测的20个东西中有几个预测的是正样本

        precision_k10 = (precision_corrects10.sum(dim=1) / topk10).sum()
        precision_k20 = (precision_corrects20.sum(dim=1) / topk20).sum()
        precision_k30 = (precision_corrects30.sum(dim=1) / topk30).sum()

        return precision_k10 / batch_size, precision_k20 / batch_size, precision_k30 / batch_size

    def _hit(self,predictions,labels):
        batch_size = predictions.shape[0]

        _, top_k10_index = torch.topk(predictions, k=10, dim=2)  # top_k_index : [batch, max_length, k]
        _, top_k20_index = torch.topk(predictions, k=20, dim=2)  # top_k_index : [batch, max_length, k]
        _, top_k30_index = torch.topk(predictions, k=30, dim=2)  # top_k_index : [batch, max_length, k]

        labels = labels.long()
        labels = torch.unsqueeze(labels, dim=2)  # labels : [batch, max_length, 1]
        # print(labels[99])
        corrects10 = (top_k10_index == labels) * (labels != 0)  # corrects : [batch, max_length, k]
        corrects20 = (top_k20_index == labels) * (labels != 0)  # corrects : [batch, max_length, k]
        corrects30 = (top_k30_index == labels) * (labels != 0)  # corrects : [batch, max_length, k]
        hit_corrects10 = torch.sum(corrects10, dim=2).to(dtype=torch.float)  # corrects : [batch, max_length]
        hit_corrects20 = torch.sum(corrects20, dim=2).to(dtype=torch.float)  # corrects : [batch, max_length]
        hit_corrects30 = torch.sum(corrects30, dim=2).to(dtype=torch.float)  # corrects : [batch, max_length]

        mask_sum = (labels != 0).sum(dim=1)  # mask_sum : [batch, 1]
        mask_sum = torch.squeeze(mask_sum, dim=1)  # mask_sum : [batch]
        hit_k10 = 0
        hit_k20 = 0
        hit_k30 = 0

        for b in range(batch_size):
            if hit_corrects10.sum(dim=1)[b] > 0:
                hit_k10 += 1
            if hit_corrects20.sum(dim=1)[b] > 0:
                hit_k20 += 1
            if hit_corrects30.sum(dim=1)[b] > 0:
                hit_k30 += 1

        # sum = (hit_corrects.sum(dim=1)).sum()
        # if sum > self.max_length:
        #     hit_k += 1
        # hit_k = (hit_corrects.sum(dim=1)).sum()
        # print(hit_k30)

        return torch.tensor(hit_k10 / batch_size), torch.tensor(hit_k20 / batch_size), torch.tensor(hit_k30 / batch_size)

    def _mrr(self, logits, labels, mask):
        # print(labels.shape)
        # print(mask.shape)
        num_items = logits.shape[2]
        logits = torch.reshape(logits, (logits.shape[0] * logits.shape[1], logits.shape[2]))
        # print(logits.shape)

        predictions = torch.transpose(logits, 0, 1)

        labels = labels.long()
        targets = torch.reshape(labels, [-1])
        # print(targets.shape)
        # print(predictions[targets].shape)
        pred_values = torch.unsqueeze(torch.diagonal(predictions[targets]), -1)
        # print(pred_values)
        tile_pred_values = torch.tile(pred_values, [1, num_items])
        # print(pred_values.repeat(1, num_items))
        # print(pred_values.repeat(1, 10))
        # print(tile_pred_values.shape)
        # print((torch.sum((logits > tile_pred_values).type(torch.float), -1)).shape)
        ranks = torch.sum((logits > tile_pred_values).type(torch.float), -1) + 1
        # print(ranks)
        # print(ranks10)
        # print(ranks)
        # ranks10 = ranks[:]
        ranks10 = torch.zeros(logits.shape[0])
        ranks20 = torch.zeros(logits.shape[0])
        ranks30 = torch.zeros(logits.shape[0])
        for i in range(len(logits)):
            # print(i)
            # print(ranks10[i])
            if ranks[i] >= 10:
                ranks10[i] = float('inf')
            else:
                ranks10[i] = ranks[i]
            if ranks[i] >= 20:
                ranks20[i] = float('inf')
            else:
                ranks20[i] = ranks[i]
            if ranks[i] >= 30:
                ranks30[i] = float('inf')
            else:
                ranks30[i] = ranks[i]

        # print(ranks10)
        # print(ranks20)
        # print(ranks30)
        ndcg = 1. / ranks
        mrr10 = 1. / ranks10
        mrr20 = 1. / ranks20
        mrr30 = 1. / ranks30
        # print(torch.sum(ndcg))
        # print(torch.sum(ndcg10))
        # print(torch.sum(ndcg20))
        # print(torch.sum(ndcg30))

        # print(ndcg10 == ndcg20)

        mask_sum = torch.sum(mask)
        # print(mask_sum)
        mask = torch.reshape(mask, [-1])
        mrr10 *= mask
        mrr20 *= mask
        mrr30 *= mask

        return torch.sum(mrr10) / mask_sum, torch.sum(mrr20) / mask_sum, torch.sum(mrr30) / mask_sum