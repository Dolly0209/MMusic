# Template code is provided at the
# https://github.com/jbnu-dslab/DGRec-pytorch/blob/master/src/models/DGRec/batch/minibatch.py

import numpy as np
import pandas as pd
import sys
import torch
sys.path.append("../..") # Adds higher directory to python modules path.

# from models.MMusic.batch.neigh_samplers import UniformNeighborSampler

np.random.seed(123)


class MinibatchIterator(object):
    def __init__(self,
                 data,
                 hyper_param,
                 device='cpu',
                 training=True,
                 ):
        # self.num_layers = 2 # Currently, only 2 layer is supported.
        # self.adj_info = data[0]
        # self.latest_sessions = data[1]
        self.training = training
        # self.train_df, self.valid_df, self.test_df = data[4], data[5], data[6]
        self.train_df,self.valid_df,self.test_df = data[0],data[1],data[2]
        self.device = device
        self.all_data = pd.concat([data[0], data[1], data[2]])
        self.placeholders={
            'timeid':'timeid',
            'userid':'userid',
            'artistid':'artistid',
            'instrumentalness': 'instrumentalness',
            'liveness': 'liveness',
            'loudness': 'loudness',
            'acousticness': 'acousticness',
            'energy': 'energy',
            'mode':'mode',
            'ke':'ke',
            'ns' :'ns',

            'input_x': 'input_session',
            'input_y': 'output_session',
            'mask_y': 'mask_y',
            'support_nodes_layer1': 'support_nodes_layer1',
            'support_nodes_layer2': 'support_nodes_layer2',
            'support_sessions_layer1': 'support_sessions_layer1',
            'support_sessions_layer2': 'support_sessions_layer2',
            'support_lengths_layer1': 'support_lengths_layer1',
            'support_lengths_layer2': 'support_lengths_layer2',
        }
        self.batch_size = hyper_param['batch_size']
        self.max_degree = 50
        # self.num_nodes = len(data[2])
        self.num_nodes = 24373
        # self.num_items = len(data[3])
        self.num_items = 90872
        self.max_length = hyper_param['max_length']
        # self.visible_time = self.user_visible_time()
        # self.test_adj, self.test_deg = self.construct_test_adj()
        if self.training:
            # self.adj, self.deg = self.construct_adj()
            # self.train_session_ids = self._remove_infoless(self.train_df, self.adj, self.deg)
            self.train_user_ids = self._remove_infoless(self.train_df)
            self.valid_user_ids = self._remove_infoless(self.valid_df)
            # self.sampler = UniformNeighborSampler(self.adj, self.visible_time, self.deg)
        
        # self.test_session_ids = self._remove_infoless(self.test_df, self.test_adj, self.test_deg)
        self.test_user_ids = self._remove_infoless(self.test_df)
        # 这里的mask，是用户兴趣音乐的mask
        self.padded_data, self.mask, self.artid, self.timeid,\
        self.instrus, self.lives, self.louds, self.acous, self.eners, self.modes, self.kes, self.ns = self._padding_sessions(self.all_data)
        # self.test_sampler = UniformNeighborSampler(self.test_adj, self.visible_time, self.test_deg)
        
        self.batch_num = 0
        self.batch_num_val = 0
        self.batch_num_test = 0

    def user_visible_time(self):
        '''
            Find out when each user is 'visible' to her friends, i.e., every user's first click/watching time.
        '''
        visible_time = []
        for l in self.latest_sessions:
            timeid = max(loc for loc, val in enumerate(l) if val == 'NULL' and loc < len(l)) + 1
            visible_time.append(timeid)
            assert timeid > 0 and timeid <= len(l), 'Wrong when create visible time {}'.format(timeid)
        return visible_time

    # def _remove_infoless(self, data, adj, deg):
    def _remove_infoless(self, data):
        # data = data.loc[deg[data['user_id']] != 0]
        reserved_user_ids = []
        print('users: {}\tlistening: {}'.format(data.user_id.nunique(), len(data)))
        data = data.groupby('user_id')['track_id'].apply(list).to_dict()
        for k, v in data.items():

            # mask = np.ones(self.max_length, dtype=np.float32)
            x = v[:-1]
            if len(x) > 0:
                reserved_user_ids.append(k)
                # print(k)
                # print(v)
        return reserved_user_ids

    def _padding_sessions(self, data):
        '''
        Pad zeros at the end of each session to length self.max_length for batch training.
        '''
        data1 = data.sort_values(by="created_at").groupby('user_id')['track_id'].apply(list).to_dict()
        times = data.groupby('user_id')['time'].apply(list).to_dict()
        artis = data.groupby('user_id')['artist_id'].apply(list).to_dict()
        instrumentalness = data.groupby('user_id')['0'].apply(list).to_dict()
        liveness = data.groupby('user_id')['1'].apply(list).to_dict()
        loudness = data.groupby('user_id')['5'].apply(list).to_dict()
        acousticness = data.groupby('user_id')['7'].apply(list).to_dict()
        energy = data.groupby('user_id')['8'].apply(list).to_dict()
        mode = data.groupby('user_id')['9'].apply(list).to_dict()
        ke = data.groupby('user_id')['10'].apply(list).to_dict()

        new_data = {}
        data_mask = {}
        timeids = {}
        artids = {}
        instrumentalnesss = {}
        livenesss = {}
        loudnesss = {}
        acousticnesss = {}
        energys = {}
        modes = {}
        kes = {}
        ns = {}
        for k, v in data1.items():
            mask = np.ones(self.max_length, dtype=np.float32)
            x = v[:-1]
            y = v[1: ]
            # assert len(x) > 0
            padded_len = self.max_length - len(x)
            if padded_len > 0:
                x.extend([0] * padded_len)
                y.extend([0] * padded_len)
                mask[-padded_len: ] = 0.
            v.extend([0] * (self.max_length - len(v)))
            x = x[:self.max_length]
            y = y[:self.max_length]
            v = v[:self.max_length]
            new_data[k] = [np.array(x, dtype=np.int32), np.array(y, dtype=np.int32), np.array(v, dtype=np.int32)]
            data_mask[k] = np.array(mask, dtype=bool)

        for k, v in times.items():
            v.extend([0] * (self.max_length - len(v)))

            v = v[:self.max_length]
            timeids[k] = np.array(v, dtype=np.int32)

        for k, v in artis.items():
            v.extend([0] * (self.max_length - len(v)))
            v = v[:self.max_length]
            artids[k] = np.array(v, dtype=np.int32)

        for k, v in instrumentalness.items():
            ns[k] = len(v)
            v.extend([0] * (self.max_length - len(v)))
            v = v[:self.max_length]
            instrumentalnesss[k] = np.array(v, dtype=np.float32)

        for k, v in loudness.items():
            v.extend([0] * (self.max_length - len(v)))
            v = v[:self.max_length]
            loudnesss[k] = np.array(v, dtype=np.float32)

        for k, v in liveness.items():
            v.extend([0] * (self.max_length - len(v)))
            v = v[:self.max_length]
            livenesss[k] = np.array(v, dtype=np.float32)

        for k, v in acousticness.items():
            v.extend([0] * (self.max_length - len(v)))
            v = v[:self.max_length]
            acousticnesss[k] = np.array(v, dtype=np.float32)
            # print(k)
            # print(v)

        for k, v in energy.items():
            v.extend([0] * (self.max_length - len(v)))
            v = v[:self.max_length]
            energys[k] = np.array(v, dtype=np.float32)

        for k, v in mode.items():
            v.extend([0] * (self.max_length - len(v)))
            v = v[:self.max_length]
            modes[k] = np.array(v, dtype=np.float32)

        for k, v in ke.items():
            v.extend([0] * (self.max_length - len(v)))
            v = v[:self.max_length]
            kes[k] = np.array(v, dtype=np.float32)

        return new_data, data_mask, artids, timeids, \
               instrumentalnesss, livenesss, loudnesss, acousticnesss, energys, modes, kes, ns
        # return new_data, data_mask, timeids

    def _batch_feed_dict(self, current_batch):
        '''
        Construct batch inputs.
        '''
        # initialize
        current_batch_user_ids = current_batch
        # print(current_batch_user_ids)
        # current_batch_sess_ids, samples, support_sizes = current_batch
        feed_dict = {}
        input_x = []
        input_y = []
        mask_y = []
        userids = []
        timeids = []
        artids = []
        instrus = []
        lives = []
        louds = []
        acous = []
        eners = []
        modes = []
        kes = []
        ns = []


        # input_x / input_y / mask_y
        for userid in current_batch_user_ids:
            # nodeid, timeid = sessid.split('_')
            # timeids.append(int(timeid))
            x, y, _ = self.padded_data[userid]
            mask = self.mask[userid]
            timeid = self.timeid[userid]
            artid = self.artid[userid]
            instru = self.instrus[userid]
            live = self.lives[userid]
            loud = self.louds[userid]
            acou = self.acous[userid]
            ener = self.eners[userid]
            mode = self.modes[userid]
            ke = self.kes[userid]
            n = self.ns[userid]

            timeids.append(timeid)
            artids.append(artid)
            instrus.append(instru)
            louds.append(loud)
            lives.append(live)
            acous.append(acou)
            modes.append(mode)
            eners.append(ener)
            kes.append(ke)
            ns.append(n)

            input_x.append(x)
            input_y.append(y)
            userids.append(userid)
            mask_y.append(mask)
        # print(input_x)
        # print(input_y)

        feed_dict.update({self.placeholders['timeid']: torch.tensor(np.array(timeids)).to(self.device)})
        feed_dict.update({self.placeholders['artistid']: torch.tensor(np.array(artids)).to(self.device)})
        feed_dict.update({self.placeholders['instrumentalness']: torch.tensor(np.array(instrus)).to(self.device)})
        feed_dict.update({self.placeholders['loudness']: torch.tensor(np.array(louds)).to(self.device)})
        feed_dict.update({self.placeholders['liveness']: torch.tensor(np.array(lives)).to(self.device)})
        feed_dict.update({self.placeholders['energy']: torch.tensor(np.array(eners)).to(self.device)})
        feed_dict.update({self.placeholders['acousticness']: torch.tensor(np.array(acous)).to(self.device)})
        feed_dict.update({self.placeholders['mode']: torch.tensor(np.array(modes)).to(self.device)})
        feed_dict.update({self.placeholders['ke']: torch.tensor(np.array(kes)).to(self.device)})
        feed_dict.update({self.placeholders['ns']: torch.tensor(np.array(ns)).to(self.device)})


        feed_dict.update({self.placeholders['userid']: torch.tensor(np.array(userids)).to(self.device)})
        feed_dict.update({self.placeholders['input_x']: torch.tensor(np.array(input_x)).to(self.device)})
        feed_dict.update({self.placeholders['input_y']: torch.tensor(np.array(input_y)).to(self.device)})
        feed_dict.update({self.placeholders['mask_y']: torch.tensor(np.array(mask_y)).to(self.device)})

        return feed_dict

    def sample(self, nodeids, timeids, sampler):
        '''
        Sample neighbors recursively. First-order, then second-order, ...
        '''
        samples = [nodeids]
        support_size = 1
        support_sizes = [support_size]
        first_or_second = ['second', 'first']
        for k in range(self.num_layers):
            t = self.num_layers - k - 1
            node = sampler([samples[k], self.samples_1_2[t], timeids, first_or_second[t], support_size])
            support_size *= self.samples_1_2[t]
            samples.append(np.reshape(node, [support_size * self.batch_size,]))
            support_sizes.append(support_size)
        return samples, support_sizes

    def next_val_minibatch_feed_dict(self, val_or_test='val'):
        '''
        ' Construct evaluation or test inputs.
        '''
        if val_or_test == 'val':
            start = self.batch_num_val * self.batch_size
            self.batch_num_val += 1
            data = self.valid_user_ids
        elif val_or_test == 'test':
            start = self.batch_num_test * self.batch_size
            self.batch_num_test += 1
            data = self.test_user_ids
        else:
            raise NotImplementedError
        
        current_batch_users = data[start: start + self.batch_size]
        # nodes = [int(sessionid.split('_')[0]) for sessionid in current_batch_sessions]
        # timeids = [int(sessionid.split('_')[1]) for sessionid in current_batch_sessions]
        # samples, support_sizes = self.sample(nodes, timeids, self.test_sampler)
        # print(current_batch_users)
        return self._batch_feed_dict(current_batch_users)

    def next_train_minibatch_feed_dict(self):
        '''
        Generate next training batch data.
        '''
        start = self.batch_num * self.batch_size
        self.batch_num += 1
        current_batch_users = self.train_user_ids[start: start + self.batch_size]
        # nodes = [int(userid) for userid in current_batch_users]
        # nodes = [int(sessionid.split('_')[0]) for sessionid in current_batch_sessions]
        # timeids = [int(sessionid.split('_')[1]) for sessionid in current_batch_sessions]
        # samples, support_sizes = self.sample(nodes, timeids, self.sampler)
        return self._batch_feed_dict(current_batch_users)
        # return self._batch_feed_dict([current_batch_sessions, samples, support_sizes])

    def construct_adj(self):
        '''
        Construct adj table used during training.
        '''
        adj = self.num_nodes*np.ones((self.num_nodes+1, self.max_degree), dtype=np.int32)
        deg = np.zeros((self.num_nodes,))
        missed = 0
        for nodeid in self.train_df.UserId.unique():
            neighbors = np.array([neighbor for neighbor in 
                                self.adj_info.loc[self.adj_info['Follower']==nodeid].Followee.unique()], dtype=np.int32)
            deg[nodeid] = len(neighbors)
            if len(neighbors) == 0:
                missed += 1
                continue
            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            adj[nodeid, :] = neighbors
        #print('Unexpected missing during constructing adj list: {}'.format(missed))
        return adj, deg

    def construct_test_adj(self):
        '''
        ' Construct adj table used during evaluation or testing.
        '''
        adj = self.num_nodes*np.ones((self.num_nodes+1, self.max_degree), dtype=np.int32)
        deg = np.zeros((self.num_nodes,))
        missed = 0
        data = self.all_data
        for nodeid in data.user_id.unique():
            neighbors = np.array([neighbor for neighbor in 
                                self.adj_info.loc[self.adj_info['Follower']==nodeid].Followee.unique()], dtype=np.int32)
            deg[nodeid] = len(neighbors)
            if len(neighbors) == 0:
                missed += 1
                continue
            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            adj[nodeid, :] = neighbors
        #print('Unexpected missing during constructing adj list: {}'.format(missed))
        return adj, deg

    def end(self):
        '''
        Indicate whether we finish a pass over all training samples.
        '''
        return self.batch_num * self.batch_size > len(self.train_user_ids) - self.batch_size

    def end_val(self, val_or_test='test'):
        '''
        ' Indicate whether we finish a pass over all testing or evaluation samples.
        '''
        batch_num = self.batch_num_val if val_or_test == 'val' else self.batch_num_test
        data = self.valid_user_ids if val_or_test == 'val' else self.test_user_ids
        end = batch_num * self.batch_size > len(data) - self.batch_size
        if end:
            if val_or_test == 'val':
                self.batch_num_val = 0
            elif val_or_test == 'test':
                self.batch_num_test = 0
            else:
                raise NotImplementedError
        if end:
            self.batch_num_val = 0
        return end

    def train_batch_len(self):
        batch_len = (len(self.train_user_ids) - self.batch_size) / self.batch_size
        return int(batch_len)

    def shuffle(self):
        '''
        Shuffle training data.
        '''
        self.train_user_ids = np.random.permutation(self.train_user_ids)
        self.batch_num = 0
        self.batch_num_val = 0
        self.batch_num_test = 0
