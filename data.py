# Template code is provided at the
# https://github.com/jbnu-dslab/DGRec-pytorch/blob/master/src/data.py

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path

    def load_adj(self, data_path):
        df_adj = pd.read_csv(data_path + '/adj.tsv', sep='\t', dtype={0: np.int32, 1: np.int32})
        return df_adj

    def load_latest_session(self, data_path):
        ret = []
        for line in open(data_path + '/latest_sessions.txt'):
            chunks = line.strip().split(',')
            ret.append(chunks)
        return ret

    def load_map(self, data_path, name='user'):
        if name == 'user':
            file_path = data_path + '/user_id_map.tsv'
        elif name == 'item':
            file_path = data_path + '/item_id_map.tsv'
        else:
            raise NotImplementedError
        id_map = {}
        for line in open(file_path):
            k, v = line.strip().split('\t')
            id_map[k] = str(v)
        return id_map

    def load_data(self):
        # adj = self.load_adj(self.data_path)
        # latest_sessions = self.load_latest_session(self.data_path)
        # user_id_map = self.load_map(self.data_path, 'user')
        # item_id_map = self.load_map(self.data_path, 'item')
        train = pd.read_csv(self.data_path + '/train_tr.csv', sep=',', dtype={3: np.float32, 4: np.float32, 5: np.float32, 6: np.float32, 7: np.float32,
                                                                         8: np.float32, 9: np.float32, 10: np.int32, 12: np.int32, 13: np.int32, 14: np.int32})

        # train = pd.read_csv(self.data_path + '/datae.csv', sep=',',
        #                     dtype={0: np.float32, 1: np.float32, 2: np.float32, 3: np.float32,
        #                            4: np.int32, 5: np.int32})
        valid = pd.read_csv(self.data_path + '/val_tr.csv', sep=',', dtype={3: np.float32, 4: np.float32, 5: np.float32, 6: np.float32, 7: np.float32,
                                                                         8: np.float32, 9: np.float32, 10: np.int32, 12: np.int32, 13: np.int32, 14: np.int32})
        # valid = pd.read_csv(self.data_path + '/datae.csv', sep=',',
        #                     dtype={0: np.float32, 1: np.float32, 2: np.float32, 3: np.float32,
        #                            4: np.int32, 5: np.int32})
        test = pd.read_csv(self.data_path + '/test_tr.csv', sep=',', dtype={3: np.float32, 4: np.float32, 5: np.float32, 6: np.float32, 7: np.float32,
                                                                         8: np.float32, 9: np.float32, 10: np.int32, 12: np.int32, 13: np.int32, 14: np.int32})
        # test = pd.read_csv(self.data_path + '/datae.csv', sep=',',
        #                    dtype={0: np.float32, 1: np.float32, 2: np.float32, 3: np.float32,
        #                           4: np.int32, 5: np.int32})
        # test = pd.read_csv(self.data_path + '/datae.csv', sep=',',
        #                    dtype={0: np.float32, 1: np.float32, 3: np.float32, 4: np.float32,5: np.int32, 6: np.int32})
        return [train, valid, test]
