""" DataLoader function w.r.t. data preparation setups in PointCNN"""
import numpy as np

import torch
import torch.utils.data as DATA

from data_aug_util import *
from data_util import *


def get_indices(sample_num, pt_num):
    if pt_num > sample_num:
        choices = np.random.choice(pt_num, sample_num, replace=False)
    else:
        choices = np.concatenate((np.random.choice(pt_num, pt_num, replace=False),
                                  np.random.choice(pt_num, sample_num - pt_num, replace=True)))
    return choices


def make_dataset(mode, filelist):

    data, point_num, label = load_seg(filelist)
    if mode=='train':
        data, point_num, label = grouped_shuffle([data, point_num, label])
        
    print('mode-{0}:{1},{2}'.format(mode, data.shape, label.shape))

    return data, point_num, label


class MyDataset(DATA.Dataset):
    def __init__(self, mode, filelist, opt):
        super(MyDataset, self).__init__()
        self.opt = opt
        self.mode = mode

        self.data, self.point_num, self.label = make_dataset(mode, filelist)

        self.knn_builder = KNNBuilder(opt.K)

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, index):

        data_slice = self.data[index] #Nx6, (x,z,y,r,g,b) or  Nx6 (x,z,y)
        point_num_curr = self.point_num[index]
        label_slice = self.label[index]

        if self.opt.augment and self.mode != 'test':
            data_slice[:,0:3] = aug(data_slice[:,0:3])

        # Slicing the input feature dim..
        if self.opt.input_feat < self.opt.max_input_feat:
            data_slice = data_slice[:, :self.opt.input_feat]

        indices = get_indices(self.opt.num_point, point_num_curr)
        data_slice = data_slice[indices,...]
        label_slice = label_slice[indices,...]
        _, knn_slice = self.knn_builder.self_build_search(data_slice[:,:3])

        data = torch.from_numpy(data_slice.transpose().astype(np.float32))  # 9xN
        label = torch.from_numpy(label_slice.astype(np.int64))  # N
        knn = torch.from_numpy(knn_slice.astype(np.int64))

        return data, knn, label
