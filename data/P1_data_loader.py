""" DataLoader function w.r.t. data preparation setups in PointNet and PointNet++"""
import os
import numpy as np
import pickle

import torch
import torch.utils.data as DATA

from data_aug_util import *
from data_util import *


def make_dataset_s3dis(path, mode, area_list, k):

    h5_filelist = [line.rstrip() for line in open(os.path.join(path, 'all_h5_files.txt'))]
    room_filelist = [line.rstrip() for line in open(os.path.join(path, 'room_filelist.txt'))]

    # Load ALL data
    data_batch_list = []
    label_batch_list = []
    knn_batch_list = []
    for h5_filename in h5_filelist:
        filepath = os.path.join(path, h5_filename)
        data_batch, label_batch, knn_batch= loadDataFile(filepath)
        data_batch_list.append(data_batch)
        label_batch_list.append(label_batch)
        # knn_batch_list.append(knn_batch)
        if 0 < k < 30:
            knn_batch_list.append(knn_batch[:,:,:k])
        elif k == 30:
            knn_batch_list.append(knn_batch)
        else:
            raise Exception('Please enter valid knn_list [maximum value in knn_list should be in (1,30)]..')
            exit(1)
    data_batches = np.concatenate(data_batch_list, 0)
    label_batches = np.concatenate(label_batch_list, 0)
    knn_batches = np.concatenate(knn_batch_list, 0)

    idx = []
    for i, room_name in enumerate(room_filelist):
        if room_name[5] in area_list:
            idx.append(i)

    data = data_batches[idx, ...]
    label = label_batches[idx]
    knn = knn_batches[idx, ...]

    if mode == 'train':
        data, label, knn, _ = shuffle_data(data, label, knn)

    print('mode-{0}:{1},{2},{3}'.format(mode, data.shape, label.shape, knn.shape))

    return data, label, knn


class S3DISDataset(DATA.Dataset):
    def __init__(self, mode, opt):
        super(S3DISDataset, self).__init__()
        self.opt = opt
        self.mode = mode
        test_area = opt.test_area.split(',')
        if mode == 'train':
            data_path = os.path.join(opt.data_dir, 's3dis_bs%.1f_s%.1f_K30_%d_normalized' % (
            opt.block_size, opt.stride, opt.num_point))
            # data_path = os.path.join(opt.data_dir, 'indoor3d_sem_seg_hdf5_data_K30')
            area_list = list(set(['1', '2', '3', '4', '5', '6']) - set(test_area))
        elif mode == 'test':
            data_path = os.path.join(opt.data_dir, 's3dis_bs%.1f_s%.1f_K30_%d_normalized' % (
            opt.block_size, opt.block_size, opt.num_point))
            area_list = test_area
        else:
            raise Exception('Mode should be train/test.')

        self.data, self.label, self.knn = make_dataset_s3dis(data_path, mode, area_list, opt.K)

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, index):

        data_slice = self.data[index]  # Nx9

        if self.opt.augment and self.mode != 'test':
            data_slice[:, 0:3] = aug(data_slice[:, 0:3])

        # Slicing the input feature dim..
        if self.opt.input_feat < 9:
            data_slice = data_slice[:, :self.opt.input_feat]
        label_slice = self.label[index]
        knn_slice = self.knn[index]

        data = torch.from_numpy(data_slice.transpose().astype(np.float32))  # 9xN
        label = torch.from_numpy(label_slice.astype(np.int32))  # N
        knn = torch.from_numpy(knn_slice.astype(np.int32))

        return data, knn, label



# Modified from https://github.com/charlesq34/pointnet2/blob/master/scannet/scannet_dataset.py
class ScannetDataset(DATA.Dataset):
    def __init__(self, root, npoints=8192, k=30, split='train'):
        super(ScannetDataset, self).__init__()

        self.npoints = npoints
        self.root = root
        self.split = split
        self.data_filename = os.path.join(self.root, 'scannet_%s.pickle' % (split))
        with open(self.data_filename, 'rb') as fp:
            self.scene_points_list = pickle.load(fp, encoding='latin1')
            self.semantic_labels_list = pickle.load(fp, encoding='latin1')
        if split == 'train':
            labelweights = np.zeros(21)
            for seg in self.semantic_labels_list:
                tmp, _ = np.histogram(seg, range(22))
                labelweights += tmp
                labelweights = labelweights.astype(np.float32)
                labelweights = labelweights / np.sum(labelweights)
                self.labelweights = 1 / np.log(1.2 + labelweights)
        elif split == 'test':
            self.labelweights = np.ones(21)

        self.knn_builder = KNNBuilder(k)

    def __getitem__(self, index):
        point_set = self.scene_points_list[index]
        semantic_seg = self.semantic_labels_list[index].astype(np.int32)
        coordmax = np.max(point_set, axis=0)
        coordmin = np.min(point_set, axis=0)
        # smpmin = np.maximum(coordmax-[1.5,1.5,3.0], coordmin)
        # smpmin[2] = coordmin[2]
        # smpsz = np.minimum(coordmax-smpmin,[1.5,1.5,3.0])
        # smpsz[2] = coordmax[2]-coordmin[2]
        isvalid = False
        for i in range(10):
            curcenter = point_set[np.random.choice(len(semantic_seg), 1)[0], :]
            curmin = curcenter - [0.75, 0.75, 1.5]
            curmax = curcenter + [0.75, 0.75, 1.5]
            curmin[2] = coordmin[2]
            curmax[2] = coordmax[2]
            curchoice = np.sum((point_set >= (curmin - 0.2)) * (point_set <= (curmax + 0.2)), axis=1) == 3
            cur_point_set = point_set[curchoice, :]
            cur_semantic_seg = semantic_seg[curchoice]
            if len(cur_semantic_seg) == 0:
                continue
            mask = np.sum((cur_point_set >= (curmin - 0.01)) * (cur_point_set <= (curmax + 0.01)), axis=1) == 3
            vidx = np.ceil((cur_point_set[mask, :] - curmin) / (curmax - curmin) * [31.0, 31.0, 62.0])
            vidx = np.unique(vidx[:, 0] * 31.0 * 62.0 + vidx[:, 1] * 62.0 + vidx[:, 2])
            isvalid = np.sum(cur_semantic_seg > 0) / len(cur_semantic_seg) >= 0.7 and len(
                vidx) / 31.0 / 31.0 / 62.0 >= 0.02
            if isvalid:
                break
        choice = np.random.choice(len(cur_semantic_seg), self.npoints, replace=True)
        point_set = cur_point_set[choice, :]
        semantic_seg = cur_semantic_seg[choice]
        mask = mask[choice]
        sample_weight = self.labelweights[semantic_seg]
        sample_weight *= mask

        _, knn_slice = self.knn_builder.self_build_search(point_set[:, :3])

        # ToDO: normalized xyz
        data = torch.from_numpy(point_set.transpose().astype(np.float32))  # 3xN
        label = torch.from_numpy(semantic_seg.astype(np.int64))  # N
        knn = torch.from_numpy(knn_slice.astype(np.int64))
        weight = torch.from_numpy(sample_weight.astype(np.float32))

        return data, knn, label, weight

    def __len__(self):
        return len(self.scene_points_list)


class ScannetDatasetWholeScene(DATA.Dataset):
    def __init__(self, root, npoints=8192, k=30, split='test'):
        super(ScannetDatasetWholeScene, self).__init__()

        self.npoints = npoints
        self.root = root
        self.split = split
        self.data_filename = os.path.join(self.root, 'scannet_%s.pickle' % (split))
        with open(self.data_filename, 'rb') as fp:
            self.scene_points_list = pickle.load(fp, encoding='latin1')
            self.semantic_labels_list = pickle.load(fp, encoding='latin1')
        if split == 'train':
            labelweights = np.zeros(21)
            for seg in self.semantic_labels_list:
                tmp, _ = np.histogram(seg, range(22))
                labelweights += tmp
                labelweights = labelweights.astype(np.float32)
                labelweights = labelweights / np.sum(labelweights)
                self.labelweights = 1 / np.log(1.2 + labelweights)
        elif split == 'test':
            self.labelweights = np.ones(21)

        self.knn_builder = KNNBuilder(k)

    def __getitem__(self, index):
        point_set_ini = self.scene_points_list[index]
        semantic_seg_ini = self.semantic_labels_list[index].astype(np.int32)
        coordmax = np.max(point_set_ini, axis=0)
        coordmin = np.min(point_set_ini, axis=0)
        nsubvolume_x = np.ceil((coordmax[0] - coordmin[0]) / 1.5).astype(np.int32)
        nsubvolume_y = np.ceil((coordmax[1] - coordmin[1]) / 1.5).astype(np.int32)
        point_sets = list()
        semantic_segs = list()
        knn_list = list()
        sample_weights = list()
        isvalid = False
        for i in range(nsubvolume_x):
            for j in range(nsubvolume_y):
                curmin = coordmin + [i * 1.5, j * 1.5, 0]
                curmax = coordmin + [(i + 1) * 1.5, (j + 1) * 1.5, coordmax[2] - coordmin[2]]
                curchoice = np.sum((point_set_ini >= (curmin - 0.2)) * (point_set_ini <= (curmax + 0.2)), axis=1) == 3
                cur_point_set = point_set_ini[curchoice, :]
                cur_semantic_seg = semantic_seg_ini[curchoice]
                if len(cur_semantic_seg) == 0:
                    continue
                mask = np.sum((cur_point_set >= (curmin - 0.001)) * (cur_point_set <= (curmax + 0.001)), axis=1) == 3
                choice = np.random.choice(len(cur_semantic_seg), self.npoints, replace=True)
                point_set = cur_point_set[choice, :]  # Nx3
                semantic_seg = cur_semantic_seg[choice]  # N
                mask = mask[choice]
                if sum(mask) / float(len(mask)) < 0.01:
                    continue
                sample_weight = self.labelweights[semantic_seg]
                sample_weight *= mask  # N

                _, knn_slice = self.knn_builder.self_build_search(point_set[:, :3])

                point_sets.append(np.expand_dims(point_set.transpose(), 0))  # 1x3xN
                knn_list.append(np.expand_dims(knn_slice, 0))
                semantic_segs.append(np.expand_dims(semantic_seg, 0))  # 1xN
                sample_weights.append(np.expand_dims(sample_weight, 0))  # 1xN
        point_sets = np.concatenate(tuple(point_sets), axis=0)
        knn_list = np.concatenate(tuple(knn_list), axis=0)
        semantic_segs = np.concatenate(tuple(semantic_segs), axis=0)
        sample_weights = np.concatenate(tuple(sample_weights), axis=0)

        data = torch.from_numpy(point_sets.astype(np.float32))  # Bx3xN
        knn = torch.from_numpy(knn_list.astype(np.int64))  # BxNxK
        label = torch.from_numpy(semantic_segs.astype(np.int64))  # BxN
        weight = torch.from_numpy(sample_weights.astype(np.float32))

        return data, knn, label, weight

    def __len__(self):
        return len(self.scene_points_list)