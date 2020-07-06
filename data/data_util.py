import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import numpy as np
import faiss
import h5py

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    knn = f['knn'][:]
    return (data, label, knn)


def loadDataFile(filename):
    return load_h5(filename)


def loadAllData(path, filelist):
    # Load ALL data
    data_batch_list = []
    label_batch_list = []
    knn_batch_list = []

    for filename in filelist:
        filepath = os.path.join(path, filename)
        data_batch, label_batch, knn_batch= loadDataFile(filepath)
        data_batch_list.append(data_batch)
        label_batch_list.append(label_batch)
        knn_batch_list.append(knn_batch)

    data_batches = np.concatenate(data_batch_list, 0)
    label_batches = np.concatenate(label_batch_list, 0)
    knn_batches = np.concatenate(knn_batch_list, 0)

    return data_batches, label_batches, knn_batches


def shuffle_data(data, labels, knn):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
          knn: B,N,K numpy array
        Return:
          shuffled data, label, knn and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], knn[idx,...], idx


def grouped_shuffle(inputs):
    for idx in range(len(inputs) - 1):
        assert (len(inputs[idx]) == len(inputs[idx + 1]))

    shuffle_indices = np.arange(inputs[0].shape[0])
    np.random.shuffle(shuffle_indices)
    outputs = []
    for idx in range(len(inputs)):
        outputs.append(inputs[idx][shuffle_indices, ...])
    return outputs


def is_h5_list(filelist):
    return all([line.strip()[-3:] == '.h5' for line in open(filelist)])


def load_seg_list(filelist):
    folder = os.path.dirname(filelist)
    return [os.path.join(folder, line.strip()) for line in open(filelist)]


def load_seg(filelist):
    points = []
    point_nums = []
    labels_seg = []

    folder = os.path.dirname(filelist)
    for line in open(filelist):
        data = h5py.File(os.path.join(folder, line.strip()))
        points.append(data['data'][...])
        point_nums.append(data['data_num'][...])
        labels_seg.append(data['label_seg'][...])

    return (np.concatenate(points, axis=0),
            np.concatenate(point_nums, axis=0),
            np.concatenate(labels_seg, axis=0))



# class KNNBuilder:
# -----------------------------------------------------------------------------
# Faiss KNN to find K nearest neighbors
# -----------------------------------------------------------------------------
class KNNBuilder:
    def __init__(self, k):
        self.k = k
        self.dimension = 3

    def build_nn_index(self, database):
        '''
        :param database: numpy array of Nx3
        :return: Faiss index, in CPU
        '''
        index = faiss.IndexFlatL2(self.dimension)  # dimension is 3
        index.add(database)
        return index

    def search_nn(self, index, query, k):
        '''
        :param index: Faiss index
        :param query: numpy array of Nx3
        :return: D: numpy array of Nxk
                 I: numpy array of Nxk
        '''
        D, I = index.search(query, k)
        return D, I

    def self_build_search(self, x):
        '''
        :param x: numpy array of Nx3
        :return: D: numpy array of Nxk
                 I: numpy array of Nxk
        '''
        x = np.ascontiguousarray(x, dtype=np.float32)
        index = self.build_nn_index(x)
        D, I = self.search_nn(index, x, self.k)
        return D, I


class KNNBuilder_batch:
    def __init__(self, k, dim):
        self.k = k
        self.dimension = dim

    def build_nn_index(self, database):
        '''
        :param database: numpy array of Nx3
        :return: Faiss index, in CPU
        '''
        index = faiss.IndexFlatL2(self.dimension)  # dimension is 3
        index.add(database)
        return index

    def search_nn(self, index, query, k):
        '''
        :param index: Faiss index
        :param query: numpy array of Nx3
        :return: D: numpy array of Nxk
                 I: numpy array of Nxk
        '''
        D, I = index.search(query, k)
        return D, I

    def self_build_search(self, x):
        '''
        :param x: numpy array of BxNxC
        :return: I: numpy array of BxNxk
        '''
        I = []
        for b in range(x.shape[0]):
            x_slice = x[b, :, :]
            x_slice = np.ascontiguousarray(x_slice, dtype=np.float32)
            index = self.build_nn_index(x_slice)
            _, I_slice = self.search_nn(index, x_slice, self.k)
            I.append(I_slice)
        I = np.stack(I)

        return I