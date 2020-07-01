import os
import numpy as np
import h5py
import faiss

import argparse

parser = argparse.ArgumentParser(description='Process input arguments.')

parser.add_argument('--src_dir', '-s', help='path to S3DIS numpy files')

parser.add_argument('--dst_dir', '-d', help='path to output h5py files')

parser.add_argument('--num_points', type=int, default=4096,
                    help='number of sampled points of each block')

parser.add_argument('--bs', type=float, default=1,
                    help='size of each block')

parser.add_argument('--stride', type=float, default=1,
                    help='stride of block')

parser.add_argument('--K', type=int, default=30,
                    help='number of nearest neighbors for each sampled point')

args = parser.parse_args()

opt = vars(args)
print('------------ Options -------------')
for k, v in sorted(opt.items()):
    print('%s: %s' % (str(k), str(v)))
print('-------------- End ----------------')

SRC_DIR = args.src_dir
DST_DIR = args.dst_dir
NUM_POINT = args.num_points
BLOCK_SIZE = args.bs
STRIDE = args.stride
K_ = args.K


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




# -----------------------------------------------------------------------------
# PREPARE BLOCK DATA FOR DEEPNETS TRAINING/TESTING
# -----------------------------------------------------------------------------

def sample_data(data, num_sample):
    """ data is in N x ...
        we want to keep num_samplexC of them.
        if N > num_sample, we will randomly keep num_sample of them.
        if N < num_sample, we will randomly duplicate samples.
    """
    N = data.shape[0]
    if (N == num_sample):
        return data, range(N)
    elif (N > num_sample):
        sample = np.random.choice(N, num_sample)
        return data[sample, ...], sample
    else:
        sample = np.random.choice(N, num_sample-N)
        dup_data = data[sample, ...]
        return np.concatenate([data, dup_data], 0), list(range(N))+list(sample)


def sample_data_label(data, label, num_sample):
    new_data, sample_indices = sample_data(data, num_sample)
    new_label = label[sample_indices]
    return new_data, new_label


def room2blocks(data, label, num_point, block_size=1.0, stride=1.0, k=10,
                random_sample=False, sample_num=None, sample_aug=1):
    """ Prepare block training data.
    Args:
        data: N x 6 numpy array, 012 are XYZ in meters, 345 are RGB in [0,1]
            assumes the data is shifted (min point is origin) and aligned
            (aligned with XYZ axis)
        label: N size uint8 numpy array from 0-12
        num_point: int, how many points to sample in each block
        block_size: float, physical size of the block in meters
        stride: float, stride for block sweeping
        random_sample: bool, if True, we will randomly sample blocks in the room
        sample_num: int, if random sample, how many blocks to sample
            [default: room area]
        sample_aug: if random sample, how much aug
    Returns:
        block_datas: K x num_point x 6 np array of XYZRGB, RGB is in [0,1]
        block_labels: K x num_point x 1 np array of uint8 labels
        
    TODO: for this version, blocking is in fixed, non-overlapping pattern.
    """
    assert(stride<=block_size)

    limit = np.amax(data, 0)[0:3]
     
    # Get the corner location for our sampling blocks    
    xbeg_list = []
    ybeg_list = []
    if not random_sample:
        num_block_x = int(np.ceil((limit[0] - block_size) / stride)) + 1
        num_block_y = int(np.ceil((limit[1] - block_size) / stride)) + 1
        for i in range(num_block_x):
            for j in range(num_block_y):
                xbeg_list.append(i*stride)
                ybeg_list.append(j*stride)
    else:
        num_block_x = int(np.ceil(limit[0] / block_size))
        num_block_y = int(np.ceil(limit[1] / block_size))
        if sample_num is None:
            sample_num = num_block_x * num_block_y * sample_aug
        for _ in range(sample_num):
            xbeg = np.random.uniform(-block_size, limit[0]) 
            ybeg = np.random.uniform(-block_size, limit[1]) 
            xbeg_list.append(xbeg)
            ybeg_list.append(ybeg)

    # Collect blocks
    block_data_list = []
    block_label_list = []
    block_knn_list = []
    idx = 0
    for idx in range(len(xbeg_list)): 
        xbeg = xbeg_list[idx]
        ybeg = ybeg_list[idx]
        xcond = (data[:,0]<=xbeg+block_size) & (data[:,0]>=xbeg)
        ycond = (data[:,1]<=ybeg+block_size) & (data[:,1]>=ybeg)
        cond = xcond & ycond
        if np.sum(cond) < 100: # discard block if there are less than 100 pts.
            continue
       
        block_data = data[cond, :]
        block_label = label[cond]
       
        # randomly subsample data
        block_data_sampled, block_label_sampled = \
            sample_data_label(block_data, block_label, num_point)

        # compute KNN
        knn_builder = KNNBuilder(k)
        _, block_knn_sampled = knn_builder.self_build_search(block_data_sampled[:,:3])

        block_data_list.append(np.expand_dims(block_data_sampled, 0))
        block_label_list.append(np.expand_dims(block_label_sampled, 0))
        block_knn_list.append(np.expand_dims(block_knn_sampled, 0))
            
    return np.concatenate(block_data_list, 0), \
           np.concatenate(block_label_list, 0), \
           np.concatenate(block_knn_list, 0)



def room2blocks_plus_normalized(data_label, num_point, block_size, stride, k,
                                random_sample, sample_num, sample_aug):
    """ room2block, with input filename and RGB preprocessing.
        for each block centralize XYZ, add normalized XYZ as 678 channels
    """
    data = data_label[:,0:6]
    data[:,3:6] /= 255.0
    label = data_label[:,-1].astype(np.uint8)
    max_room_x = max(data[:,0])
    max_room_y = max(data[:,1])
    max_room_z = max(data[:,2])
    
    data_batch, label_batch, knn_batch = room2blocks(data, label, num_point, block_size, stride, k,
                                                    random_sample, sample_num, sample_aug)
    new_data_batch = np.zeros((data_batch.shape[0], num_point, 9))
    for b in range(data_batch.shape[0]):
        new_data_batch[b, :, 6] = data_batch[b, :, 0]/max_room_x
        new_data_batch[b, :, 7] = data_batch[b, :, 1]/max_room_y
        new_data_batch[b, :, 8] = data_batch[b, :, 2]/max_room_z
        minx = min(data_batch[b, :, 0])
        miny = min(data_batch[b, :, 1])
        data_batch[b, :, 0] -= (minx+block_size/2)
        data_batch[b, :, 1] -= (miny+block_size/2)
    new_data_batch[:, :, 0:6] = data_batch
    return new_data_batch, label_batch, knn_batch


def room2blocks_wrapper_normalized(data_label_filename, num_point, block_size=1.0, stride=1.0, k=10,
                                   random_sample=False, sample_num=None, sample_aug=1):
    if data_label_filename[-3:] == 'txt':
        data_label = np.loadtxt(data_label_filename)
    elif data_label_filename[-3:] == 'npy':
        data_label = np.load(data_label_filename)
    else:
        print('Unknown file type! exiting.')
        exit()
    return room2blocks_plus_normalized(data_label, num_point, block_size, stride, k,
                                       random_sample, sample_num, sample_aug)



# --------------------------------------
# ----- BATCH WRITE TO HDF5 -----
# --------------------------------------
H5_BATCH_SIZE = 1000
data_dim = [NUM_POINT, 9]
label_dim = [NUM_POINT]
knn_dim = [NUM_POINT, K_]
data_dtype = 'float32'
label_dtype = 'uint8'
knn_dtype = 'int16'
batch_data_dim = [H5_BATCH_SIZE] + data_dim
batch_label_dim = [H5_BATCH_SIZE] + label_dim
batch_knn_dim = [H5_BATCH_SIZE] + knn_dim
h5_batch_data = np.zeros(batch_data_dim, dtype = np.float32)
h5_batch_label = np.zeros(batch_label_dim, dtype = np.uint8)
h5_batch_knn = np.zeros(batch_knn_dim, dtype = np.int16)
buffer_size = 0  # state: record how many samples are currently in buffer
h5_index = 0 # state: the next h5 file to save


# Write numpy array data and label to h5_filename
def save_h5(h5_filename, data, label, knn, data_dtype='float32', label_dtype='uint8', knn_type = 'int16'):
    h5_fout = h5py.File(h5_filename)
    h5_fout.create_dataset(
            'data', data=data,
            compression='gzip', compression_opts=4,
            dtype=data_dtype)
    h5_fout.create_dataset(
            'label', data=label,
            compression='gzip', compression_opts=1,
            dtype=label_dtype)
    h5_fout.create_dataset(
            'knn', data=knn,
            compression='gzip', compression_opts=4,
            dtype=knn_dtype)
    h5_fout.close()


def insert_batch(data, label, knn, last_batch=False):
    global h5_batch_data, h5_batch_label, h5_batch_knn
    global buffer_size, h5_index
    data_size = data.shape[0]
    # If there is enough space, just insert
    if buffer_size + data_size <= h5_batch_data.shape[0]:
        h5_batch_data[buffer_size:buffer_size+data_size, ...] = data
        h5_batch_label[buffer_size:buffer_size+data_size] = label
        h5_batch_knn[buffer_size:buffer_size+data_size, ...] = knn
        buffer_size += data_size
    else: # not enough space
        capacity = h5_batch_data.shape[0] - buffer_size
        assert(capacity>=0)
        if capacity > 0:
           h5_batch_data[buffer_size:buffer_size+capacity, ...] = data[0:capacity, ...]
           h5_batch_label[buffer_size:buffer_size+capacity, ...] = label[0:capacity, ...]
           h5_batch_knn[buffer_size:buffer_size+capacity, ...] = knn[0:capacity, ...]
        # Save batch data and label to h5 file, reset buffer_size
        h5_filename =  output_filename_prefix + '_' + str(h5_index) + '.h5'
        save_h5(h5_filename, h5_batch_data, h5_batch_label, h5_batch_knn, data_dtype, label_dtype, knn_dtype)
        print('Stored {0} with size {1}'.format(h5_filename, h5_batch_data.shape[0]))
        h5_index += 1
        buffer_size = 0
        # recursive call
        insert_batch(data[capacity:, ...], label[capacity:, ...], knn[capacity:, ...], last_batch)
    if last_batch and buffer_size > 0:
        h5_filename =  output_filename_prefix + '_' + str(h5_index) + '.h5'
        save_h5(h5_filename, h5_batch_data[0:buffer_size, ...], h5_batch_label[0:buffer_size, ...], h5_batch_knn[0:buffer_size, ...], data_dtype, label_dtype, knn_dtype)
        print('Stored {0} with size {1}'.format(h5_filename, buffer_size))
        h5_index += 1
        buffer_size = 0
    return



if __name__ == '__main__':
    DATA_DIR = os.path.join(DST_DIR, 's3dis_bs%.1f_s%.1f_K%d_%d_normalized' %(BLOCK_SIZE, STRIDE, K_, NUM_POINT))
    if not os.path.exists(DATA_DIR): os.mkdir(DATA_DIR)

    file_lists = [line.rstrip() for line in open(os.path.join(SRC_DIR, 'npy_file_list.txt'))]
    file_lists = [os.path.join(SRC_DIR, f) for f in file_lists]

    output_filename_prefix = os.path.join(DATA_DIR, 'ply_data_all')
    output_room_filelist = os.path.join(DATA_DIR, 'room_filelist.txt')
    fout_room = open(output_room_filelist, 'w')

    sample_cnt = 0

    for i, data_label_filename in enumerate(file_lists):
        print(data_label_filename)
        data, label, knn = room2blocks_wrapper_normalized(data_label_filename, NUM_POINT, block_size=BLOCK_SIZE,
                                                                   stride=STRIDE, k=K_,
                                                                   random_sample=False, sample_num=None)
        print('{0}, {1}, {2}'.format(data.shape, label.shape, knn.shape))
        for _ in range(data.shape[0]):
            fout_room.write(os.path.basename(data_label_filename)[0:-4] + '\n')

        sample_cnt += data.shape[0]
        insert_batch(data, label, knn, i == len(file_lists) - 1)

    fout_room.close()
    print("Total samples: {0}".format(sample_cnt))

    with open(os.path.join(DATA_DIR, 'all_h5_files.txt'), 'w') as f:
        for j in range(h5_index):
            f.write('ply_data_all_%d.h5\n' %j)