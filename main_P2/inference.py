from datetime import datetime
import os
import sys
import argparse
import torch
import math
import h5py

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'data'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from model import import_class
from network import *
from data_util import KNNBuilder_batch


parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model_name', type=str, default='PS2Net', help='name of model')
parser.add_argument('--data_dir', default='../datasets/S3DIS/P2/', help='path to point clouds')

parser.add_argument('--dataset_name', type=str, default='S3DIS', help='the dataset to use [options: S3DIS, ScanNet]')
parser.add_argument('--classes', type=int, default=13, help='S3DIS or ScanNet')
parser.add_argument('--log_dir', default='../log_s3dis/P2_**', help='Log dir [default: log]')
parser.add_argument('--checkpoint', type=str, default=None, help='name of pre-trained network parameters')

parser.add_argument('--max_point_num', type=int, default=8192, help='Maximum point number [default: 8192]')
parser.add_argument('--sample_num', type=int, default=2048, help='Sampled point number [default: 2048]')
parser.add_argument('--repeat_num', type=int, default=1, help='Repeat number')

parser.add_argument('--max_input_feat',  type=int, default=6, help='The maximum dimension of raw input features [Option: 3 or 6]')
parser.add_argument('--input_feat', type=int, default=6, help='The dimension of raw input features [Option: 3 or 6]')

parser.add_argument('--K', type=int, default=20, help='the maximum value of KNN')
parser.add_argument('--knn_list', default='[20,20,20,20]', help='the number of nearest neighbors to extract local feature in each encoder')
parser.add_argument('--local_feat_integrate', type=str, default='concatenate', help='the way to integrate local contextual features \
                    which is generated by maxpooling over neighborhood. option:[concatenate, add]')
parser.add_argument('--num_clusters', default='[16,16,16,16]', help='NetVLAD hyperparameter, the number of clusters for each encoder')
parser.add_argument('--output_dim', default='[128,128,128,128]', help='size of the output space for dimension reduction after NetVLAD in each encoder')

parser.add_argument('--encoder_paras', default='[[64,64,128],[64,64,128],[64,64,128],[64,64,128]]',
                    help='The architecture of stacked encoder network.')
parser.add_argument('--encoder_dropout', default='[[0,0,0],[0,0,0],[0,0,0],[0,0,0]]',
                    help='The dropout_ratio in the stacked encoder networks.')
parser.add_argument('--decoder_paras', default='[512, 256, 128]', 
                    help='The architecture of decoder network. [MLP, MLP]')
parser.add_argument('--decoder_dropout', default='[0, 0, 0]', 
                    help='The dropout_ratio in the decoder network (the last dropout_ratio is set to 0)')

parser.add_argument('--activation', type=str, default='relu', help='activation function: relu, prelu, elu, leakyrelu')
parser.add_argument('--normalization', type=str, default='batch', help='normalization function: batch or instance')
parser.add_argument('--norm_momentum', type=float, default=0.1, help='normalization momentum, typically 0.1.')

parser.add_argument('--augment', type=bool, default=False, help='Data augment or not [default: False]')
parser.add_argument('--test_area', type=int, default=5, help='Which areas to use for test, option:[1-6] [default: 5]')
opt = parser.parse_args()

opt.batch_size = opt.repeat_num * math.ceil(opt.max_point_num / opt.sample_num)
opt.nThreads = opt.batch_size

args = vars(opt)
print('------------ Options -------------')
for k, v in sorted(args.items()):
    print('%s: %s' % (str(k), str(v)))
print('-------------- End ----------------')


LOG_DIR = opt.log_dir
if not os.path.exists(LOG_DIR): 
    raise Exception('Please assign the log_dir used for storing training data!')
    exit()


if __name__ == '__main__':

    #Prepare input testing data
    if opt.dataset_name == 'S3DIS':
        test_filepath = os.path.join(opt.data_dir, 'val_files_Area_%d.txt' %opt.test_area)
    elif opt.dataset_name == 'ScanNet':
        test_filepath = os.path.join(opt.data_dir, 'test_files.txt')
    else:
        raise Exception('Please enter correct dataset name. Options: S3DIS or ScanNet')

    test_file_list = [os.path.join(os.path.dirname(test_filepath), line.strip()) for line in open(test_filepath)]

    network = import_class('network', opt.model_name)(opt)
    if opt.checkpoint is not None: 
        checkpoint = torch.load(os.path.join(LOG_DIR, 'checkpoints', opt.checkpoint))
        network.load_state_dict(checkpoint['state_dict'])
    else:
        raise Exception('Please enter correct checkpoint path to load...')

    network = network.cuda()
    softmax = torch.nn.Softmax(dim=1)

    knn_builder = KNNBuilder_batch(opt.K, 3)

    print('Evaluating begins..')
    with torch.no_grad():
        for filename in test_file_list:
            print('{}-Processing {}...'.format(datetime.now(),filename))
            h5f = h5py.File(filename)
            data = h5f['data'][...].astype(np.float32)
            if opt.input_feat < opt.max_input_feat:
                data = data[:, :, :opt.input_feat]

            data_num = h5f['data_num'][...]
            # label = h5f['label_seg'][...]
            batch_num = data.shape[0]

            labels_pred = np.full((batch_num, opt.max_point_num), -1, dtype=np.int32)
            confidences_pred = np.zeros((batch_num, opt.max_point_num), dtype=np.float32)

            print('{}-{:d} testing batches.'.format(datetime.now(), batch_num))
            for batch_idx in range(batch_num):
                if batch_idx % 10 == 0:
                    print('{}-Processing {} of {} batches.'.format(datetime.now(), batch_idx, batch_num))
                points_batch = data[[batch_idx] * opt.batch_size, ...]
                point_num = data_num[batch_idx]

                tile_num = math.ceil((opt.sample_num * opt.batch_size) / point_num)
                indices_shuffle = np.tile(np.arange(point_num), tile_num)[0:opt.sample_num * opt.batch_size]
                np.random.shuffle(indices_shuffle)
                indices_batch_shuffle = np.reshape(indices_shuffle, (opt.batch_size, opt.sample_num, 1))

                input_data = torch.from_numpy(points_batch.astype(np.float32)) #(B,N,C), N is the point_num
                sampled_indices = torch.from_numpy(indices_batch_shuffle.astype(np.int64)) #(B,n,1), n is the sample_num
                sampled_indices = sampled_indices.expand(-1,-1,input_data.shape[2]).contiguous()
                input_data = torch.gather(input_data, dim=1, index=sampled_indices) #(B,n,C)

                knn_idx = knn_builder.self_build_search(input_data[:,:,:3].detach())
                knn_idx = torch.from_numpy(knn_idx.astype(np.int64)).cuda()
                input_data = input_data.transpose(1,2).cuda() #(B,C,n)

                network.eval()
                score = network(input_data, knn_idx)
                preds = softmax(score) #(B,class,n)
                preds = preds.transpose(1,2) #(B,n, class)

                preds_2d = np.reshape(preds.cpu().detach().numpy(),(opt.sample_num*opt.batch_size, -1))

                predictions = [(-1, 0.0)] * point_num
                for idx in range(opt.sample_num * opt.batch_size):
                    point_idx = indices_shuffle[idx]
                    probs = preds_2d[idx, :]
                    confidence = np.amax(probs)
                    label = np.argmax(probs)
                    if confidence > predictions[point_idx][1]:
                        predictions[point_idx] = [label, confidence]
                labels_pred[batch_idx, 0:point_num] = np.array([label for label, _ in predictions])
                confidences_pred[batch_idx, 0:point_num] = np.array([confidence for _, confidence in predictions])

            filename_pred = filename[:-3] + '_pred.h5'
            print('{}-Saving {}...'.format(datetime.now(), filename_pred))
            file = h5py.File(filename_pred, 'w')
            file.create_dataset('data_num', data=data_num)
            file.create_dataset('label_seg', data=labels_pred)
            file.create_dataset('confidence', data=confidences_pred)
            has_indices = 'indices_split_to_full' in h5f
            if has_indices:
                file.create_dataset('indices_split_to_full', data=h5f['indices_split_to_full'][...])
            file.close()

        print('{}-Done!'.format(datetime.now()))
