import os
import math
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'data'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

import argparse
import torch

from model import import_class, compute_iou, compute_iou_scannet
from network import *


parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model_name', type=str, default='PS2Net', help='name of model')
parser.add_argument('--dataset_name', type=str, default='ScanNet', help='the dataset to use [options: S3DIS, ScanNet]')

parser.add_argument('--data_dir', default='../datasets/ScanNet/P3/', help='path to point clouds')
parser.add_argument('--block_size', type=float, default=1.0, help='the size of blocks to split rooms')
parser.add_argument('--stride', type=float, default=1.0, help='the stride to slide along xy plane to split rooms')
parser.add_argument('--classes', type=int, default=21, help='S3DIS or ScanNet')
parser.add_argument('--log_dir', default='../log_scannet/P3_**', help='Log dir [default: log]')
parser.add_argument('--checkpoint', type=str, default=None, help='name of pre-trained network parameters')

parser.add_argument('--nThreads', default=1, type=int, help='# threads for loading data')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training')
parser.add_argument('--max_volumes', type=int, default=18, help='Maximum processing volumrs during testing')

parser.add_argument('--num_point', type=int, default=4096, help='Point number [default: 4096]')
parser.add_argument('--input_feat', type=int, default=3, help='The dimension of raw input features [Option: 3 or 9]')
parser.add_argument('--K', type=int, default=20, help='the maximum value of KNN')
parser.add_argument('--knn_list', default='[20,20,20,20]', help='the number of nearest neighbors to extract local feature in each encoder')
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
parser.add_argument('--lossweight_path', type=str, default=None, help='the path to the label weights for weighted cross-entropy loss')

parser.add_argument('--augment', type=bool, default=False, help='Data augment or not [default: False]')
parser.add_argument('--test_area', type=str, default='5', help='Which areas to use for test, option:[1-6] [default: 5]')
opt = parser.parse_args()

args = vars(opt)
print('------------ Options -------------')
for k, v in sorted(args.items()):
    print('%s: %s' % (str(k), str(v)))
print('-------------- End ----------------')


LOG_DIR = opt.log_dir
if not os.path.exists(LOG_DIR): 
    print('Please assign the log_dir used for storing training data!')
    exit()

LOG_FOUT = open(os.path.join(LOG_DIR, 'log_test_ep%s.txt' %opt.checkpoint.split('_')[0]), 'w')
LOG_FOUT.write(str(opt) + '\n')


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


if __name__ == '__main__':

    if opt.dataset_name == 'S3DIS':
        from P1_data_loader import S3DISDataset
        testset = S3DISDataset('test', opt)
    elif opt.dataset_name == 'ScanNet':
        from P1_data_loader import ScannetDatasetWholeScene
        testset = ScannetDatasetWholeScene(opt.data_dir, opt.num_point, opt.K, 'test')
    else:
        raise Exception('Please enter correct dataset name. Options: S3DIS or ScanNet')

    testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=False,
                                              num_workers=opt.nThreads, drop_last=True)
    log_string('#testing point clouds = %d' % len(testset))

    network = import_class('network', opt.model_name)(opt)
    if opt.checkpoint is not None: 
        checkpoint = torch.load(os.path.join(LOG_DIR, 'checkpoints', opt.checkpoint))
        network.load_state_dict(checkpoint['state_dict'])

    network = network.cuda()
    
    softmax = torch.nn.Softmax(dim=1)

    print('Evaluating begins..')
    with torch.no_grad():

        predicted_label_total = []
        gt_label_total = []

        for i, data in enumerate(testloader):
            input_pc, knn_idx, input_label = data

            if opt.dataset_name == 'ScanNet':
                input_pc = input_pc.view(-1, input_pc.shape[2], input_pc.shape[3]).cuda()
                knn_idx = knn_idx.view(-1, knn_idx.shape[2], knn_idx.shape[3]).long().cuda()
                input_label = input_label.view(-1, input_label.shape[2]).long().cuda()
            else:
                input_pc = input_pc.cuda()
                knn_idx = knn_idx.long().cuda()
                input_label = input_label.long().cuda()

            network.eval()

            # When evaluate ScanNet whole scene, process the scene in batch if the scene contains too many blocks.
            if input_pc.shape[0] > opt.max_volumes:
                num_batches = math.ceil(input_pc.shape[0] / opt.max_volumes)
                # print('The number of resultant volumes (%d) is larger than max_volumes (%d), processing in %d batches'
                #      %(input_pc.shape[0], opt.max_volumes, num_batches))
                pred = []

                for j in range(num_batches):
                    start_idx = j*opt.max_volumes
                    if j != num_batches -1:
                        end_idx = (j+1)*opt.max_volumes
                    else:
                        end_idx = input_pc.shape[0] + 1
                    pc_slice = input_pc[start_idx:end_idx,:,:]
                    knn_slice = knn_idx[start_idx:end_idx,:,:]

                    score_slice = network(pc_slice, knn_slice)
                    pred_slice = softmax(score_slice)
                    pred.append(pred_slice)
                pred = torch.cat(pred, dim=0)
            else:
                score = network(input_pc, knn_idx)
                pred = softmax(score)

            # accumulate accuracy
            _, predicted_label = torch.max(pred.detach(), dim=1, keepdim=False)

            predicted_label_total.append(predicted_label.cpu().detach())
            gt_label_total.append(input_label.cpu().detach())

        # compute iou
        predicted_label_total = torch.cat(predicted_label_total, dim=0).view(-1, opt.num_point)
        print(predicted_label_total.size())
        gt_label_total = torch.cat(gt_label_total, dim=0).view(-1, opt.num_point)
        if opt.dataset_name == 'S3DIS':
            test_accuracy, test_macc, test_iou, iou_perclass = compute_iou(predicted_label_total, gt_label_total, opt.classes)
        elif opt.dataset_name == 'ScanNet':
            test_accuracy, test_macc, test_iou, iou_perclass = compute_iou_scannet(predicted_label_total, gt_label_total, opt.classes)
        log_string('Test overall accuracy is %f' %test_accuracy)    
        log_string('Test_macc is %f' %test_macc)
        log_string('Test_iou is %f' %test_iou)
        log_string(str(iou_perclass))

    LOG_FOUT.close()
