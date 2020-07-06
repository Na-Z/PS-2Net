import time
import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'data'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

import argparse
import torch

from visualizer import Visualizer
from model import Model, compute_iou, compute_iou_scannet


parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model_name', type=str, default='PS2Net', help='name of model')
parser.add_argument('--dataset_name', type=str, default='ScanNet', help='the dataset to use [options: S3DIS, ScanNet]')

parser.add_argument('--data_dir', default='../datasets/ScanNet/P3/', help='path to point clouds')
parser.add_argument('--block_size', type=float, default=1.0, help='the size of blocks to split rooms')
parser.add_argument('--stride', type=float, default=1.0, help='the stride to slide along xy plane to split rooms')
parser.add_argument('--classes', type=int, default=21, help='S3DIS or ScanNet')
parser.add_argument('--num_point', type=int, default=4096, help='Point number [default: 4096]')
parser.add_argument('--input_feat', type=int, default=3, help='The dimension of raw input features [Option: 3 or 9]')

parser.add_argument('--nThreads', default=6, type=int, help='# threads for loading data')
parser.add_argument('--batch_size', type=int, default=6, help='Batch Size during training')
parser.add_argument('--max_epoch', type=int, default=200, help='Epoch to run [default: 200]')
parser.add_argument('--log_dir', default='../log_scannet/P3_**', help='Log dir [default: log]')
parser.add_argument('--pretrain_path', type=str, default=None, 
                    help='path to pre-trained network parameters')

parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
parser.add_argument('--display_id', type=int, default=300, help='window id of the web display')
parser.add_argument('--iter_error_print', type=int, default=100, help='the number of iterations to print training error')
parser.add_argument('--best_iou', type=float, default=0.25, help='theasfold (testing mIoU) to save network')

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
parser.add_argument('--decoder_dropout', default='[0.3, 0, 0]', 
                    help='The dropout_ratio in the decoder network (the last dropout_ratio is set to 0)')

parser.add_argument('--activation', type=str, default='relu', help='activation function: relu, prelu, elu, leakyrelu')
parser.add_argument('--normalization', type=str, default='batch', help='normalization function: batch or instance')
parser.add_argument('--norm_momentum', type=float, default=0.1, help='normalization momentum, typically 0.1.')
parser.add_argument('--lossweight_path', type=str, default=None, help='the path to the label weights for weighted cross-entropy loss')

parser.add_argument('--optimizer', default='adam', help='adam or sgd [default: adam]')
parser.add_argument('--weight_decay', type=float, default=0.00001, help='weight decay value for optimizer [default: 0]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--decay_step', type=int, default=100, help='Decay step (epoch) for lr decay [default: 20]')
parser.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate for lr decay [default: 0.5]')

parser.add_argument('--augment', type=bool, default=False, help='Data augment or not [default: False]')
parser.add_argument('--test_area', type=str, default='5', help='Which areas to use for test, option:[1-6] [default: 5]')
opt = parser.parse_args()

args = vars(opt)
print('------------ Options -------------')
for k, v in sorted(args.items()):
    print('%s: %s' % (str(k), str(v)))
print('-------------- End ----------------')


LOG_DIR = opt.log_dir
if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(opt) + '\n')
save_model_dir = os.path.join(LOG_DIR, 'checkpoints/')
if not os.path.exists(save_model_dir): os.mkdir(save_model_dir)


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def print_current_errors(epoch, i, errors, t):
    message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
    for k, v in errors.items():
        message += '%s: %.3f ' % (k, v)
    log_string(message)


if __name__ == '__main__':

    if opt.dataset_name == 'S3DIS':
        from P1_data_loader import S3DISDataset
        trainset = S3DISDataset('train', opt)
        testset = S3DISDataset('test', opt)
    elif opt.dataset_name == 'ScanNet':
        from P1_data_loader import ScannetDataset
        trainset = ScannetDataset(opt.data_dir, opt.num_point, opt.K, 'train')
        testset = ScannetDataset(opt.data_dir, opt.num_point, opt.K, 'test')
    else:
        raise Exception('Please enter correct dataset name. Options: S3DIS or ScanNet')

    dataset_size = len(trainset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True,
                                              num_workers=opt.nThreads, drop_last=True)
    log_string('#training point clouds = %d' % dataset_size)
    testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=False,
                                             num_workers=opt.nThreads, drop_last=True)
    log_string('#testing point clouds = %d' % len(testset))

    visualizer = Visualizer(opt)
    model = Model(opt)
    if opt.pretrain_path is not None: 
        checkpoint = torch.load(opt.pretrain_path)
        model.network.load_state_dict(checkpoint['state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer'])

    print('Training begins..')
    best_iou = 0
    for epoch in range(opt.max_epoch):

        epoch_iter = 0
        for i, data in enumerate(trainloader):
            iter_start_time = time.time()
            epoch_iter += opt.batch_size

            input_pc, knn_idx, input_label = data

            model.set_input(input_pc, knn_idx, input_label)

            model.optimize()

            if i % opt.iter_error_print == 0:
                t = (time.time() - iter_start_time) / opt.batch_size
                errors = model.get_current_errors()
                print_current_errors(epoch, epoch_iter, errors, t)
                visualizer.plot_current_errors(epoch, float(epoch_iter) / dataset_size, errors)

        if epoch >= 0 and epoch % 3 == 0:
            print('Evaluating...')

            with torch.no_grad():
                batch_amount = 0
                model.test_loss.data.zero_()

                predicted_label_total = []
                gt_label_total = []

                for i, data in enumerate(testloader):
                    input_pc, knn_idx, input_label = data
                    model.set_input(input_pc, knn_idx, input_label)
                    model.test_model()

                    batch_size_test = input_label.size()[0]
                    batch_amount += batch_size_test

                    #accumulate loss
                    model.test_loss += model.loss.detach() * batch_size_test

                    _, predicted_label = torch.max(model.score.detach(), dim=1, keepdim=False)
                    predicted_label_total.append(predicted_label.cpu().detach())
                    gt_label_total.append(model.input_label.cpu().detach())

                model.test_loss /= batch_amount

                # compute iou
                predicted_label_total = torch.stack(predicted_label_total, dim=0).view(-1, opt.num_point)
                print(predicted_label_total.size())
                gt_label_total = torch.stack(gt_label_total, dim=0).view(-1, opt.num_point)
                if opt.dataset_name == 'S3DIS':
                    model.test_accuracy, model.test_macc, model.test_iou, iou_perclass = compute_iou(predicted_label_total, gt_label_total, opt.classes)
                elif opt.dataset_name == 'ScanNet':
                    model.test_accuracy, model.test_macc, model.test_iou, iou_perclass = compute_iou_scannet(predicted_label_total, gt_label_total, opt.classes)
                log_string(str(iou_perclass))

                current_test_iou = model.test_iou
                if current_test_iou > best_iou:
                    best_iou = current_test_iou
                    if current_test_iou > opt.best_iou:
                        print('test_iou>%d, saving network...' %opt.best_iou)
                        model.save_network(save_model_dir, '%d_%f' %(epoch, current_test_iou))
                log_string('Tested network. So far best IOU result is: %f' % best_iou)

        if epoch % opt.decay_step == 0 and epoch > 0:
            model.update_learning_rate()
            log_string('=====Epoch-%i: Learning_rate-%f' % (epoch, model.lr))

    LOG_FOUT.close()
