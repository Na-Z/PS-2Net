import os
import numpy as np

import torch
import torch.nn as nn
from collections import OrderedDict

from network import *


def import_class(module_name, class_name):
    """
    dynamic import of a class from a given package
    :param module_name: path to the package
    :param class_name: class to be dynamically loaded
    :return: dynamically loaded class
    """
    try:
        print('Loading %s.%s ...' %(module_name, class_name))
        module = __import__(module_name)
        return getattr(module, class_name)
    except ModuleNotFoundError as exc:
        print('%s.%s could not be found' %(module_name, class_name))
        exit(1)


class Model():
    def __init__(self, opt):
        self.opt = opt

        Network = import_class('network', opt.model_name)
        self.network = Network(opt)

        if opt.lossweight_path is not None:
            labelweights = np.load(opt.lossweight_path)
            labelweights = torch.from_numpy(labelweights.astype(np.float32)).cuda()
        else:
            labelweights = None

        self.loss_function = nn.CrossEntropyLoss(weight=labelweights, size_average=True)

        if self.opt.gpu_id >= 0:
            self.network = self.network.cuda()
            self.loss_function = self.loss_function.cuda()

        self.lr = self.opt.learning_rate
        if self.opt.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.network.parameters(),
                                             lr=self.lr,
                                             momentum=self.opt.momentum,
                                             weight_decay=opt.weight_decay)
        elif self.opt.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.network.parameters(),
                                              lr=self.lr,
                                              betas=(0.9, 0.999),
                                              weight_decay=opt.weight_decay)

        self.input_data = torch.FloatTensor(self.opt.batch_size, 9, self.opt.num_point).uniform_()
        self.input_knn = torch.LongTensor(self.opt.batch_size, self.opt.num_point, self.opt.K).fill_(1)
        self.input_label = torch.LongTensor(self.opt.batch_size, self.opt.num_point).fill_(1)

        self.test_loss = torch.FloatTensor([0])
        self.test_accuracy = torch.FloatTensor([0])
        self.test_iou = torch.FloatTensor([0])
        self.test_macc = torch.FloatTensor([0])

        if self.opt.gpu_id >= 0:
            self.input_data = self.input_data.cuda()
            self.input_knn = self.input_knn.cuda()
            self.input_label = self.input_label.cuda()
            self.test_loss = self.test_loss.cuda()

    def set_input(self, input_data, knn_idx, input_label):
        self.input_data.resize_(input_data.size()).copy_(input_data)
        self.input_knn.resize_(knn_idx.size()).copy_(knn_idx)
        self.input_label.resize_(input_label.size()).copy_(input_label)
        self.data = self.input_data.detach()
        self.knn_idx = self.input_knn.detach()
        self.label = self.input_label.detach()

    def forward(self, epoch=None):
        self.score = self.network(self.data, self.knn_idx)
        try:
            assert self.score.shape[1] == self.opt.classes
        except Exception as e:
            print('Exception: The predicted number of classes is not equal to that of groud truth classes..')

    def optimize(self, epoch=None):
        self.network.train()
        self.forward(epoch=epoch)

        #input params:  score: (B,C,N)
        #               label: (B,N)
        self.loss = self.loss_function(self.score, self.label)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()


    def test_model(self):
        self.network.eval()
        self.forward()
        self.loss = self.loss_function(self.score, self.label)

    def get_current_visuals(self):
        #TODO
        pass

    def get_current_errors(self):
        _, predicted_label = torch.max(self.score.detach(), dim=1, keepdim=False)
        correct_mask = torch.eq(predicted_label, self.input_label).float()
        train_accuracy = torch.mean(correct_mask)

        return OrderedDict([
            ('train_loss', self.loss.item()),
            ('train_accuracy', train_accuracy),
            ('test_loss', self.test_loss.item()),
            ('test_oacc', self.test_accuracy),
            ('test_macc', self.test_macc),
            ('test_iou', self.test_iou)
        ])

    def save_network(self, save_dir, epoch_info):
        save_filename = '%s.pth' % epoch_info
        save_path = os.path.join(save_dir, save_filename)
        checkpoint = {'state_dict': self.network.state_dict(),
                      'optimizer': self.optimizer.state_dict()}
        torch.save(checkpoint, save_path)

    def update_learning_rate(self):
        lr = self.lr * self.opt.decay_rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        print('update encoder learning rate: %f -> %f' % (self.lr, lr))
        self.lr = lr



def compute_iou(predicted_label, gt_label, NUM_CLASS):
    """
    :param predicted_label: (B,N) tensor
    :param gt_label: (B,N) tensor
    :return: iou: scaler
    """
    gt_classes = [0 for _ in range(NUM_CLASS)]
    positive_classes = [0 for _ in range(NUM_CLASS)]
    true_positive_classes = [0 for _ in range(NUM_CLASS)]

    for i in range(gt_label.size()[0]):
        pred_pc = predicted_label[i]
        gt_pc = gt_label[i]

        for j in range(gt_pc.shape[0]):
            gt_l = int(gt_pc[j])
            pred_l = int(pred_pc[j])
            gt_classes[gt_l] += 1
            positive_classes[pred_l] += 1
            true_positive_classes[gt_l] += int(gt_l == pred_l)

    oa = sum(true_positive_classes)/float(sum(positive_classes))
    print('Overall accuracy: {0}'.format(oa))
    acc_list = []
    iou_list = []

    for i in range(NUM_CLASS):

        acc_class = true_positive_classes[i] / float(gt_classes[i])
        print('Class_%d: acc_class is %f' % (i, acc_class))
        acc_list.append(acc_class)

        iou_class = true_positive_classes[i] / float(gt_classes[i]+positive_classes[i]-true_positive_classes[i])
        print('Class_%d: iou_class is %f' % (i, iou_class))
        iou_list.append(iou_class)

    m_acc = np.asarray(acc_list).mean()
    m_iou = sum(iou_list)/NUM_CLASS

    return oa, m_acc, m_iou, iou_list


def compute_iou_scannet(predicted_label, gt_label, NUM_CLASS):
    """
    :param pred_label: (B,N) array
    :param gt_label: (B,N) array
    :return: iou: scaler
    """
    gt_classes = [0 for _ in range(NUM_CLASS)]
    positive_classes = [0 for _ in range(NUM_CLASS)]
    true_positive_classes = [0 for _ in range(NUM_CLASS)]

    for i in range(gt_label.size()[0]):
        pred_pc = predicted_label[i]
        gt_pc = gt_label[i]

        for j in range(gt_pc.shape[0]):
            gt_l = int(gt_pc[j])
            pred_l = int(pred_pc[j])
            gt_classes[gt_l] += 1
            positive_classes[pred_l] += 1
            true_positive_classes[gt_l] += int(gt_l == pred_l)

    #discard the unannotated class
    gt_classes = gt_classes[1:]
    positive_classes = positive_classes[1:]
    true_positive_classes = true_positive_classes[1:]

    oa = sum(true_positive_classes)/float(sum(positive_classes))
    print('Overall accuracy: {0}'.format(oa))
    acc_list = []
    iou_list = []

    for i in range(NUM_CLASS-1):
        acc_class = true_positive_classes[i] / float(gt_classes[i])
        print('Class_%d: acc_class is %f' % (i, acc_class))
        acc_list.append(acc_class)

        iou_class = true_positive_classes[i] / float(gt_classes[i]+positive_classes[i]-true_positive_classes[i])
        print('Class_%d: iou_class is %f' % (i, iou_class))
        iou_list.append(iou_class)

    m_acc = np.asarray(acc_list).mean()
    m_iou = sum(iou_list)/(NUM_CLASS-1)

    return oa, m_acc, m_iou, iou_list
