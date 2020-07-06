#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import h5py


#==================== Merge S3DIS files ========================
def merge_files(area_dir):
    categories_list = os.listdir(area_dir)

    for category in categories_list:
        output_path = os.path.join(area_dir,category,"pred.npy")
        label_length = np.load(os.path.join(area_dir,category,"label.npy")).shape[0]

        merged_label_zero = np.zeros((label_length),dtype=int)
        merged_confidence_zero = np.zeros((label_length),dtype=float)
        merged_label_half = np.zeros((label_length), dtype=int)
        merged_confidence_half = np.zeros((label_length), dtype=float)

        final_label = np.zeros((label_length), dtype=int)
        pred_list = [pred for pred in os.listdir(os.path.join(area_dir,category))
                     if pred.split(".")[-1] == "h5" and "pred" in pred]
        for pred_file in pred_list:
            print(os.path.join(area_dir,category, pred_file))
            data = h5py.File(os.path.join(area_dir,category, pred_file))
            labels_seg = data['label_seg'][...].astype(np.int64)
            indices = data['indices_split_to_full'][...].astype(np.int64)
            confidence = data['confidence'][...].astype(np.float32)
            data_num = data['data_num'][...].astype(np.int64)

            if 'zero' in pred_file:
                for i in range(labels_seg.shape[0]):
                    merged_label_zero[indices[i][:data_num[i]]] = labels_seg[i][:data_num[i]]
                    merged_confidence_zero[indices[i][:data_num[i]]] = confidence[i][:data_num[i]]
            else:
                for i in range(labels_seg.shape[0]):
                    merged_label_half[indices[i][:data_num[i]]] = labels_seg[i][:data_num[i]]
                    merged_confidence_half[indices[i][:data_num[i]]] = confidence[i][:data_num[i]]

        final_label[merged_confidence_zero >= merged_confidence_half] = merged_label_zero[merged_confidence_zero >= merged_confidence_half]
        final_label[merged_confidence_zero < merged_confidence_half] = merged_label_half[merged_confidence_zero < merged_confidence_half]

        np.savetxt(output_path, final_label, fmt='%d')
        print("saved to ",output_path)


#==================== Evaluate S3DIS  ========================
def evaluate_area(area_dir):

    gt_label_filenames = []
    pred_label_filenames = []

    rooms = os.listdir(area_dir)
    for room in rooms:
        path_gt_label = os.path.join(area_dir, room,'label.npy')
        path_pred_label = os.path.join(area_dir, room,'pred.npy')
        pred_label_filenames.append(path_pred_label)
        gt_label_filenames.append(path_gt_label)

    num_room = len(gt_label_filenames)

    print(num_room)
    print(len(pred_label_filenames))
    assert(num_room == len(pred_label_filenames))

    gt_classes = [0 for _ in range(13)]
    positive_classes = [0 for _ in range(13)]
    true_positive_classes = [0 for _ in range(13)]

    for i in range(num_room):
        print(i,"/"+str(num_room))
        print(pred_label_filenames[i])
        pred_label = np.loadtxt(pred_label_filenames[i])
        gt_label = np.load(gt_label_filenames[i])
        for j in range(gt_label.shape[0]):
            gt_l = int(gt_label[j])
            pred_l = int(pred_label[j])
            gt_classes[gt_l] += 1
            positive_classes[pred_l] += 1
            true_positive_classes[gt_l] += int(gt_l==pred_l)

    print(gt_classes)
    print(positive_classes)
    print(true_positive_classes)

    print('\tOverall accuracy: {0}'.format(sum(true_positive_classes)/float(sum(positive_classes))))

    print('\tIoU:')
    iou_list = []
    acc_list = []
    for i in range(13):
        acc = true_positive_classes[i]/float(gt_classes[i])
        acc_list.append(acc)
        iou = true_positive_classes[i]/float(gt_classes[i]+positive_classes[i]-true_positive_classes[i])
        print('\t\t class %d-%f' %(i, iou))
        iou_list.append(iou)

    print('\tmAcc is %f' %(sum(acc_list)/13.0))
    print('\tmIoU is %f' %(sum(iou_list)/13.0))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafolder', default='../datasets/S3DIS/P2/', help='Path to input *_pred.h5')
    parser.add_argument('--test_area', type=int, default=5, help='Which areas to use for test, option:[1-6] [default: 5]')
    args = parser.parse_args()
    print(args)

    area_dir = os.path.join(args.datafolder, 'Area_%d' %args.test_area)
    merge_files(area_dir)

    print('=========Evaluate Area%d==========' %args.test_area)
    evaluate_area(area_dir)
