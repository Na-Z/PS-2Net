#!/usr/bin/python3
"""Merge blocks and evaluate scannet"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os,sys
# import plyfile
import numpy as np
import argparse
import h5py
import pickle

CLASS_ = 21

def compute_iou(pred_label, gt_label, NUM_CLASS):
    """
    :param pred_label: (N) array
    :param gt_label: (N) array
    :return: iou: scaler
    """
    gt_classes = [0 for _ in range(NUM_CLASS)]
    positive_classes = [0 for _ in range(NUM_CLASS)]
    true_positive_classes = [0 for _ in range(NUM_CLASS)]

    for i in range(gt_label.shape[0]):
        gt_l = int(gt_label[i])
        pred_l = int(pred_label[i])
        gt_classes[gt_l] += 1
        positive_classes[pred_l] += 1
        true_positive_classes[gt_l] += int(gt_l == pred_l)


    #discard the unannotated class
    gt_classes = gt_classes[1:]
    positive_classes = positive_classes[1:]
    true_positive_classes = true_positive_classes[1:]

    print('Overall Precision: {0}'.format(sum(true_positive_classes)/float(sum(positive_classes))))
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

    return m_acc, m_iou, iou_list



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafolder', '-d', help='Path to input *_pred.h5', required=True)
    parser.add_argument('--picklefile', '-p', help='Path to scannet_test.pickle', required=True)
    args = parser.parse_args()
    print(args)

    file_list = os.listdir(args.datafolder)
    pred_list = [pred for pred in file_list if pred.split(".")[-1] == "h5" and "pred" in pred]

    pts_acc_list = []
    vox_acc_list = []

    #load scannet_test.pickle file
    file_pickle = open(args.picklefile, 'rb')
    xyz_all = pickle.load(file_pickle, encoding='latin1') # encoding keyword for python3
    labels_all = pickle.load(file_pickle, encoding='latin1')
    file_pickle.close()

    pickle_dict = {}
    for room_idx, xyz in enumerate(xyz_all):

        room_pt_num = xyz.shape[0]
        room_dict = {}

        room_dict["merged_label_zero"] = np.zeros((room_pt_num),dtype=int)
        room_dict["merged_confidence_zero"] = np.zeros((room_pt_num),dtype=float)
        room_dict["merged_label_half"] = np.zeros((room_pt_num), dtype=int)
        room_dict["merged_confidence_half"] = np.zeros((room_pt_num), dtype=float)
        room_dict["final_label"] = np.zeros((room_pt_num), dtype=int)

        pickle_dict[room_idx] = room_dict

    # load block preds and merge them to room scene
    for pred_file in pred_list:

        print("process:", os.path.join(args.datafolder, pred_file))
        test_file = pred_file.replace("_pred","")

        # load pred .h5
        data_pred = h5py.File(os.path.join(args.datafolder, pred_file))

        pred_labels_seg = data_pred['label_seg'][...].astype(np.int64)
        pred_indices = data_pred['indices_split_to_full'][...].astype(np.int64)
        pred_confidence = data_pred['confidence'][...].astype(np.float32)
        pred_data_num = data_pred['data_num'][...].astype(np.int64)

        
        if 'zero' in pred_file:
            for b_id in range(pred_labels_seg.shape[0]):
                indices_b = pred_indices[b_id]
                for p_id in range(pred_data_num[b_id]):
                    room_indices = indices_b[p_id][0]
                    inroom_indices = indices_b[p_id][1]
                    pickle_dict[room_indices]["merged_label_zero"][inroom_indices] = pred_labels_seg[b_id][p_id]
                    pickle_dict[room_indices]["merged_confidence_zero"][inroom_indices] = pred_confidence[b_id][p_id]
        else:
            for b_id in range(pred_labels_seg.shape[0]):
                indices_b = pred_indices[b_id]
                for p_id in range(pred_data_num[b_id]):
                    room_indices = indices_b[p_id][0]
                    inroom_indices = indices_b[p_id][1]
                    pickle_dict[room_indices]["merged_label_half"][inroom_indices] = pred_labels_seg[b_id][p_id]
                    pickle_dict[room_indices]["merged_confidence_half"][inroom_indices] = pred_confidence[b_id][p_id]

    for room_id in pickle_dict.keys():

        final_label = pickle_dict[room_id]["final_label"]
        merged_label_zero = pickle_dict[room_id]["merged_label_zero"]
        merged_label_half = pickle_dict[room_id]["merged_label_half"]
        merged_confidence_zero = pickle_dict[room_id]["merged_confidence_zero"]
        merged_confidence_half = pickle_dict[room_id]["merged_confidence_half"]

        final_label[merged_confidence_zero >= merged_confidence_half] = merged_label_zero[merged_confidence_zero >= merged_confidence_half]
        final_label[merged_confidence_zero < merged_confidence_half] = merged_label_half[merged_confidence_zero < merged_confidence_half]

    # eval

    #point-level gt and predictions
    label_all = []
    pred_all = []

    #voxel-level gt and predictions
    uvlabel_all = []
    uvpred_all = []

    for room_id, pts in enumerate(xyz_all):

        label = labels_all[room_id]
        pred = pickle_dict[room_id]["final_label"]
        data_num = pts.shape[0]

        label_all.extend(label)
        pred_all.extend(pred)

        # compute pts acc (ignore label 0 which is scannet unannotated)
        c_accpts = np.sum(np.equal(pred,label))
        c_ignore = np.sum(np.equal(label,0))
        pts_acc_list.append([c_accpts, data_num - c_ignore])

        # compute voxel accuracy (follow scannet and pointnet++)
        res = 0.0484
        coordmax = np.max(pts, axis=0)
        coordmin = np.min(pts, axis=0)
        nvox = np.ceil((coordmax - coordmin) / res)
        vidx = np.ceil((pts - coordmin) / res)
        vidx = vidx[:, 0] + vidx[:, 1] * nvox[0] + vidx[:, 2] * nvox[0] * nvox[1]
        uvidx, vpidx = np.unique(vidx, return_index=True)

        # compute voxel label
        uvlabel = np.array(label)[vpidx]

        # compute voxel pred (follow pointnet++ majority voting)
        uvpred_tp = []
        label_pred_dict = {}

        for uidx in uvidx:
            label_pred_dict[int(uidx)] = []
        for k, p in enumerate(pred):
            label_pred_dict[int(vidx[k])].append(p)
        for uidx in uvidx:
            uvpred_tp.append(np.argmax(np.bincount(label_pred_dict[int(uidx)])))

        uvlabel_all.extend(uvlabel)
        uvpred_all.extend(np.array(uvpred_tp))

        # compute voxel accuracy (ignore label 0 which is scannet unannotated)
        c_accvox = np.sum(np.equal(uvpred_tp, uvlabel))
        c_ignore = np.sum(np.equal(uvlabel,0))

        vox_acc_list.append([c_accvox, (len(uvlabel) - c_ignore)])

    # compute avg pts acc
    pts_acc_sum = np.sum(pts_acc_list,0)
    print("pts acc", pts_acc_sum[0]*1.0/pts_acc_sum[1])

    #compute avg voxel acc
    vox_acc_sum = np.sum(vox_acc_list,0)
    print("voxel acc", vox_acc_sum[0]*1.0/vox_acc_sum[1])

    #compute pts miou
    label_all = np.array(label_all)
    print('pts_label_all.shape:{}'.format(label_all.shape))
    pred_all = np.array(pred_all)
    print('pts_pred_all.shape:{}'.format(pred_all.shape))
    pts_macc, pts_miou, pts_iou = compute_iou(pred_all, label_all, CLASS_)
    print('Test_pts_macc is %f' %pts_macc)
    print('Test_pts_miou is %f' %pts_miou)
    print(str(pts_iou))

    #compute voxel miou
    uvlabel_all = np.array(uvlabel_all)
    print('voxel_label_all.shape:{}'.format(uvlabel_all.shape))
    uvpred_all = np.array(uvpred_all)
    print('voxel_pred_all.shape:{}'.format(uvpred_all.shape))   
    uv_macc, uv_miou, uv_iou = compute_iou(uvpred_all, uvlabel_all, CLASS_)
    print('Test_voxel_macc is %f' %uv_macc)
    print('Test_voxel_miou is %f' %uv_miou)
    print(str(uv_iou))

if __name__ == '__main__':
    main()
