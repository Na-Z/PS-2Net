#!/usr/bin/python()
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import argparse

object_dict = {
            'clutter':   12,
            'ceiling':   0,
            'floor':     1,
            'wall':      2,
            'beam':      3,
            'column':    4,
            'door':      6,
            'window':    5,
            'table':     7,
            'chair':     8,
            'sofa':      9,
            'bookcase': 10,
            'board':    11}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', '-s', help='Path to S3DIS data directory')
    parser.add_argument('--dst_dir', '-d', help='Path to output directory')
    args = parser.parse_args()

    SRC_DIR = args.src_dir
    DST_DIR = args.dst_dir

    path_Dir_Areas = ['Area_6', 'Area_3', 'Area_4', 'Area_1', 'Area_2']
    print(path_Dir_Areas)


    for Area in path_Dir_Areas:
        path_Dir_Rooms = os.listdir(os.path.join(SRC_DIR, Area))
        for Room in path_Dir_Rooms:
            xyz_Room = np.zeros((1,6))
            label_Room = np.zeros((1,1))
            path_Annotations = os.path.join(SRC_DIR, Area, Room, "Annotations")
            print(path_Annotations)
            # make store directories
            path_prepare_label = os.path.join(DST_DIR, Area, Room)
            if not os.path.exists(path_prepare_label):
                os.makedirs(path_prepare_label)
            #############################
            path_objects = os.listdir(path_Annotations)
            for Object in path_objects:
                if object_dict.has_key(Object.split("_",1)[0]):
                    print(Object.split("_",1)[0] + " value:" ,object_dict[Object.split("_",1)[0]])
                    xyz_object = np.loadtxt(os.path.join(path_Annotations,Object))[:,:]#(N,6)
                    label_object = np.tile([object_dict[Object.split("_",1)[0]]],(xyz_object.shape[0],1))#(N,1)
                else:
                    continue

                xyz_Room = np.vstack((xyz_Room,xyz_object))
                label_Room = np.vstack((label_Room,label_object))

            xyz_Room = np.delete(xyz_Room,[0],0)
            label_Room = np.delete(label_Room,[0],0)

            np.save(path_prepare_label+"/xyzrgb.npy",xyz_Room)
            np.save(path_prepare_label+"/label.npy",label_Room)


