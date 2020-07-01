""" 
Convert original dataset files to data_label file (Format: *.npy, each line is XYZRGBL).
We aggregated all the points from each instance in the room.
"""

import os
import glob
import numpy as np


def collect_point_label(anno_path, out_filename, file_format='txt'):
    """ Convert original dataset files to data_label file (each line is XYZRGBL).
        We aggregated all the points from each instance in the room.

    Args:
        anno_path: path to annotations. e.g. Area_1/office_2/Annotations/
        out_filename: path to save collected points and labels (each line is XYZRGBL)
        file_format: txt or numpy, determines what file format to save.
    Returns:
        None
    Note:
        the points are shifted before save, the most negative point is now at origin.
    """
    points_list = []
    
    for f in glob.glob(os.path.join(anno_path, '*.txt')):
        cls = os.path.basename(f).split('_')[0]
        if cls not in g_classes: # note: in some room there is 'staris' class..
            cls = 'clutter'
        points = np.loadtxt(f)
        labels = np.ones((points.shape[0],1)) * g_class2label[cls]
        points_list.append(np.concatenate([points, labels], 1)) # Nx7
    
    data_label = np.concatenate(points_list, 0)
    xyz_min = np.amin(data_label, axis=0)[0:3]
    data_label[:, 0:3] -= xyz_min
    
    if file_format=='txt':
        fout = open(out_filename, 'w')
        for i in range(data_label.shape[0]):
            fout.write('%f %f %f %d %d %d %d\n' % \
                          (data_label[i,0], data_label[i,1], data_label[i,2],
                           data_label[i,3], data_label[i,4], data_label[i,5],
                           data_label[i,6]))
        fout.close()
    elif file_format=='numpy':
        np.save(out_filename, data_label)
    else:
        print('ERROR!! Unknown file format: %s, please use txt or numpy.' % \
            (file_format))
        exit()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', '-s', help='Path to S3DIS raw data', required=True) #'*/Stanford3dDataset_v1.2_Aligned_Version/'
    parser.add_argument('--dst_dir', default='../../Datasets/S3DIS/P1/npy_data',
                        help='Path to store converted numpy files of ScanNet')
    args = parser.parse_args()
    print(args)

    SRC_DIR = args.src_dir
    DES_DIR = args.dst_dir

    anno_paths = [line.rstrip() for line in open('../../Datasets/S3DIS/meta/anno_paths.txt')]
    anno_paths = [os.path.join(SRC_DIR, p) for p in anno_paths]

    g_classes = [x.rstrip() for x in open('../../Datasets/S3DIS/meta/class_names.txt')]
    g_class2label = {cls: i for i, cls in enumerate(g_classes)}

    if not os.path.exists(DES_DIR): os.mkdir(DES_DIR)

    fout = open(os.path.join(DES_DIR, 'npy_file_list.txt'), 'w')
    # Note: there is an extra character in the v1.2 data in Area_5/hallway_6/ceiling_1.txt/line 180389. It's fixed manually.
    for anno_path in anno_paths:
        print(anno_path)
        try:
            elements = anno_path.split('/')
            out_filename = elements[-3] + '_' + elements[-2] + '.npy' # Area_1_hallway_1.npy
            collect_point_label(anno_path, os.path.join(DES_DIR, out_filename), 'numpy')
            fout.write('%s\n' %out_filename)
        except:
            print(anno_path, 'ERROR!!')
    fout.close()
