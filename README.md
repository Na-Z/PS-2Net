# PS<sup>2</sup>Net: A Locally and Globally Aware Network for Point-Based Semantic Segmentation
Created by <a href="https://github.com/Na-Z" target="_blank">Na Zhao</a> from 
<a href="http://www.nus.edu.sg/" target="_blank">National University of Singapore</a>

![teaser](https://github.com/Na-Z/PS-2Net/blob/master/teaser.jpg)

## Introduction
This repository contains the PyTorch implementation for our ICPR 2020 Paper 
"*PS<sup>2</sup>Net: A Locally and Globally Aware Network for Point-Based SemanticSegmentation*" by Na Zhao, Tat Seng Chua, Gim Hee Lee 
[[arXiv](https://arxiv.org/pdf/1908.05425.pdf)]

In this paper, we present the PS<sup>2</sup>-Net - a locally and globally aware deep learning framework for semantic segmentation 
on 3D scene-level point clouds. In order to deeply incorporate local structures and global context to support 3D scene 
segmentation, our network is built on four repeatedly stacked encoders, where each encoder has twobasic components: 
EdgeConv that captures local structures and NetVLAD that models global context. Different from existing state-of-the-art 
methods for point-based scene semantic segmentation that either violate or do not achieve permutation invariance, our 
PS<sup>2</sup>-Net is designed to be permutation invariant which is an essential property of any deep network used to process 
unordered point clouds. We further provide theoretical proof to guarantee the permutation invariance property of our 
network. We perform extensive experiments on two large-scale 3D indoor scene datasets and demonstrate that our PS<sup>2</sup>-Net 
is able to achieve state-of-the-art performances as compared to existing approaches.

## Setup
- Install `python` --This repo is tested with `python 3.6.5`.
- Install `pytorch` with CUDA -- This repo is tested with `torch 0.4.0`, `CUDA 9.0`. 
It may wrk with newer versions, but that is not gauranteed.
- Install `faiss` with CPU version by `conda install faiss-cpu -c pytorch` -- This repo is tested with `faiss 1.4.0`
- Install dependencies
    ```
    pip install -r requirements.txt
    ```
    
## Usage
### Data preparation
For S3DIS, follow the [README](https://github.com/Na-Z/PS-2Net/blob/master/preprocess/s3dis/README.md) under `./preprocess/s3dis` folder.

For ScanNet, follow the [README](https://github.com/Na-Z/PS-2Net/blob/master/preprocess/scannet/README.md) under `./preprocess/scannet` folder.

### Visualization
We use visdom for visualization. Loss values and performance are plotted in real-time. Please start the visdom server before training:
    ```python -m visdom.server```

The visualization results can be viewed in browser with the address of: `http://localhost:8097`.



### Running experiments on S3DIS
#### Under data preparation setup (P1):
+ train on each area:
    ```
    python main_P1/train.py --dataset_name S3DIS --data_dir ./datasets/S3DIS/P1/ --classes 13 --input_feat 9 --log_dir $LOG_DIR  --test_area $Area_Index
    ```
+ test on the corresponding area:
    ```
    python main_P1/test.py --dataset_name S3DIS --data_dir ./datasets/S3DIS/P1/ --classes 13 --input_feat 9 --log_dir $LOG_DIR  --checkpoint $CHECKPOINT_FILENAME --test_area $Area_Index
    ```
    
#### Under data preparation setup (P2): 
+ train on each area:
    ```
    python main_P2/train.py --dataset_name S3DIS --dataset_size 114004 --data_dir ./datasets/S3DIS/P2/ --classes 13 --input_feat 6 --log_dir $LOG_DIR  --test_area $Area_Index
    ```
+ test on the corresponding area:
    ```
    python main_P2/inference.py --dataset_name S3DIS --data_dir ./datasets/S3DIS/P2/ --classes 13 --input_feat 6 --log_dir $LOG_DIR  --checkpoint $CHECKPOINT_FILENAME --test_area $Area_Index
    python main_P2/eval_s3dis.py --datafolder ./datasets/S3DIS/P2/ --test_area $Area_Index
    ```    

Note that these commands are for training and evaluating only one area (specified by `--test_area $Area_Index` option) validation. 
Please iterate `--test_area` option to obtain results on other areas. The final result is computed based on **6-fold cross validation**.


### Running experiments on ScanNet
#### Under data preparation setup (P3):
+ train:
    ```
    python main_P1/train.py --dataset_name ScanNet --data_dir ./datasets/ScanNet/P3/ --classes 21 --input_feat 3 --log_dir $LOG_DIR 
    ```
+ test:
    ```
    python main_P1/test.py --dataset_name ScanNet --data_dir ./datasets/ScanNet/P3/ --classes 21 --input_feat 3 --log_dir $LOG_DIR  --checkpoint $CHECKPOINT_FILENAME
    ```
    
#### Under data preparation setup (P2): 
+ train:
    ```
    python main_P2/train.py --dataset_name ScanNet --dataset_size 93402 --data_dir ./datasets/ScanNet/P2/ --classes 21 --input_feat 3 --log_dir $LOG_DIR  
    ```
+ test:
    ```
    python main_P2/inference.py --dataset_name ScanNet --data_dir ./datasets/ScanNet/P2/ --classes 21 --input_feat 3 --log_dir $LOG_DIR  --checkpoint $CHECKPOINT_FILENAME 
    python main_P2/eval_scannet.py --datafolder ./datasets/ScanNet/P2/ --picklefile ./datasets/ScanNet/P3/
    ```    

## Citation
Please cite our paper if it is helpful to your research:

    @article{zhao2019ps,
      title={PS\^{} 2-Net: A Locally and Globally Aware Network for Point-Based Semantic Segmentation},
      author={Zhao, Na and Chua, Tat-Seng and Lee, Gim Hee},
      journal={arXiv preprint arXiv:1908.05425},
      year={2019}
    }


## Acknowledgements
Our implementation leverages on the source code or data from the following repositories:
- [PointNet](https://github.com/charlesq34/pointnet/)
- [PointNet++](https://github.com/charlesq34/pointnet2/)
- [PointCNN](https://github.com/yangyanli/PointCNN)
- [SO-Net](https://github.com/lijx10/SO-Net)