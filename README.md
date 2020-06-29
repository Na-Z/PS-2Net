# PS^2Net: A Locally and Globally Aware Network for Point-Based SemanticSegmentation
Created by <a href="https://github.com/Na-Z" target="_blank">Na Zhao</a> from 
<a href="http://www.nus.edu.sg/" target="_blank">National University of Singapore</a>

![teaser](https://github.com/Na-Z/PS-2Net/blob/master/teaser.pdf)

## Introduction
This repository contains the PyTorch implementation for our ICPR 2020 Paper 
"PS^2Net: A Locally and Globally Aware Network for Point-Based SemanticSegmentation" by Na Zhao, Tat Seng Chua, Gim Hee Lee 
[[arXiv](https://arxiv.org/pdf/1908.05425.pdf)]

In this paper, we present the PS^2-Net - a locally and globally aware deep learning framework for semantic segmentation 
on 3D scene-level point clouds. In order to deeply incorporate local structures and global context to support 3D scene 
segmentation, our network is built on four repeatedly stacked encoders, where each encoder has twobasic components: 
EdgeConv that captures local structures and NetVLAD that models global context. Different from existing state-of-the-art 
methods for point-based scene semantic segmentation that either violate or do not achieve permutation invariance, our 
PS^2-Net is designed to be permutation invariant which is an essential property of any deep network used to process 
unordered point clouds. We further provide theoretical proof to guarantee the permutation invariance property of our 
network. We perform extensive experiments on two large-scale 3D indoor scene datasets and demonstrate that our PS^2-Net 
is able to achieve state-of-the-art performances as compared to existing approaches.

Code will be released soon.
