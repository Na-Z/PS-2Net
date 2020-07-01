###Prepare ScanNet Data according to data preparation setups in PointNet++ (P3):
We exactly follow the instructions in [PointNet++](https://github.com/charlesq34/pointnet2/tree/master/scannet) 
to prepare data for ScanNet. The K-Nearest-Neighbors are computed when loading the data in `P1_data_loader.py`.

You can download the [preprocessed data (1.72G)](https://shapenet.cs.stanford.edu/media/scannet_data_pointnet2.zip) provided 
by PointNet++ official github repo. After unzipping the downloaded file, move the two pickle files into `./datasets/ScanNet/P1`.

Or you can follow the instructions in [PointNet++](https://github.com/charlesq34/pointnet2/tree/master/scannet) to 
prepare the data by yourself.

Note that we used the data provided by PointNet++ in our experiments. 

The `./datasets/ScanNet/P3/` directory should look like this:
```bash
├── ScanNet
    ├── P3
        ├── scannet_train.pickle
        ├── scannet_test.pickle
```

###Prepare ScanNet Data according to data preparation setups in PointCNN (P2):
We exactly follow the instructions in [PointCNN](https://github.com/yangyanli/PointCNN) 
to prepare data for ScanNet. The K-Nearest-Neighbors are computed when loading the data in `P2_data_loader.py`.

PointCNN conducts further processing on [PointNet++ preprocessed data (1.72G)](https://shapenet.cs.stanford.edu/media/scannet_data_pointnet2.zip):

    python prepare_scannet_seg_data.py -s ../../datasets/ScanNet/P3/ -d ../../datasets/ScanNet/P2/
    python prepare_scannet_seg_filelists.py -f ../../datasets/ScanNet/P2/
    
Eventually, the `./datasets/ScanNet/P2/` directory should look like this:
```bash
├── ScanNet
    ├── P2
        ├── filelists
            ├── train_files_g_0.txt  
            ├── train_files_g_1.txt 
            ├── train_files_g_2.txt
        ├── train
            ├── half_0.h5 
            ├── ...
            ├── half_11.h5
            ├── zero_0.h5
            ├── ...  
            ├── zero_11.h5
        ├── test
            ├── half_0.h5 
            ├── ...
            ├── half_3.h5
            ├── zero_0.h5
            ├── ...  
            ├── zero_3.h5
        ├── train_files.txt
        ├── test_files.txt    
```