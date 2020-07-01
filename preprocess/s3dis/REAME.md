###Prepare S3DIS Data according to data preparation setups in PointNet (P1):
We follow the instructions in [PointNet](https://github.com/charlesq34/pointnet/tree/master/sem_seg) 
to prepare data for S3DIS, except that we also compute the static K-Nearest-Neighbors for each point 
in the coordinate space.
Follows are the detail preparation setups:
1. Download [S3DIS Dataset Version 1.2](http://buildingparser.stanford.edu/dataset.html).
2. Re-organize raw data into `npy` files by running

   ```python s3dis_filetype_transfer.py -s $path_to_S3DIS_raw_data```
   
   The generated numpy files are stored in `./datasets/S3DIS/P1/npy_data/` by default.
3. To prepare the HDF5 data as in Pointnet, run 

    ```python preprocess_s3dis.py -s ../../datasets/S3DIS/P1/npy_data -d ../../datasets/S3DIS/P1/```
    
    One folder will be generated under `./datasets/S3DIS/P1/` by default.   

Eventually, the `./datasets/S3DIS/P1/` directory should look like this:
```bash
├── S3DIS
    ├── P1
        ├── npy_data
            ├── Area_1_conferenceRoom_1.npy
            ├── ... 
            ├── Area_6_pantry_1.npy
            ├── npy_file_list.txt
        ├── s3dis_bs1.0_s1.0_K30_4096_normalized
            ├── all_h5_files.txt
            ├── ply_data_all_0.h5
            ├── ...
            ├── ply_data_all_7.h5
            ├── room_filelist.txt
```


###Prepare S3DIS Data according to data preparation setups in PointCNN (P2):
We exactly follow the instructions in [PointCNN](https://github.com/yangyanli/PointCNN) 
to prepare data for S3DIS. The K-Nearest-Neighbors are computed when loading the data in `P2_data_loader.py`.

Run the following scripts after downloading [S3DIS Dataset Version 1.2](http://buildingparser.stanford.edu/dataset.html).

    python prepare_s3dis_label.py -s $path_to_S3DIS_raw_data -d ../../datasets/S3DIS/P2/
    python prepare_s3dis_data.py -f ../../datasets/S3DIS/P2/
    python prepare_s3dis_filelists.py -f ../../datasets/S3DIS/P2/
   
Eventually, the `./datasets/S3DIS/P2/` directory should look like this:
```bash
├── S3DIS
    ├── P2
        ├── Area_1
            ├── room_1 (e.g. conferenceRoom_1)  
                ├── half_0.h5
                ├── zero_0.h5
                ├── label.npy
                ├── xyzrgb.npy
            ├── ...           
        ├── ...
        ├── Area_6
        ├── filelists
            ├── train_files_for_val_on_Area_1_g_0.txt 
            ├── ...
            ├── train_files_for_val_on_Area_6_g_55.txt
        ├── train_files_for_val_on_Area_1.txt
        ├── ...
        ├── train_files_for_val_on_Area_6.txt
        ├── val_files_Area_1.txt
        ├── ...
        ├── val_files_Area_6.txt
```