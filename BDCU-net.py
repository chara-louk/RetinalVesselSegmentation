#load model from github
!git clone https://github.com/rezazad68/BCDU-Net

#change target path
import shutil
import os

downloaded_drive_path = "/kaggle/input/drive2004/DRIVE"

target_drive_path = "/content/BCDU-Net/Retina Blood Vessel Segmentation/DRIVE"

shutil.copytree(downloaded_drive_path, target_drive_path)

#run the model
%cd /content/BCDU-Net/Retina\ Blood\ Vessel\ Segmentation
!python prepare_datasets_DRIVE.py

!python save_patch.py

#in the save_path.py change the number of patches to work (N_subimgs = 1000)
!python save_patch.py
!python train_retina.py
