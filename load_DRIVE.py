#import DRIVE 2004
import kagglehub

# Download latest version
path = kagglehub.dataset_download("zionfuo/drive2004")

print("Path to dataset files:", path)
import os
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt

root_dir = "/root/.cache/kagglehub/datasets/zionfuo/drive2004/versions/1/DRIVE"
train_images_dir = os.path.join(root_dir, "training", "images")
train_masks_dir = os.path.join(root_dir, "training", "mask")
train_vessel_masks_dir = os.path.join(root_dir, "training", "1st_manual")

test_images_dir = os.path.join(root_dir, "test", "images")
test_vessel_masks_dir = os.path.join(root_dir, "test", "1st_manual")
test_masks_dir = os.path.join(root_dir, "test", "mask")

def load_data(image_name, dataset_type='train'):

    if dataset_type == 'train':
        images_dir = train_images_dir
        vessel_masks_dir = train_vessel_masks_dir
        masks_dir = train_masks_dir
    elif dataset_type == 'test':
        images_dir = test_images_dir
        vessel_masks_dir = test_vessel_masks_dir
        masks_dir = test_masks_dir

    image_path = os.path.join(images_dir, image_name)
    image = Image.open(image_path).convert('L')
    image = img_to_array(image) / 255.0

    base_name = image_name.split('_')[0]

    if dataset_type == 'train':
        vessel_mask_name = base_name + "_manual1.gif"
    else:
        vessel_mask_name = base_name + "_manual1.gif"
    vessel_mask_path = os.path.join(vessel_masks_dir, vessel_mask_name)

    vessel_mask = Image.open(vessel_mask_path).convert('L')
    vessel_mask = img_to_array(vessel_mask) / 255.0

    if dataset_type == 'train':
        optic_disk_mask_name = base_name + "_training_mask.gif"
    else:
        optic_disk_mask_name = base_name + "_test_mask.gif"
    optic_disk_mask_path = os.path.join(masks_dir, optic_disk_mask_name)

    optic_disk_mask = Image.open(optic_disk_mask_path).convert('L')
    optic_disk_mask = img_to_array(optic_disk_mask) / 255.0

    return image, vessel_mask, optic_disk_mask

# visualise 
def visualize_sample(image_name, dataset_type='train'):

    image, vessel_mask, optic_disk_mask = load_data(image_name, dataset_type)

    plt.figure(figsize=(12, 8))

    plt.subplot(1, 3, 1)
    plt.title("Input Image")
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Vessel Mask")
    plt.imshow(vessel_mask, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Optic Disk Mask")
    plt.imshow(optic_disk_mask, cmap='gray')
    plt.axis('off')

    plt.show()

# Example
image_name = '21_training.tif'
visualize_sample(image_name, 'train')

image_name = '01_test.tif'
visualize_sample(image_name, 'test')

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch

class DRIVEVesselDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.image_names = [f for f in os.listdir(images_dir) if f.endswith(".tif")]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        base_name = img_name.split('_')[0]

        image = Image.open(os.path.join(self.images_dir, img_name)).convert("RGB")
        mask = Image.open(os.path.join(self.masks_dir, base_name + "_manual1.gif")).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        mask = (mask > 0).float()  

        return image, mask

