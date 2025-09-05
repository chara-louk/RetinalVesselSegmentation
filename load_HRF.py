import kagglehub
from torch.utils.data import random_split, DataLoader
import torchvision.transforms as T

# Download latest version
path = kagglehub.dataset_download("ipythonx/retinal-vessel-segmentation")

print("Path to dataset files:", path)

import os

hrf_images_dir = "/root/.cache/kagglehub/datasets/ipythonx/retinal-vessel-segmentation/versions/1/HRF/images"
hrf_masks_dir = "/root/.cache/kagglehub/datasets/ipythonx/retinal-vessel-segmentation/versions/1/HRF/manual1"

import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch

class HRFDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

        self.image_names = sorted([
            f for f in os.listdir(images_dir)
            if f.lower().endswith(('.jpg', '.jpeg'))
        ])
        self.mask_names = sorted([
            f for f in os.listdir(masks_dir)
            if f.endswith('.tif')
        ])

        self.image_mask_map = self.create_image_mask_mapping()

    def create_image_mask_mapping(self):
        image_mask_map = []

        for img_name in self.image_names:
            base_id = os.path.splitext(img_name)[0].lower()

            for mask_name in self.mask_names:
                if os.path.splitext(mask_name)[0].lower() == base_id:
                    image_mask_map.append((img_name, mask_name))
                    break
            else:
                print(f"No mask found for image: {img_name}")

        return image_mask_map

    def __len__(self):
        return len(self.image_mask_map)

    def __getitem__(self, idx):
        img_name, mask_name = self.image_mask_map[idx]

        image_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, mask_name)

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Apply transforms to image
        if self.transform:
            image = self.transform(image)

        # Resize mask to match image
        mask = T.Resize((224, 224))(mask)
        mask = T.ToTensor()(mask)
        mask = (mask > 0).float()  # Binary mask

        return image, mask


hrf_images_dir = "/root/.cache/kagglehub/datasets/ipythonx/retinal-vessel-segmentation/versions/1/HRF/images"
hrf_masks_dir = "/root/.cache/kagglehub/datasets/ipythonx/retinal-vessel-segmentation/versions/1/HRF/manual1"



# Transforms
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Creat DataLoader
hrf_dataset = HRFDataset(
    images_dir=hrf_images_dir,
    masks_dir=hrf_masks_dir,
    transform=transform
)

train_size = int(0.8 * len(hrf_dataset))
test_size = len(hrf_dataset) - train_size
train_dataset, test_dataset = random_split(hrf_dataset, [train_size, test_size])

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4)

print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of test samples: {len(test_dataset)}")
