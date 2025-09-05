import kagglehub

# Download latest version
path = kagglehub.dataset_download("rashasarhanalharthi/chase-db1")

print("Path to dataset files:", path)

import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch

class CHASEDB1Dataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

        self.image_names = sorted([f for f in os.listdir(images_dir) if f.endswith(".tif")])
        self.mask_names = sorted([f for f in os.listdir(masks_dir) if f.endswith(".tif")])

        self.image_mask_map = self.create_image_mask_mapping()

    def __len__(self):
        return len(self.image_mask_map)

    def create_image_mask_mapping(self):
        image_mask_map = []
        for img_name in self.image_names:
            img_id = img_name.split('_')[0]

            matched_mask = None
            for mask_name in self.mask_names:
                print(f"Checking mask: {mask_name}")

                if img_id == mask_name.split('_')[0]:
                    print(f"Match found! Image: {img_name}, Mask: {mask_name}")
                    matched_mask = mask_name
                    break

            if matched_mask:
                image_mask_map.append((img_name, matched_mask))
            else:
                print(f"No match found for image: {img_name}")

        return image_mask_map

    def __getitem__(self, idx):
      img_name, mask_name = self.image_mask_map[idx]

      image_path = os.path.join(self.images_dir, img_name)
      mask_path = os.path.join(self.masks_dir, mask_name)

      image = Image.open(image_path).convert("RGB")
      mask = Image.open(mask_path).convert("L")

      if self.transform:
          image = self.transform(image)
          mask = T.Resize((224, 224))(mask)   # Resize mask to match model output
          mask = T.ToTensor()(mask)
      else:
          image = T.ToTensor()(image)
          mask = T.Resize((224, 224))(mask)
          mask = T.ToTensor()(mask)

      mask = (mask > 0).float()  # Convert to binary mask

      return image, mask



transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Directories
train_images_dir = "/root/.cache/kagglehub/datasets/rashasarhanalharthi/chase-db1/versions/1/new/chase/training/training/images"
val_images_dir = "/root/.cache/kagglehub/datasets/rashasarhanalharthi/chase-db1/versions/1/new/chase/test/test/images"
train_masks_dir = "/root/.cache/kagglehub/datasets/rashasarhanalharthi/chase-db1/versions/1/new/chase/training/training/1st_manual"
val_masks_dir = "/root/.cache/kagglehub/datasets/rashasarhanalharthi/chase-db1/versions/1/new/chase/test/test/1st_manual"

train_dataset = CHASEDB1Dataset(
    images_dir=train_images_dir,
    masks_dir=train_masks_dir,
    transform=transform
)

val_dataset = CHASEDB1Dataset(
    images_dir=val_images_dir,
    masks_dir=val_masks_dir,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)

print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(val_dataset)}")
