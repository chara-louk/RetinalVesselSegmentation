from transformers import AutoImageProcessor, AutoModelForImageClassification

processor = AutoImageProcessor.from_pretrained("facebook/levit-256")
model = AutoModelForImageClassification.from_pretrained("facebook/levit-256")

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
from torchvision import transforms

#--------------LeViT set up---------------------------

class LevitSegmentationModel(nn.Module):
    def __init__(self, model_name="facebook/levit-256", out_channels=1):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)

        hidden_size = self.encoder.config.hidden_sizes[-1]  

        # decoder with 3 upsampling layers
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_size, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape

        x_cpu = x.cpu()
        image_list = [transforms.ToPILImage()(x_cpu[i]) for i in range(B)]
        inputs = self.processor(images=image_list, return_tensors="pt").to(x.device)

        outputs = self.encoder(**inputs).last_hidden_state 

        N = outputs.shape[1]  
        side = int(N ** 0.5) 
        x = outputs.permute(0, 2, 1).contiguous().view(B, -1, side, side)  

        out = self.decoder(x)  

        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)

        return out


#----------------Training--------------------
from torch.optim import Adam
from tqdm import tqdm

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for images, masks in tqdm(loader, desc="Training"):
        images = images.to(device)
        masks = masks.to(device)

        preds = model(images)
        loss = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0

    total_acc, total_sens, total_spec, total_iou = 0, 0, 0, 0
    n_batches = 0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            preds = model(images)
            loss = criterion(preds, masks)

            val_loss += loss.item()

            # Compute metrics
            acc, sens, spec, iou = compute_metrics(preds, masks)
            total_acc += acc
            total_sens += sens
            total_spec += spec
            total_iou += iou
            n_batches += 1

    avg_loss = val_loss / n_batches
    avg_acc = total_acc / n_batches
    avg_sens = total_sens / n_batches
    avg_spec = total_spec / n_batches
    avg_iou = total_iou / n_batches

    return avg_loss, avg_acc, avg_sens, avg_spec, avg_iou

import torch
from torch import nn
from torch.utils.data import DataLoader
import time
from torch.optim import Adam



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LevitSegmentationModel().to(device)

optimizer = Adam(model.parameters(), lr=1e-4)
criterion = nn.BCELoss() 

num_epochs = 150

start_time = time.time()  

for epoch in range(num_epochs):
    epoch_start = time.time()  

    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc, val_sens, val_spec, val_iou = validate(model, test_loader, criterion, device)

    epoch_end = time.time()
    epoch_duration = epoch_end - epoch_start

    print(f"Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"Acc: {val_acc:.4f} | "
          f"Sens: {val_sens:.4f} | "
          f"Spec: {val_spec:.4f} | "
          f"IoU: {val_iou:.4f} | "
          f"Time: {epoch_duration:.2f} sec")

# total training time
end_time = time.time()
total_duration = end_time - start_time
print(f"\nTotal training time: {total_duration / 60:.2f} minutes")


#----------------------Visualization----------------
import matplotlib.pyplot as plt
import torch

def unnormalize(img):
    mean = torch.tensor([0.5, 0.5, 0.5]).view(-1, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(-1, 1, 1)
    img = img * std + mean 
    img = torch.clamp(img, 0, 1)  
    return img

def visualize_predictions(model, dataloader):
    model.eval()
    images, masks = next(iter(dataloader))
    images, masks = images.cpu(), masks.cpu()

    with torch.no_grad():
        preds = model(images.cuda()).cpu()

    for i in range(min(4, len(images))):
        plt.figure(figsize=(12, 3))

        plt.subplot(1, 3, 1)
        img = images[i]
        if img.shape[0] == 3:
            img = unnormalize(img)
            img = img.permute(1, 2, 0).numpy()
            plt.imshow(img)
        else:
            plt.imshow(img[0].numpy(), cmap='gray')
        plt.title("Input")

        plt.subplot(1, 3, 2)
        plt.imshow(masks[i][0].numpy(), cmap="gray")
        plt.title("Ground Truth")

        plt.subplot(1, 3, 3)
        plt.imshow(preds[i][0].numpy(), cmap="gray")
        plt.title("Prediction")

        plt.tight_layout()
        plt.show()

visualize_predictions(model, val_loader)


