import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig
import os

#Hybrid CNN and ViT encoder

class CNNEncoder(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU()
        )

    def forward(self, x):
        s1 = self.layer1(x)
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        return s1, s2, s3


class ViTEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        if pretrained:
            self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k", output_attentions=True)
        else:
            config = ViTConfig(output_attentions=True)
            self.vit = ViTModel(config)

    def forward(self, x):
        outputs = self.vit(pixel_values=x)
        features = outputs.last_hidden_state[:, 1:, :] 
        attentions = outputs.attentions  
        return features, attentions


class UNetDecoder(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(input_dim, 256, kernel_size=2, stride=2), nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1), nn.ReLU()
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2), nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU()
        )

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2), nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU()
        )

        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x, skips):
        s1, s2, s3 = skips  

        x = x.permute(0, 2, 1).contiguous().view(-1, 768, 14, 14)

        x = self.up1(x) 
        s3 = F.interpolate(s3, size=(28, 28), mode='bilinear', align_corners=False)
        x = torch.cat([x, s3], dim=1)  
        x = self.conv1(x)

        x = self.up2(x) 
        s2 = F.interpolate(s2, size=(56, 56), mode='bilinear', align_corners=False)
        x = torch.cat([x, s2], dim=1) 
        x = self.conv2(x)

        x = self.up3(x) 
        s1 = F.interpolate(s1, size=(112, 112), mode='bilinear', align_corners=False)
        x = torch.cat([x, s1], dim=1) 
        x = self.conv3(x)

        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return torch.sigmoid(self.final(x))

# ----------Generator class---------------
class ViTUNetHybrid(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_encoder = CNNEncoder()
        self.vit_encoder = ViTEncoder()
        self.decoder = UNetDecoder()

    def forward(self, x, return_attn=False):
        skips = self.cnn_encoder(x)
        features, attentions = self.vit_encoder(x)
        mask = self.decoder(features, skips)
        if return_attn:
            return mask, attentions
        return mask

# ---------Discriminator class---------------
import torch.nn as nn
from torch.nn.utils import spectral_norm

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=4):  # 3 for image + 1 for mask
        super().__init__()
        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=1))
            # No Sigmoid here
        )

    def forward(self, x):
        return self.model(x)


# ------------------------ train the model ----------------------
def train_vit_gan(model_G, model_D, train_loader, optimizer_G, optimizer_D, epochs=20):
    adv_loss_fn = nn.BCEWithLogitsLoss()
    bce_loss_fn = nn.BCELoss()
    tversky_loss_fn = TverskyLoss(alpha=0.7, beta=0.3)

    model_G.cuda()
    model_D.cuda()

    for epoch in range(epochs):
        model_G.train()
        model_D.train()
        total_g_loss, total_d_loss = 0, 0

        for images, masks in train_loader:
            images = images.cuda()
            masks = masks.cuda()

            #Train Discriminator
            fake_masks = model_G(images).detach()
            real_input = torch.cat([images, masks], dim=1)
            fake_input = torch.cat([images, fake_masks], dim=1)

            d_real = model_D(real_input)
            d_fake = model_D(fake_input)

            real_labels = torch.full_like(d_real, 0.9).cuda()
            fake_labels = torch.full_like(d_fake, 0.1).cuda()

            d_loss = (adv_loss_fn(d_real, real_labels) + adv_loss_fn(d_fake, fake_labels)) / 2

            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()

            #Train Generator
            fake_masks = model_G(images)
            g_input = torch.cat([images, fake_masks], dim=1)
            d_output = model_D(g_input)

            adv_loss = adv_loss_fn(d_output, real_labels)

            seg_loss = 0.5 * bce_loss_fn(fake_masks, masks) + 0.5 * tversky_loss_fn(fake_masks, masks)

            g_loss = seg_loss + 0.05 * adv_loss

            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()

        print(f"Epoch {epoch+1}: G_loss = {total_g_loss / len(train_loader):.4f}, D_loss = {total_d_loss / len(train_loader):.4f}")

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        return 1 - tversky

root_dir = "/root/.cache/kagglehub/datasets/zionfuo/drive2004/versions/1/DRIVE"
train_images_dir = os.path.join(root_dir, "training/images")
train_masks_dir = os.path.join(root_dir, "training/1st_manual")

train_dataset = DRIVEVesselDataset(train_images_dir, train_masks_dir)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

model_G = ViTUNetHybrid()
model_D = PatchDiscriminator()

optimizer_G = torch.optim.Adam(model_G.parameters(), lr=2e-4, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(model_D.parameters(), lr=2e-4, betas=(0.5, 0.999))

train_vit_gan(model_G, model_D, train_loader, optimizer_G, optimizer_D, epochs=150)

#---------------------Evaluation metrics--------------------
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix
import torch.nn.functional as F

def evaluate_model(model, dataloader, threshold=0.5):
    model.eval()
    model.cuda()

    total_acc, total_spec, total_sens, total_iou = 0, 0, 0, 0
    total_samples = 0

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.cuda()
            masks = masks.cuda()

            preds = model(images)
            preds = (preds > threshold).float()

            for i in range(preds.size(0)):
                pred = preds[i].squeeze().cpu().numpy().flatten()
                true = masks[i].squeeze().cpu().numpy().flatten()

                tn, fp, fn, tp = confusion_matrix(true, pred, labels=[0, 1]).ravel()

                acc = (tp + tn) / (tp + tn + fp + fn)
                sens = tp / (tp + fn + 1e-8)
                spec = tn / (tn + fp + 1e-8)
                iou = tp / (tp + fp + fn + 1e-8)

                total_acc += acc
                total_sens += sens
                total_spec += spec
                total_iou += iou
                total_samples += 1

    return {
        "Accuracy": total_acc / total_samples,
        "Sensitivity": total_sens / total_samples,
        "Specificity": total_spec / total_samples,
        "IoU": total_iou / total_samples
    }

#----------------------------Visualization and EValuation-----------------
import matplotlib.pyplot as plt

def show_from_loader(model, loader, threshold=0.5, batch_idx=0):
    model.eval()
    for i, (images, masks) in enumerate(loader):
        if i != batch_idx:
            continue  

        images = images.cuda()
        preds = model(images).detach().cpu()

        for j in range(min(4, images.size(0))):  
            img = images[j].detach().cpu().permute(1, 2, 0).numpy()
            mask = masks[j].cpu().squeeze().numpy()
            pred = preds[j].squeeze().numpy()

            plt.figure(figsize=(15, 4))

            plt.subplot(1, 3, 1)
            plt.title("Input Image")
            plt.imshow(img)
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.title("Ground Truth")
            plt.imshow(mask, cmap='gray')
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.title("Prediction")
            plt.imshow(pred > threshold, cmap='gray')
            plt.axis('off')

            plt.tight_layout()
            plt.show()

        break  

show_from_loader(model_G, test_loader, threshold=0.4, batch_idx=0)


