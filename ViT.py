from transformers import ViTModel, ViTConfig,AutoImageProcessor
import torch.nn as nn
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")


class ViTEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        if pretrained:
            self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        else:
            config = ViTConfig()
            self.vit = ViTModel(config)

    def forward(self, x):
      outputs = self.vit(pixel_values=x)
      return outputs.last_hidden_state[:, 1:, :]

#----------------Decoder------------------
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, input_dim=768, output_channels=1):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(input_dim, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, output_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, N, D = x.shape
        H = W = int(N ** 0.5)
        assert H * W == N, f"Cannot reshape: N={N} is not a square"
        x = x.permute(0, 2, 1).contiguous().view(B, D, H, W)
        x = self.decoder(x) 
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)  
        return x


#------------------ViT setup--------------
class ViTUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ViTEncoder(pretrained=True)
        self.decoder = Decoder(input_dim=768)

    def forward(self, x):
        features = self.encoder(x)
        out = self.decoder(features)
        return out


#------------Evaluation Metrics----------
import torch

def evaluate_model(model, dataloader, device="cuda"):
    model.eval()
    TP = TN = FP = FN = 0

    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            preds = (preds > 0.5).float()
            masks = (masks > 0.5).float()

            TP += (preds * masks).sum().item()
            TN += ((1 - preds) * (1 - masks)).sum().item()
            FP += (preds * (1 - masks)).sum().item()
            FN += ((1 - preds) * masks).sum().item()

    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)
    sensitivity = TP / (TP + FN + 1e-8)   
    specificity = TN / (TN + FP + 1e-8)
    iou = TP / (TP + FP + FN + 1e-8)

    return {
        "Accuracy": accuracy,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "IoU": iou,
    }


#---------------Training--------------
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ViTUNet().to(device)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


start_time = time.time()  

for epoch in range(150):
    epoch_start = time.time()  

    model.train()
    total_loss = 0
    for images, masks in train_loader:
        images, masks = images.cuda(), masks.cuda()

        preds = model(images)
        loss = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    metrics = evaluate_model(model, test_loader)

    epoch_end = time.time() 
    epoch_duration = epoch_end - epoch_start

    print(f"Epoch {epoch}: "
          f"Loss = {avg_loss:.4f} | "
          f"Acc: {metrics['Accuracy']:.4f}, "
          f"Sens: {metrics['Sensitivity']:.4f}, "
          f"Spec: {metrics['Specificity']:.4f}, "
          f"IoU: {metrics['IoU']:.4f} | "
          f"Time: {epoch_duration:.2f} sec")

# total training time
end_time = time.time()
print(f"Total training time: {(end_time - start_time) / 60:.2f} minutes")


#---------------Visualization-------------------
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

visualize_predictions(model, test_loader)

