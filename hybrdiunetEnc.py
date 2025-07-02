import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig


# --- CNN Encoder for skip connections ---
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
        s1 = self.layer1(x)  # (B,64,224,224)
        s2 = self.layer2(s1) # (B,128,112,112)
        s3 = self.layer3(s2) # (B,256,56,56)
        return s1, s2, s3


# --- ViT Encoder (unchanged) ---
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
        features = outputs.last_hidden_state[:, 1:, :]  # remove CLS token
        attentions = outputs.attentions  # list of attention matrices from each layer
        return features, attentions


# --- Hybrid Decoder with skip connections ---
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
        s1, s2, s3 = skips  # Shapes: s1 (B,64,224,224), s2 (B,128,112,112), s3 (B,256,56,56)

        x = x.permute(0, 2, 1).contiguous().view(-1, 768, 14, 14)

        x = self.up1(x)  # (B, 256, 28, 28)
        s3 = F.interpolate(s3, size=(28, 28), mode='bilinear', align_corners=False)
        x = torch.cat([x, s3], dim=1)  # (B, 512, 28, 28)
        x = self.conv1(x)

        x = self.up2(x)  # (B, 128, 56, 56)
        s2 = F.interpolate(s2, size=(56, 56), mode='bilinear', align_corners=False)
        x = torch.cat([x, s2], dim=1)  # (B, 256, 56, 56)
        x = self.conv2(x)

        x = self.up3(x)  # (B, 64, 112, 112)
        s1 = F.interpolate(s1, size=(112, 112), mode='bilinear', align_corners=False)
        x = torch.cat([x, s1], dim=1)  # (B, 128, 112, 112)
        x = self.conv3(x)

        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return torch.sigmoid(self.final(x))


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
