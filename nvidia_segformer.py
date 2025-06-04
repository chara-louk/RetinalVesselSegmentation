from transformers import SegformerForSemanticSegmentation, AutoImageProcessor

processor = AutoImageProcessor.from_pretrained("nvidia/mit-b0")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0", num_labels=1)

import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        smooth = 1e-6
        intersection = (inputs * targets).sum(dim=(2, 3))
        dice = (2. * intersection + smooth) / (inputs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + smooth)
        dice_loss = 1 - dice.mean()
        bce_loss = self.bce(inputs, targets)
        return bce_loss + dice_loss

criterion = BCEDiceLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)


best_val_dice = 0.0

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        outputs = model(pixel_values=images).logits
        outputs = nn.functional.interpolate(outputs, size=(224, 224), mode='bilinear')

        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Validation
    model.eval()
    val_dice, val_iou, val_acc = 0, 0, 0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(pixel_values=images).logits
            outputs = nn.functional.interpolate(outputs, size=(224, 224), mode='bilinear')
            dice, iou, acc = compute_metrics(outputs, masks)
            val_dice += dice
            val_iou += iou
            val_acc += acc

    val_dice /= len(val_loader)
    val_iou /= len(val_loader)
    val_acc /= len(val_loader)

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}, "
          f"Val Dice: {val_dice:.4f}, IoU: {val_iou:.4f}, Acc: {val_acc:.4f}")

    # Save model if improved
    if val_dice > best_val_dice:
        best_val_dice = val_dice
        torch.save(model.state_dict(), "best_model.pth")

#Visualization
import matplotlib.pyplot as plt
import torch.nn.functional as F

model.eval()
with torch.no_grad():
    image, mask = train_dataset[0]
    image = image.unsqueeze(0).to(device)
    output = model(pixel_values=image).logits
    pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Predicted Vessel Mask")
plt.imshow(pred_mask > 0.5, cmap="gray")
plt.subplot(1, 2, 2)
plt.title("Ground Truth")
plt.imshow(mask.squeeze(), cmap="gray")
plt.show()

