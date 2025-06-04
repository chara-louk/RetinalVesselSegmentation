#Load the Unet model from hugging face
!git clone https://huggingface.co/yasinelh/retinal_vessel_U-Net

import torch
from models import unet

model = unet(3, 1)

model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

print("Model loaded successfully!")

#use it on DRIVE dataset
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt


def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.equalizeHist(image)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    image = cv2.resize(image, (512, 512))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    image = transform(image).unsqueeze(0)
    return image


def predict(image_path, model):
    model.eval()
    image_tensor = preprocess_image(image_path)

    with torch.no_grad():
        output = model(image_tensor)
        prediction = torch.sigmoid(output)
        prediction = (prediction >= 0.3).float()

    return prediction.squeeze().cpu().numpy()

def visualize_prediction(image_path, prediction):
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    original_image = cv2.resize(original_image, (512, 512))

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Predicted Vessel Mask")
    plt.imshow(prediction, cmap="gray")
    plt.axis("off")

    plt.show()




# Run prediction on a test image
test_image_name = "14_test.tif"
test_image_path = os.path.join(test_images_dir, test_image_name)

prediction_mask = predict(test_image_path, model)
visualize_prediction(test_image_path, prediction_mask)
