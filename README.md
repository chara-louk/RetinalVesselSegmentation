# Automated Retina Vessel Segmentation

## Introduction
In recent years, the prevalence of ophthalmological diseases such as diabetic retinopathy, glaucoma, and hypertensive retinopathy has increased significantly. Early detection is critical, as delayed diagnosis can lead to irreversible vision loss, highlighting the importance of effective clinical tools. Machine learning and deep learning techniques now enable clinicians to identify diseases at an early stage, reducing the risk of progression and improving patient outcomes. This paper analyzes deep learning architecture for automated vessel segmentation. Models based on Convolutional Neural Networks (CNNs) and Transformers are developed and compared for generating vascular masks from retinal fundus images. Furthermore, a novel hybrid model that combines CNNs and Vision Transformers (ViTs) within a Generative Adversarial Network (GAN) framework is proposed. GANs leverage a generator–discriminator competition to improve prediction quality. In this work, the generator combines CNN and ViT encoders with a U-Net decoder to produce vessel masks, while a PatchGAN discriminator ensures their realism. Training is guided by a combination of segmentation losses and adversarial loss. The proposed model achieves strong performance on three public benchmark datasets: DRIVE, CHASE_DB1, and HRF.
Beyond segmentation accuracy, this work emphasizes the interpretability and transparency of deep learning models in medical applications. To this end, Explainable Artificial Intelligence (XAI) techniques are incorporated. Specifically, Grad-CAM and Attention Rollout are used to generate heatmaps highlighting the regions most influential to model predictions, while LIME and SHAP provide further insight into the decision-making process.
Overall, the proposed approaches aim to advance automated retinal vessel segmentation by combining high performance with interpretability, contributing to the development of AI systems that are both effective and clinically trustworthy.

## Methods

### Data
Experiments were conducted on three publicly available retinal vessel segmentation datasets:
- **DRIVE**: 40 color fundus images 
- **CHASE_DB1**: 28 high-resolution retinal images
- **HRF**: 45 images with varying resolutions and pathologies

### Baseline Models
State-of-the-art models for retinal vessel segmentation were cloned from GitHub and Hugging Face for the experimental study. 
- **U-Net** [Hugging Face](yasinelh/retinal_vessel_U-Net)
- **SAU-Net** [Hugging Face](yasinelh/retinal_vessel_U-Net)
- **BDCU-Net** [GitHub](https://github.com/rezazad68/BCDU-Net)

For the study of Transformer based models these Hugging Face models were used and trained:
- **ViT** (google/vit-base-patch16-224-in21k)
- **SegFormer** (nvidia/segformer-b0-finetuned-ade-512-512)
- **LeViT** (facebook/levit-256)
- **ConViT** (timm/convit_base.fb_in1k)

These models were used as backbone to export the feature map of the images, and a simple decoder was created to create the final prediction. The decoder was the same for all the models, and all the models were trained for 150 epochs.

### Proposed GAN-based model
The proposed architecture integrates **Convolutional Neural Networks (CNNs)** and **Vision Transformers (ViTs)** within a **Generative Adversarial Network (GAN)** framework to achieve accurate and realistic retinal vessel segmentation.

A GAN consists of two components:
- **Generator (G):** Produces synthetic data (here, segmentation masks).
- **Discriminator (D):** Evaluates whether the input is real (ground truth) or generated (fake).

These two networks engage in an **adversarial learning process**:  
- The Generator tries to “fool” the Discriminator by generating realistic masks.  
- The Discriminator improves its ability to detect fakes.  
Through this competition, both models improve over time.

In this work, the **classic GAN framework** was adapted for **image segmentation** rather than data synthesis.  
Instead of random noise, the Generator receives a retinal fundus image as input and outputs a vessel mask. The Discriminator (a PatchGAN) evaluates the authenticity of these generated masks.

#### Generator Architecture
The **Generator** is implemented as a **hybrid ViT-UNet**:
- **CNN Encoder:** Extracts fine-grained local features and vessel details.  
- **ViT Encoder:** Captures long-range dependencies and global context. The Hugging Face model "google/vit-base-patch16-224-in21k" was used for the backbone of the ViT encoder.
- **U-Net Decoder:** Uses skip connections to combine hierarchical CNN features with ViT representations, reconstructing high-resolution segmentation masks.

#### Discriminator Architecture (PatchGAN)
The **Discriminator** follows the **PatchGAN** approach:
- Evaluates local 70×70 patches instead of the whole image.  
- Takes the fundus image and mask pair as input.  
- Outputs a probability map of “real” vs “fake” for each patch.

#### Training 
The model was trained for 150 epochs with batch size 4. A combination of BCE and Tversky loss functions were used, giving focus on the FN of the prediction, boosting the detiction of fine vessels.
- **Optimizer:** Adam (lr = 1×10⁻⁴)  
- **Batch size:** 4  
- **Epochs:** 150
- **Hardware:** Google Colab environment (NVIDIA Tesla T4 GPU)
- **Loss weighting:** Real labels smoothed to 0.9, fake to 0.1 (label smoothing)  
- **Losses combined:** Adversarial + Segmentation (BCE + Tversky)

### Explainable AI (XAI)

To enhance the interpretability and transparency of the proposed models, **XAI techniques** were applied to visualize and understand the regions of the retinal images most influential to the predictions. The following methods were used:

- **Grad-CAM (Gradient-weighted Class Activation Mapping):** Highlights regions in the feature maps that contribute most to the prediction. This helps identify which parts of the retina influenced vessel segmentation.  
- **Attention Rollout:** Aggregates attention scores across Transformer layers to visualize the contribution of different image regions to the model’s decision.  
- **LIME (Local Interpretable Model-agnostic Explanations):** Generates pixel-level explanations by perturbing input images and observing changes in predictions. The Python's library lime was used to implement the technique. 
- **SHAP (SHapley Additive exPlanations):** Quantifies the contribution of each pixel to the final prediction, providing global and local interpretability.  The Python's library shap was used to implement the technique.

