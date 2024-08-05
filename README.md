# Optical-Based Self-Attention Dynamic Frame Detection (SADFFD) Network

## Introduction
Welcome to the repository for the Optical-Based Self-Attention Dynamic Frame Detection (SADFFD) Network. This project implements a novel deepfake detection system using a combination of optical flow and self-attention mechanisms. Our approach is designed to identify both spatial and temporal inconsistencies common in deepfake videos.

## Abstract
Deepfake videos manipulate faces in videos to create fake content, posing significant challenges. To address this, we developed the SADFFD Network. Our method utilizes optical flow to capture movement between video frames and self-attention mechanisms to detect inconsistencies. The system is tested on UADFV and Celeb-DF V1 datasets, showing high accuracy in identifying fake videos.

## Approach
Our method involves the following steps:
1. **Data Preprocessing**: 
    - Detect faces in video frames using Haar cascades.
    - Crop and resize faces to 224x224 pixels.
    - Convert frames to grayscale.
    - Calculate optical flow between frames.
    - Save optical flow fields as .npy files.
![fake_image_of](https://github.com/user-attachments/assets/5bac503e-9400-4000-bb7d-89ba9d438d16)
![real_image_of](https://github.com/user-attachments/assets/01830732-9f33-4fda-8992-d481e763a9fc)



2. **Model Architecture**:
    - **Upper Branch**: Extracts basic spatial features using convolution and MBConv blocks.
    - **Lower Branch**: Uses self-attention mechanisms to focus on crucial facial areas.
    - Concatenation of features from both branches.
    - Global Average Pooling (GAP) and classification using an activation function.

## Datasets
We evaluated our model on two datasets:
- **UADFV Dataset**: Includes 49 real and 49 fake videos created using FaceSwap.
- **Celeb-DF V1 Dataset**: Contains 158 real and 795 fake videos created using advanced GAN-based techniques.

## Results
Our model achieved the following results:
- **Celeb-DF V1 Dataset**:
    - Accuracy: 79.88%
    - Precision: 90%
    - Recall: 76%
- **UADFV Dataset**:
    - Accuracy: 85.26%
    - Precision: 95%
    - Recall: 76%

## Conclusion
The SADFFD Network is a robust tool for deepfake detection, capable of capturing temporal inconsistencies and spatial deviations. Future work includes refining the model, expanding the datasets, and integrating the system into real-time applications.
