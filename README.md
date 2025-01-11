
# MRI Style Transfer with CycleGAN

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) &nbsp; 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbhishekSinghDhadwal/Generating-T2-MRI-from-T1-Scans-Using-CycleGAN/blob/main/MRI_T1_to_T2_Style_Transfer.ipynb)

CycleGAN-based model for generating T2-weighted MRI images from T1 scans to enhance diagnostic accuracy with unpaired datasets. Augments pre-existing dataset for enhancing training quality. 

## Overview

Generating MRI images with varying contrasts (T1 and T2) can be expensive. Misdiagnosis due to insufficient imaging data is a common issue. By using deep learning (CycleGAN), this project generates artificial MRI images of different contrast levels from existing scans. These additional images aim to assist radiologists in making more accurate diagnoses while reducing the need for expensive imaging procedures.

## Problem Statement

Medical imaging diagnostics, particularly MRI scans, require expert interpretation. However:
- Misdiagnosis and disagreements between radiologists are common.
- Access to diverse contrast scans (T1 and T2) is essential but can be expensive and time-intensive.

**Objective:** To develop a Generative Adversarial Network (GAN) capable of generating synthetic T2-weighted MRI images from T1-weighted MRI images and vice versa using an unpaired dataset. This can provide radiologists with enhanced imaging options at minimal cost.

---

## Key Features

- **Unpaired Data Handling**: Implements a CycleGAN to train on unpaired MRI datasets.
- **Image Augmentation**: Includes augmentation techniques to enhance the dataset.
- **Modified U-Net Architecture**: For better image generation.
- **Loss Functions**:
  - Discriminator loss.
  - Generator loss.
  - Cycle consistency loss.
  - Identity loss.
- **Visualization**: Outputs synthetic MRI images and a GIF animation showcasing the transformation process.

---

## Notebook Pipeline

1. **Data Understanding**: Load and explore MRI datasets (T1 and T2 domains).
2. **Data Preprocessing**:
   - Reshape, augment, and normalize images.
   - Prepare datasets for training with batching and shuffling.
3. **Model Development**:
   - Build U-Net-based generators for T1→T2 and T2→T1 transformations.
   - Design discriminators to distinguish between real and generated images.
4. **Model Training**:
   - Train the generators and discriminators using adversarial, cycle consistency, and identity losses.
   - Save models periodically for reproducibility.
5. **Visualization**:
   - Generate side-by-side comparisons of input and transformed images.
   - Create an animated GIF for transformation visualization.

---

## Dataset

The dataset contains unpaired T1-weighted and T2-weighted MRI images:
- **T1 Images**: 43 samples.
- **T2 Images**: 46 samples.

Augmentation techniques expand the datasets to improve training quality.

---

## How to run the Notebook

The notebook should work in both Colab and Jupyter. Separate commented out instructions are available for each mode in the notebook. Modify the notebook to match locations of the dataset RAR file.

---

## Results

- Successfully generated T2-weighted images from T1-weighted inputs and vice versa.
- Achieved cycle consistency and ensured minimal distortion between input and output images.
- The animated GIF demonstrates the transformation process over multiple epochs.

---

## Dependencies

The project requires the following Python libraries:

- TensorFlow (for building and training the neural networks)
- TensorFlow Addons (for instance normalization)
- Matplotlib (for visualization)
- scikit-image (for image processing)
- Pillow (for image handling)
- tqdm (for progress visualization during training)
- imageio (for creating animations)
- GitHub TensorFlow Docs (for embedding visualizations in notebooks)