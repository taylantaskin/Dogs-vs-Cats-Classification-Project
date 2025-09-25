# 🚀 Part 1: Project Setup and Data Preparation

---

This section summarizes the environment setup, the dataset details, and the steps taken for detailed image preprocessing before training the model.

### 1. Environment and Setup

The project was executed in a high-performance environment to ensure fast training:

* **Deep Learning Framework:** TensorFlow version **2.18.0**
* **Hardware:** A **GPU** (`/physical_device:GPU:0`) was detected and utilized for accelerated training.

### 2. Dataset Overview

The project uses the **Kaggle Dogs vs Cats dataset**. We verified the files and checked the class distribution:

* **Total Training Images:** **25,000**
* **Test Images:** **12,500**
* **Class Balance:** The training set is perfectly balanced: **12,500 Dogs (50.0%)** and **12,500 Cats (50.0%)**.

### 3. Data Preprocessing Details

The images were processed to prepare them for the Convolutional Neural Network (CNN):

* **Resizing:** All images were resized to a fixed size of **128x128** pixels.
* **Batch Size:** Images are fed to the model in batches of **32**.
* **Data Split:** The 25,000 training images were split into three sets:
    * **Training Set:** **15,000** images
    * **Validation Set:** **5,000** images
    * **Test Set:** **5,000** images
* **Data Augmentation:** To make the model more robust and prevent overfitting, random transformations (including **rotation, shifting, zooming, and flipping**) were applied **only to the training set**.

---
