# CNN Image Classification Project

## Project Overview
This project is a deep learning image classification project using a **Convolutional Neural Network (CNN)**. The main goal is to classify images from a selected dataset and practice preprocessing, model training, evaluation, and hyperparameter tuning.

## Dataset
We used the **Dogs vs. Cats** dataset from Kaggle. The dataset has:
- **Training images:** 25,000
- **Validation images:** 5,000
- **Classes:** Dogs and Cats

## Data Preprocessing
- Images were resized to **128x128 pixels**.
- Images were **normalized** (scaled to 0-1 range).
- Data augmentation was applied to improve model performance:
  - Rotation
  - Horizontal flip
  - Zoom
  - Brightness changes
- Train-validation split was applied to prepare the model for evaluation.

## Model Architecture
The CNN model includes:
- **Convolutional layers** with ReLU activation
- **MaxPooling layers**
- **Dropout layers** to prevent overfitting
- **Dense (fully connected) layers**  
  - Dense layer size: **128 units**
- **Output layer** with sigmoid activation (binary classification)

## Training
- Hyperparameters were optimized for:
  - Dropout rate
  - Dense layer size
  - Learning rate
  - Batch size
  - Optimizer
- Early stopping and learning rate reduction were used to avoid overfitting.
- Training was done in **short epochs** for quick testing, then extended for best configuration.

## Results
- Best validation accuracy achieved: **96.56%**
- The model showed good fit without severe overfitting.
- Hyperparameter optimization helped identify the most effective configuration.

## Recommendations
- Use the best configuration (higher dropout, dense 128 units) for final training.
- Consider using **more epochs** and early stopping for final model.
- Data augmentation and regularization improve performance.

## Kaggle Notebook
[[Link to Kaggle Notebook](#)
](https://www.kaggle.com/code/taylantakn/dogs-vs-cats-classification-project?scriptVersionId=264017772)
