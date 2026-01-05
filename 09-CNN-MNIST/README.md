# Convolutional Neural Network for MNIST Digit Classification

## Overview
This project implements a Convolutional Neural Network (CNN) using TensorFlow/Keras to classify handwritten digits from the MNIST dataset. The model achieves approximately 94% accuracy through a structured 7-layer architecture optimized for image recognition tasks.

## Technologies Used
- Python 3
- TensorFlow/Keras
- NumPy
- matplotlib

## Problem Statement
The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), each 28x28 pixels. The goal is to build a CNN that can accurately classify these digits, demonstrating fundamental computer vision and deep learning techniques.

## Model Architecture

### CNN Structure (7 Layers)
1. **Input Layer**: Shape [28, 28, 1] (grayscale images)
2. **Conv2D Layer 1**: 4 filters, 3x3 kernel, stride 1, ReLU activation
3. **MaxPool2D Layer 1**: 2x2 pool size, stride 2
4. **Conv2D Layer 2**: 4 filters, 3x3 kernel, stride 1, ReLU activation
5. **MaxPool2D Layer 2**: 2x2 pool size, stride 2
6. **Flatten Layer**: Converts 2D feature maps to 1D vector
7. **Dense Layer 1**: 64 units, ReLU activation
8. **Output Layer**: 10 units (one per digit), Softmax activation

### Training Configuration
- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy
- **Metrics**: Sparse Categorical Accuracy
- **Batch Size**: 2048
- **Epochs**: 15 (with early stopping)
- **Validation Split**: 20%
- **Early Stopping**: Monitors validation loss with patience of 3 epochs

## Data Preprocessing

### Preprocessing Steps
1. **Load Data**: MNIST dataset loaded via TensorFlow/Keras
2. **Reshape**: Add channel dimension (28, 28) → (28, 28, 1)
3. **Normalize**: Scale pixel values from [0, 255] to [0, 1]
4. **Train-Test Split**: 60,000 training images, 10,000 test images

## Key Features
- Convolutional layers for automatic feature extraction
- Max pooling for dimensionality reduction and translation invariance
- Early stopping to prevent overfitting
- Training history visualization
- Comprehensive performance metrics

## Usage
Run the script directly (no external dataset required - MNIST loads automatically):
```bash
python cnn_mnist.py
```

## Output
The script generates:
- Model architecture summary showing all layers and parameters
- Training progress with epoch-by-epoch metrics
- `accuracy_history.png`: Training and validation accuracy curves
- `loss_history.png`: Training and validation loss curves
- Comprehensive performance summary with test accuracy

## Model Performance
- **Training Accuracy**: ~94%
- **Validation Accuracy**: ~94%
- **Test Accuracy**: ~94%

The model demonstrates:
- Good convergence with minimal overfitting
- Effective feature extraction through convolutional layers
- Balanced complexity and computational efficiency
- Strong generalization to unseen data

## CNN Architecture Highlights

### Convolutional Blocks
- **Localized Receptive Fields**: 3x3 kernels capture local patterns
- **Feature Maps**: 4 filters per layer extract different features
- **ReLU Activation**: Introduces non-linearity for complex pattern learning

### Pooling Layers
- **Dimensionality Reduction**: Reduces spatial dimensions while preserving features
- **Translation Invariance**: Makes model robust to small shifts in digit position
- **Computational Efficiency**: Reduces parameters in subsequent layers

### Dense Layers
- **Feature Integration**: Combines extracted features for classification
- **Softmax Output**: Produces probability distribution over 10 digit classes

## Learning Outcomes
This project demonstrates:
- CNN architecture design for image classification
- Effective use of convolutional and pooling layers
- Data preprocessing for computer vision tasks
- Model training with early stopping
- Performance evaluation and visualization
- Deep learning framework proficiency (TensorFlow/Keras)

## Course Information
**Course**: CS577 - Principles & Techniques of Data Science  
**Institution**: San Diego State University  
**Author**: Aiden B Jajo
