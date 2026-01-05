# CNN for CIFAR-10 Image Classification

## Overview
This project implements an advanced Convolutional Neural Network (CNN) using TensorFlow/Keras to classify images from the CIFAR-10 dataset. The model achieves approximately 81% test accuracy through a deep architecture featuring batch normalization, dropout regularization, and multiple convolutional blocks.

## Technologies Used
- Python 3
- TensorFlow/Keras
- NumPy
- matplotlib
- scikit-learn

## Problem Statement
The CIFAR-10 dataset consists of 60,000 color images (32x32 pixels) across 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. This project builds a robust CNN that can accurately classify these diverse object categories, demonstrating advanced computer vision techniques.

## Model Architecture

### Deep CNN Structure
**Convolutional Block 1:**
- Conv2D: 32 filters, 3x3 kernel, ReLU, same padding
- Batch Normalization
- Conv2D: 32 filters, 3x3 kernel, ReLU, same padding
- Batch Normalization
- MaxPooling2D: 2x2 pool, stride 2

**Convolutional Block 2:**
- Conv2D: 64 filters, 3x3 kernel, ReLU, same padding
- Batch Normalization
- Conv2D: 64 filters, 3x3 kernel, ReLU, same padding
- Batch Normalization
- MaxPooling2D: 2x2 pool, stride 2

**Convolutional Block 3:**
- Conv2D: 128 filters, 3x3 kernel, ReLU, same padding
- Batch Normalization
- Conv2D: 128 filters, 3x3 kernel, ReLU, same padding
- Batch Normalization
- MaxPooling2D: 2x2 pool, stride 2

**Fully Connected Layers:**
- Flatten
- Dropout: 0.2 rate
- Dense: 128 units, ReLU activation
- Dense: 10 units, Softmax activation (output)

### Training Configuration
- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy
- **Metrics**: Sparse Categorical Accuracy
- **Batch Size**: 32
- **Epochs**: 25
- **Validation Split**: 20%

## Data Preprocessing
1. **Load Data**: CIFAR-10 dataset via TensorFlow/Keras
2. **Normalize**: Scale pixel values from [0, 255] to [0, 1]
3. **Flatten Labels**: Convert 2D label arrays to 1D format
4. **Dataset Split**: 50,000 training images, 10,000 test images

## Key Features

### Advanced Techniques
- **Batch Normalization**: Stabilizes and accelerates training
- **Dropout Regularization**: Reduces overfitting (20% dropout rate)
- **Progressive Filter Increase**: 32 → 64 → 128 filters for hierarchical feature learning
- **Multiple Conv Layers per Block**: Deeper feature extraction

### Model Capabilities
- Multi-class image classification (10 categories)
- Hierarchical feature learning
- Robust to overfitting through regularization
- Efficient training with batch normalization

## Usage
Run the script directly (CIFAR-10 loads automatically):
```bash
python cnn_cifar10.py
```

## Output
The script generates:
- Model architecture summary with layer details
- Training progress over 25 epochs
- `cifar10_accuracy_history.png`: Training and validation accuracy curves
- `cifar10_loss_history.png`: Training and validation loss curves
- Confusion matrix (10x10) for all classes
- Classification report with precision, recall, and F1-scores
- `cifar10_cnn_model.h5`: Saved trained model
- Comprehensive summary with findings and applications

## Model Performance

### Accuracy Metrics
- **Training Accuracy**: ~97.37%
- **Validation Accuracy**: ~81.48%
- **Test Accuracy**: ~81.38%

### Analysis
The model demonstrates:
- Strong feature extraction through deep convolutional architecture
- Good generalization to unseen data
- Slight overfitting (training accuracy higher than validation)
- Effective regularization through dropout and batch normalization
- Robust performance across diverse object categories

## Practical Applications

### Real-World Use Cases
1. **Autonomous Vehicles**: Identifying road objects and traffic signs
2. **Manufacturing**: Detecting defects in production pipelines
3. **Agriculture**: Diagnosing plant diseases from images
4. **Mobile Applications**: Photo sorting and automatic tagging
5. **Security Systems**: Object detection in surveillance footage

### Featured Application: Medical Imaging
**Disease Diagnosis System:**
- Analyzes X-rays, MRIs, and CT scans
- Identifies patterns indicating medical conditions
- Supports early detection and diagnosis
- Assists healthcare professionals with classification accuracy

## Deep Learning Highlights

### Convolutional Architecture
- **Hierarchical Learning**: Early layers detect edges, later layers detect complex patterns
- **Translation Invariance**: MaxPooling provides robustness to object position
- **Parameter Efficiency**: Shared weights across spatial dimensions

### Batch Normalization Benefits
- Reduces internal covariate shift
- Allows higher learning rates
- Provides regularization effect
- Accelerates convergence

### Regularization Strategy
- Dropout prevents co-adaptation of neurons
- Batch normalization adds noise during training
- Combined approach effectively reduces overfitting

## CIFAR-10 Classes
0. Airplane
1. Automobile
2. Bird
3. Cat
4. Deer
5. Dog
6. Frog
7. Horse
8. Ship
9. Truck

## Model Persistence
The trained model is saved as `cifar10_cnn_model.h5` and can be loaded for:
- Future predictions without retraining
- Transfer learning to similar tasks
- Deployment in production environments

## Course Information
**Course**: CS577 - Principles & Techniques of Data Science  
**Institution**: San Diego State University  
**Author**: Aiden B Jajo
