"""
Author: Aiden B Jajo
Institution: San Diego State University
Course: CS577 - Principles & Techniques of Data Science
Project: CNN for CIFAR-10 Image Classification
"""

# Import required libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

print("=== Part 1: Data Preprocessing ===")

# Step 1: Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
print(f"Training data shape: {x_train.shape}, Labels shape: {y_train.shape}")
print(f"Test data shape: {x_test.shape}, Labels shape: {y_test.shape}")

# Step 2: Normalize pixel values to [0, 1] range for improved training performance
x_train = x_train / 255.0
x_test = x_test / 255.0

# Step 3: Flatten label arrays to 1D format
y_train = y_train.flatten()
y_test = y_test.flatten()

print("Data preprocessing completed.\n")

print("=== Part 2: Building the CNN ===")

# Step 4: Initialize Sequential model
model = keras.Sequential()

# Step 5: Add input layer matching CIFAR-10 image shape (32x32x3)
model.add(layers.InputLayer(input_shape=(32, 32, 3)))

# Step 6: Add first convolutional block
# Conv2D layers with Batch Normalization and MaxPooling
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=2, strides=2, padding='valid'))

# Step 7: Add second convolutional block with increased filters
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=2, strides=2, padding='valid'))

# Step 8: Add third convolutional block for deeper feature extraction
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=2, strides=2, padding='valid'))

# Step 9: Add dense layers for classification
model.add(layers.Flatten())
model.add(layers.Dropout(0.2))  # Dropout to reduce overfitting
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))  # 10 classes for CIFAR-10

# Display model architecture
print("\nModel Architecture:")
model.summary()

print("\n=== Part 3: Training the CNN ===")

# Step 10: Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

# Step 11: Train the model
print("\nTraining the model...")
train_history = model.fit(x_train, y_train,
                         batch_size=32,
                         epochs=25,
                         validation_split=0.2,
                         verbose=1)

print("\nTraining completed.\n")

print("=== Part 4: Model Evaluation ===")

# Step 12: Plot training and validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(train_history.history['sparse_categorical_accuracy'], label='Train Acc')
plt.plot(train_history.history['val_sparse_categorical_accuracy'], label='Val Acc')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('cifar10_accuracy_history.png')
plt.close()

# Step 13: Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_history.history['loss'], label='Train Loss')
plt.plot(train_history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('cifar10_loss_history.png')
plt.close()

print("Training history plots saved as 'cifar10_accuracy_history.png' and 'cifar10_loss_history.png'.\n")

# Step 14: Evaluate model on test dataset
print("Evaluating on test set...")
evaluate_history = model.evaluate(x_test, y_test)

# Step 15: Generate predictions
y_pred = np.argmax(model.predict(x_test), axis=1)

# Step 16: Display confusion matrix and classification report
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\n=== Part 5: Summary and Applications ===")

# Step 17: Display comprehensive summary
print("\n" + "=" * 70)
print("MODEL SUMMARY AND FINDINGS")
print("=" * 70)

print("\nModel Architecture:")
print("  - 3 Convolutional blocks (Conv2D + BatchNorm + MaxPool)")
print("  - Each block increases feature depth: 32 → 64 → 128 filters")
print("  - Dropout layer (0.2) for regularization")
print("  - 2 Dense layers for classification")

print("\nModel Performance:")
print(f"  - Test Accuracy: {evaluate_history[1] * 100:.2f}%")

print("\nFindings and Insights:")
print("1. The CNN model consists of stacked Conv2D layers with Batch Normalization and MaxPooling.")
print("2. A Dropout layer was added before the Dense layers to reduce overfitting.")

train_acc = train_history.history['sparse_categorical_accuracy'][-1] * 100
val_acc = train_history.history['val_sparse_categorical_accuracy'][-1] * 100
print(f"3. The training accuracy reached {train_acc:.2f}%, while the validation accuracy stabilized at ~{val_acc:.2f}%.")
print("4. This indicates a slightly overfit model, but it still performs well on unseen data.")

test_acc = evaluate_history[1] * 100
print(f"5. The test accuracy was approximately {test_acc:.2f}%, confirming generalization capability.")

print("\nPractical Applications:")
print("This type of CNN-based image classification is useful for:")
print("  • Identifying road objects in autonomous vehicles")
print("  • Detecting defects in manufacturing pipelines")
print("  • Diagnosing plant diseases in agricultural fields")
print("  • Enhancing photo sorting/tagging in mobile apps")
print("  • Object detection in surveillance and security systems")

print("\nSpecific Application Example:")
print("Medical Imaging for Disease Diagnosis:")
print("  - Analysis of X-rays, MRIs, and CT scans to detect medical conditions")
print("  - Identifies patterns in medical images to accurately classify patient conditions")
print("  - Enables early detection and diagnosis support for healthcare professionals")

print("=" * 70)

# Step 18: Save the trained model
model.save('cifar10_cnn_model.h5')
print("\nModel saved as 'cifar10_cnn_model.h5'")
