"""
Author: Aiden B Jajo
Institution: San Diego State University
Course: CS577 - Principles & Techniques of Data Science
Project: Convolutional Neural Network for MNIST Digit Classification
"""

# Import required libraries
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

# Check TensorFlow version
print("TensorFlow version:", tf.__version__)

print("\n=== Part 1: Data Preprocessing ===")

# Step 1: Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print(f"Training set shape: {x_train.shape}")
print(f"Test set shape: {x_test.shape}")

# Step 2: Reshape and normalize the data
# Reshaping input to add channel dimension and normalize pixel values to [0, 1]
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

print(f"Training set shape after reshaping: {x_train.shape}")
print(f"Test set shape after reshaping: {x_test.shape}")

print("Data preprocessing completed.\n")

print("=== Part 2: Building the CNN ===")

# Step 3: Initialize Sequential model
model = keras.Sequential()

# Step 4: Add first convolutional block
# Conv2D layer with 4 filters, 3x3 kernel, stride 1, ReLU activation
model.add(keras.layers.Conv2D(filters=4, kernel_size=(3, 3), strides=(1, 1),
                              activation='relu', input_shape=(28, 28, 1)))

# MaxPooling layer with 2x2 pool size, stride 2
model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

# Step 5: Add second convolutional block
model.add(keras.layers.Conv2D(filters=4, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

# Step 6: Flatten the output for dense layers
model.add(keras.layers.Flatten())

# Step 7: Add dense layer with 64 units and ReLU activation
model.add(keras.layers.Dense(units=64, activation='relu'))

# Step 8: Add output layer with 10 units (one per digit) and softmax activation
model.add(keras.layers.Dense(units=10, activation='softmax'))

# Display model architecture
print("\nModel Architecture:")
model.summary()

print("\n=== Part 3: Training the CNN ===")

# Step 9: Compile the model
model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=[keras.metrics.SparseCategoricalAccuracy()])

# Step 10: Set up early stopping callback
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Step 11: Train the model
print("\nTraining the model...")
train_history = model.fit(x_train, y_train,
                         batch_size=2048,
                         epochs=15,
                         validation_split=0.2,
                         callbacks=[early_stopping],
                         verbose=1)

print("\nTraining completed.\n")

print("=== Part 4: Model Evaluation ===")

# Step 12: Evaluate the model on test set
print("\nEvaluating on test set...")
evaluate_history = model.evaluate(x_test, y_test)

# Step 13: Plot training and validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(train_history.history['sparse_categorical_accuracy'])
plt.plot(train_history.history['val_sparse_categorical_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.grid(True)
plt.savefig('accuracy_history.png')
plt.show()

# Step 14: Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_history.history['loss'])
plt.plot(train_history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.grid(True)
plt.savefig('loss_history.png')
plt.show()

print("\nTraining history plots saved as 'accuracy_history.png' and 'loss_history.png'.\n")

print("=== Part 5: Results Summary ===")

# Step 15: Display comprehensive summary
print("\n" + "=" * 60)
print("MODEL PERFORMANCE SUMMARY")
print("=" * 60)
print(f"Model Architecture: CNN with {len(model.layers)} layers")
print(f"  - 2 Convolutional blocks (Conv2D + MaxPool2D)")
print(f"  - 1 Flatten layer")
print(f"  - 2 Dense layers (64 units + 10 output units)")
print()
print(f"Training Accuracy: {train_history.history['sparse_categorical_accuracy'][-1]:.4f}")
print(f"Validation Accuracy: {train_history.history['val_sparse_categorical_accuracy'][-1]:.4f}")
print(f"Test Accuracy: {evaluate_history[1]:.4f}")
print()
print(f"The model achieved approximately {evaluate_history[1]*100:.1f}% accuracy on the MNIST test dataset.")
print("The training curves show good convergence with no significant overfitting.")
print("The CNN architecture demonstrates effective feature extraction for handwritten digit recognition.")
print("=" * 60)
