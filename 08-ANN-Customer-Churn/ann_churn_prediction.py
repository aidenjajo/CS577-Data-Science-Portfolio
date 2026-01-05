"""
Author: Aiden B Jajo
Institution: San Diego State University
Course: CS577 - Principles & Techniques of Data Science
Project: Artificial Neural Network for Customer Churn Prediction
"""

# Import required libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# Check TensorFlow version
print("TensorFlow version:", tf.__version__)

print("\n=== Part 1: Data Preprocessing ===")

# Step 1: Load the dataset
dataset = pd.read_excel('Churn_Modelling.xlsx', sheet_name='in')
print("\nDataset Shape:", dataset.shape)
print("\nFirst 5 rows of the dataset:")
print(dataset.head())

# Step 2: Prepare features and target variable
# Use columns [3:-1] as the X set (as instructed)
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

print("\nFeature set X shape:", X.shape)
print("Target set y shape:", y.shape)

# Step 3: Encode categorical variables
# Geography and Gender are categorical (columns 1 and 2 in X)
label_encoder_gender = LabelEncoder()
X[:, 2] = label_encoder_gender.fit_transform(X[:, 2])

# One-hot encode Geography
ct = ColumnTransformer(transformers=[
    ('encoder', OneHotEncoder(drop='first'), [1])  # drop='first' avoids dummy variable trap
], remainder='passthrough')
X = np.array(ct.fit_transform(X))

print("\nAfter encoding, the first row of X looks like:")
print(X[0])

# Step 4: Train-test split and feature scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print("\nAfter scaling, the first row of X_train looks like:")
print(X_train[0])

print("Data preprocessing completed.\n")

print("=== Part 2: Building the ANN ===")

# Step 5: Build a Keras Sequential model
ann = tf.keras.models.Sequential()

# Input layer and first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu', input_shape=(X_train.shape[1],)))

# Add additional hidden layers
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Step 6: Add output layer with sigmoid activation for binary classification
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Print model summary
print("\nModel Architecture:")
ann.summary()

print("\n=== Part 3: Training the ANN ===")

# Step 7: Compile the model
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 8: Set up early stopping to prevent overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Step 9: Train the model on the training set
history = ann.fit(X_train, y_train, batch_size=32, epochs=100, 
                  validation_split=0.2, callbacks=[early_stopping], verbose=1)

# Step 10: Visualize training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.close()

print("\nTraining completed. Training history saved as 'training_history.png'.\n")

print("=== Part 4: Making Predictions ===")

# Step 11: Predict if a specific customer will leave the bank
# Customer details: CreditScore=600, Geography='France', Gender='Male', Age=40, 
# Tenure=3, Balance=60000, NumOfProducts=2, HasCrCard=1, IsActiveMember=1, EstimatedSalary=50000
customer = np.array([[600, 'France', 'Male', 40, 3, 60000, 2, 1, 1, 50000]])

# Encode Gender
customer[0, 2] = label_encoder_gender.transform([customer[0, 2]])[0]

# One-hot encode Geography (same transformation as we applied to X)
customer_encoded = ct.transform(customer)

# Apply scaling
customer_scaled = sc.transform(customer_encoded)

# Make prediction
prediction = ann.predict(customer_scaled)
will_exit = prediction[0, 0] > 0.5

print("\nCustomer Prediction:")
print("Probability of leaving:", prediction[0, 0])
print("Will the customer leave?", "Yes" if will_exit else "No")

# Step 12: Predict the test set results
y_pred = (ann.predict(X_test) > 0.5).astype(int)
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)
print("\nAccuracy Score:", acc)

print("\n=== Part 5: Model Summary ===")

# Step 13: Display comprehensive summary of model architecture and results
print("\nModel Architecture and Test Results Summary:")
print("=" * 50)
print("1. Model Architecture:")
print("   - Input Layer: Features shape", X_train.shape[1])
print("   - Hidden Layer 1: 6 neurons with ReLU activation")
print("   - Hidden Layer 2: 6 neurons with ReLU activation")
print("   - Hidden Layer 3: 6 neurons with ReLU activation")
print("   - Output Layer: 1 neuron with Sigmoid activation")
print("2. Test Results:")
print("   - Accuracy Score:", acc)
print("   - Confusion Matrix:")
print("     True Negatives:", cm[0, 0])
print("     False Positives:", cm[0, 1])
print("     False Negatives:", cm[1, 0])
print("     True Positives:", cm[1, 1])
print("=" * 50)
