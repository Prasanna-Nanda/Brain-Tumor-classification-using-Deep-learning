# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 22:40:39 2025

@author: W10
"""

import tensorflow as tf
from tensorflow.keras.applications import VGG16, InceptionV3, ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

# Load and preprocess your dataset
def load_dataset(data_dir, img_size=(224, 224)):
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        image_size=img_size,
        batch_size=32,
        label_mode='int'
    )
    images = []
    labels = []
    for img_batch, label_batch in dataset:
        images.append(img_batch.numpy())
        labels.append(label_batch.numpy())
    X = np.concatenate(images, axis=0)
    y = np.concatenate(labels, axis=0)
    return X, y

# Set paths to your dataset
train_dir = "D:/me/project/files"
test_dir = "D:/me/project/files"

# Load train and test datasets
X_train, y_train = load_dataset(train_dir)
X_test, y_test = load_dataset(test_dir)

# Define function to extract features from a pre-trained model
def extract_features(model, X):
    features = model.predict(X, verbose=1)
    return features

# Load pre-trained models without the top classification layers
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
inception_v3 = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
resnet50 = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add global average pooling layers
vgg16_model = Model(inputs=vgg16.input, outputs=GlobalAveragePooling2D()(vgg16.output))
inception_v3_model = Model(inputs=inception_v3.input, outputs=GlobalAveragePooling2D()(inception_v3.output))
resnet50_model = Model(inputs=resnet50.input, outputs=GlobalAveragePooling2D()(resnet50.output))

# Freeze the layers of the models
for model in [vgg16_model, inception_v3_model, resnet50_model]:
    for layer in model.layers:
        layer.trainable = False

# Extract features from the models
X_train_vgg16 = extract_features(vgg16_model, X_train)
X_train_inception = extract_features(inception_v3_model, X_train)
X_train_resnet = extract_features(resnet50_model, X_train)

X_test_vgg16 = extract_features(vgg16_model, X_test)
X_test_inception = extract_features(inception_v3_model, X_test)
X_test_resnet = extract_features(resnet50_model, X_test)

# Concatenate features from all models
X_train_features = np.concatenate([X_train_vgg16, X_train_inception, X_train_resnet], axis=1)
X_test_features = np.concatenate([X_test_vgg16, X_test_inception, X_test_resnet], axis=1)

# Normalize features
scaler = StandardScaler()
X_train_features = scaler.fit_transform(X_train_features)
X_test_features = scaler.transform(X_test_features)

# Define a Deep Belief Network pipeline
rbm1 = BernoulliRBM(n_components=512, learning_rate=0.01, n_iter=10, random_state=42)
rbm2 = BernoulliRBM(n_components=256, learning_rate=0.01, n_iter=10, random_state=42)
classifier = LogisticRegression(max_iter=1000, random_state=42)

# Combine RBMs and Logistic Regression into a pipeline
pipeline = Pipeline(steps=[
    ('rbm1', rbm1),
    ('rbm2', rbm2),
    ('classifier', classifier)
])

# Train the pipeline
pipeline.fit(X_train_features, y_train)

# Evaluate the pipeline
accuracy = pipeline.score(X_test_features, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
