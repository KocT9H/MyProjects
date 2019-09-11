# -*- coding: utf-8 -*-

"""
Example of Fully Connected Neural Network
Classify numbers according to MNIST data set

@author: Konstantin Verein
@Email:  koct9h@gmail.com

Based on MIT course:
© MIT 6.S191: Introduction to Deep Learning
introtodeeplearning.com

"""

import tensorflow as tf
import numpy as np
tf.enable_eager_execution()

# Get MNIST DB
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = np.expand_dims(train_images, axis=-1) / 255.
train_labels = np.int64(train_labels)
test_images = np.expand_dims(test_images, axis=-1) / 255.
test_labels = np.int64(test_labels)

# Create Fully Connected model
model = build_fc_model()
model.compile(optimizer=tf.train.GradientDescentOptimizer(learning_rate=1e-1),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


def build_fc_model():
    fc_model = tf.keras.Sequential([
        # Define a Flatten layer
        tf.keras.layers.Flatten(),
        # Define hidden layer of 128
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        # Define result
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    return fc_model


# Define the batch size and the number of epochs to use during training
batchSize = 64
epochs = 5

model.fit(train_images, train_labels, batch_size=batchSize, epochs=epochs)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
