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


def build_cnn_model():
    cnnModel = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=24, kernel_size=(3, 3), input_shape=(28, 28, 1), activation=tf.nn.relu),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(filters=36, kernel_size=(3, 3), activation=tf.nn.relu),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    return cnnModel


# Create CNN model
cnn_model = build_cnn_model()
print(cnn_model.summary())

cnn_model.compile(optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# Define the batch size and the number of epochs to use during training
batchSize = 64
epochs = 5

cnn_model.fit(train_images, train_labels, batch_size=batchSize, epochs=epochs)

test_loss, test_acc = cnn_model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
