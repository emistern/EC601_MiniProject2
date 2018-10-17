#This program will be used to learn TensorFlow, using the examples from https://github.com/tensorflow/workshops, "MNIST"

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# get the version of TensorFlow
# print(tf.__version__)

# Import MNIST Dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Create a list of all of the labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Get the size of the training set, both images and labels
print(train_images.shape)
print(len(train_labels))
# The training labels are ints between 0 and 9
print(train_labels)

# get the size of the test dataset
print(test_images.shape)
print(len(test_labels))

# Pre-process the images
# Set up the figures
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)

# Scale the images between 0 and 1
train_images = train_images / 255.0

test_images = test_images / 255.0

#Plot the first 25 images
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])

# Need to include this to show the plots
plt.show()


