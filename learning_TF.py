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


