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

# # Get the size of the training set, both images and labels
# print(train_images.shape)
# print(len(train_labels))
# # The training labels are ints between 0 and 9
# print(train_labels)

# # get the size of the test dataset
# print(test_images.shape)
# print(len(test_labels))

# Pre-process the images
# Show the first image with a color bar
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
#plt.show()


# Begin building the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), #this first layer just 'unpacks' the 2d 28x28 array
    keras.layers.Dense(128, activation=tf.nn.relu), #deeply connected layer with 128 nodes
    keras.layers.Dense(10, activation=tf.nn.softmax) #deeply connected softmax layer, 10 nodes.
])

# Compile the model, this adds in the optimizer, loss function, and metrics
model.compile(optimizer=tf.train.AdamOptimizer(), # this is the algorithm used to find the best weights, backprop
              loss='sparse_categorical_crossentropy', # cross entropy loss, (for binary models)
              metrics=['accuracy']) # accuracy is the fraction of images correctly classified


# Train the model
model.fit(train_images, train_labels, epochs=5) 
# One epoch is the time step that is incremented every time it has went 
# through all the samples in the training set

# Evaluate the accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

# Since the test accuracy (.8775) is a bit less than the training accuracy (.8907), 
# we know we have overfitting.





