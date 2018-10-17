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

# Make predictions
# The model predicts labels for the test_images
predictions = model.predict(test_images)
# This gives us the probability that the image is each of the 9 items
print(predictions[0])
# The highest prediction is the one we predict the image to be
print(np.argmax(predictions[0]))
# Check if the first label of test_images is the same as the highest prediction
print(test_labels[0])

# Plotting functions
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
 
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

# # Check out the 0th image predictions
# i = 0
# plt.figure(figsize=(6,3))
# plt.subplot(1,2,1)
# plot_image(i, predictions, test_labels, test_images)
# plt.subplot(1,2,2)
# plot_value_array(i, predictions,  test_labels)


# # CHeck out the 12th image, which happens to fail...
# i = 12
# plt.figure(figsize=(6,3))
# plt.subplot(1,2,1)
# plot_image(i, predictions, test_labels, test_images)
# plt.subplot(1,2,2)
# plot_value_array(i, predictions,  test_labels)

#Plot many images on 1 figure!
# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)

# Need to figure out how to move the images up on the figure so i can see them all

# Show plots
plt.show()



