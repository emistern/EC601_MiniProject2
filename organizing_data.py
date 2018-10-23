# The purpose of this program is to go into the downloaded cocodataset and get the desired Categories to use
# for training an image classifier. It will then shuffle the images and split them by 80/20 training/testing
# and then 80/20 training/validation. It will return three separate txt files with the names of the image files
# in each dataset

# This follows the cocodatasetAPI Python Demo found here: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoDemo.ipynb

# Import Needed Libraries
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import shutil
import os

# Setup the pylab parameters
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

# Input path to directory of images/annotations
dataDir='./Coco_Dataset/'
dataType='val2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

# initialize COCO api for instance annotations
coco=COCO(annFile)

# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
#print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
#print('COCO supercategories: \n{}'.format(' '.join(nms)))


# get all images containing given categories, select one at random
#catIds = coco.getCatIds(catNms=['chair','toilet']);
catIds_chair = coco.getCatIds(catNms=['chair']);
imgIds_chair = coco.getImgIds(catIds=catIds_chair);
catIds_toilet = coco.getCatIds(catNms=['toilet']);
imgIds_toilet = coco.getImgIds(catIds=catIds_toilet);

# Make the datasets the same size
smaller_class = min(len(imgIds_chair), len(imgIds_toilet))
shortened_imgIds_chair = imgIds_chair[:smaller_class]
shortened_imgIds_toilet = imgIds_toilet[:smaller_class]

#################################################################
######## Split the datasets into training,val, test #############
#################################################################
# # Need to split the dataset into training, testing, and validation set. 
# # split the entire list to have 80/20 training vs testing

# What is 80% of the chairs list?
eighty_pct_full_images = int(.8*len(shortened_imgIds_chair))
print('80% of images is: ', eighty_pct_full_images)
big_training_dataset_chair = shortened_imgIds_chair[:eighty_pct_full_images]
testing_dataset_chair = shortened_imgIds_chair[eighty_pct_full_images:]
eighty_pct_training = int(.8*len(big_training_dataset_chair))
print('80% of Training images is: ', eighty_pct_training)
training_dataset_chair = big_training_dataset_chair[:eighty_pct_training]
validation_dataset_chair = big_training_dataset_chair[eighty_pct_training:]

# What is 80% of the toilet list?
big_training_dataset_toilet = shortened_imgIds_toilet[:eighty_pct_full_images]
testing_dataset_toilet = shortened_imgIds_toilet[eighty_pct_full_images:]
training_dataset_toilet = big_training_dataset_toilet[:eighty_pct_training]
validation_dataset_toilet = big_training_dataset_toilet[eighty_pct_training:]


#################################################################
######## Move the training items to correct Folders #############
#################################################################
# # Copy the chair images into the training folder
# image_file_chair = []
# for image in training_dataset_chair:
# 	image = str(image)
# 	paddedimage = image.zfill(12)
# 	image_name = paddedimage+'.jpg'
# 	image_file_chair.append(image_name)

# source = "./Coco_Dataset/val2017/"
# dest = "./training/chair"
# for image in image_file_chair:
# 	try:
# 		shutil.copy2(source+image, dest)
# 	except:
# 		pass

# # # Copy the toilet images into the training folder
# image_file_toilet = []
# for image in training_dataset_toilet:
# 	image = str(image)
# 	paddedimage = image.zfill(12)
# 	image_name = paddedimage+'.jpg'
# 	image_file_toilet.append(image_name)

# source = "./Coco_Dataset/val2017/"
# dest = "./training/toilet"
# for image in image_file_toilet:
# 	try:
# 		shutil.copy2(source+image, dest)
# 	except:
# 		pass



#################################################################
###### Move the each specified class of photos to it's dir ######
#################################################################
# Copy the chair images into the chair folder
image_file_chair = []
for image in shortened_imgIds_chair:
	image = str(image)
	paddedimage = image.zfill(12)
	image_name = paddedimage+'.jpg'
	image_file_chair.append(image_name)

source = "./Coco_Dataset/val2017/"
dest = "./training/chair"
for image in image_file_chair:
	try:
		shutil.copy2(source+image, dest)
	except:
		pass

# # Copy the toilet images into the toilet folder
image_file_toilet = []
for image in shortened_imgIds_toilet:
	image = str(image)
	paddedimage = image.zfill(12)
	image_name = paddedimage+'.jpg'
	image_file_toilet.append(image_name)

source = "./Coco_Dataset/val2017/"
dest = "./training/toilet"
for image in image_file_toilet:
	try:
		shutil.copy2(source+image, dest)
	except:
		pass








