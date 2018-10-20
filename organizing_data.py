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
import glob

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
#catIds = coco.getCatIds(catNms=['chair','person']);
catIds_chair = coco.getCatIds(catNms=['furniture']);
imgIds_chair = coco.getImgIds(catIds=catIds_chair);
catIds_person = coco.getCatIds(catNms=['person']);
imgIds_person = coco.getImgIds(catIds=catIds_person);

# Get the number of images in dataset


# Make the datasets the same size
smaller_class = min(len(imgIds_chair), len(imgIds_person))
shortened_imgIds_chair = imgIds_chair[:smaller_class]
shortened_imgIds_person = imgIds_person[:smaller_class]


# # Need to split the dataset into training, testing, and validation set. 
# # split the entire list to have 80/20 training vs testing

# What is 80% of the images list?
eighty_pct_full_images = int(.8*len(shortened_imgIds_chair))
print('80% of images is: ', eighty_pct_full_images)
big_training_dataset_chair = shortened_imgIds_chair[:eighty_pct_full_images]
testing_dataset_chair = shortened_imgIds_chair[eighty_pct_full_images:]
eighty_pct_training = int(.8*len(big_training_dataset_chair))
print('80% of Training images is: ', eighty_pct_training)
training_dataset_chair = big_training_dataset_chair[:eighty_pct_training]
validation_dataset_chair = big_training_dataset_chair[eighty_pct_training:]

#################################################################
###### Split training,testing,validation into txt files #########
#################################################################
# Write the training, testing, and validation data sets to a file
# with open("training_dataset.txt", "w") as f:
#     f.writelines(map("{}\n".format, training_dataset))

# with open("validation_dataset.txt", "w") as f:
#     f.writelines(map("{}\n".format, validation_dataset))

# with open("testing_dataset.txt", "w") as f:
#     f.writelines(map("{}\n".format, testing_dataset))

# #################################################################
# ########### Move training data into its own folder ##############
# #################################################################
# # Loop over the images in training and if it has an CATID of person move it to the person folder.
# # move all of the training data set into a folder. then change the directory of coco to that directory.
# # then in that file split the directory into person and furniture data.
# image_file = []
# for image in training_dataset:
# 	image = str(image)
# 	paddedimage = image.zfill(12)
# 	image_name = paddedimage+'.jpg'
# 	image_file.append(image_name)

# # SHould add in error checking, if this is already done then pass.
# # source = "./Coco_Dataset/val2017/"
# # dest = "./training"

# # for image in image_file:
# # 	shutil.move(source+image, dest)
# # 	# move each image to the training directory

# #################################################################
# ############# Move each id data into its own folder #############
# #################################################################
# catId_person = coco.getCatIds(catNms=['person']);
# catId_furniture = coco.getCatIds(catNms=['furniture']);
# imgIds_person = coco.getImgIds(catIds=catId_person );
# imgIds_furniture = coco.getImgIds(catIds=catId_furniture );

# training_image_files = glob.glob("./training/*.jpg")
# #print(training_image_files)
# test_str = './training/000000032861.jpg'
# test2=test_str.rsplit('/',1)[1]
# test3 = test2.rsplit('.',1)[0]
# print(test3.lstrip("0"))
# for image in training_image_files:
# 	temp1 = image.rsplit('/',1)[1]
# 	temp2 = temp1.rsplit('.',1)[0]
# 	temp3 = temp2.lstrip("0")
# 	if int(temp3) in imgIds_person:
# 		#shutil.move(image, "./training/person")
# 		print(temp3, "Person")
# 	if int(temp3) in imgIds_furniture:
# 		#shutil.move(image, "./training/furniture")
# 		print(temp3, "FURNITURE")

# print(len(imgIds_furniture))
# print(len(imgIds_person))
# print(len(imgIds))


# # Loop over the list of files in the directory. get the id by looking at
# # the contents between 0 and .jpg. Then check if that ID is in the 
# # imgIds_person list, if it is then move it to the person directory.


# # for iImage in range(len(training_dataset)):
# # 	print(coco.getCatIds(imgIds = [training_dataset[iImage]]))


