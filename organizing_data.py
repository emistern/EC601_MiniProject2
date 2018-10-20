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
catIds = coco.getCatIds(catNms=['person','furniture']);
imgIds = coco.getImgIds(catIds=catIds );
#imgIds = coco.getImgIds(imgIds = [241668])
#img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]

# Use the commented out code below to show one image 
# I = io.imread(img['coco_url'])
# plt.axis('off')
# plt.imshow(I)
# plt.show()

# Get the number of images in dataset
print(len(imgIds))

# Need to split the dataset into training, testing, and validation set. 
# split the entire list to have 80/20 training vs testing

# What is 80% of the images list?
eighty_pct_full_images = int(.8*len(imgIds))
print('80% of images is: ', eighty_pct_full_images)
big_training_dataset = imgIds[:eighty_pct_full_images]
testing_dataset = imgIds[eighty_pct_full_images:]
eighty_pct_training = int(.8*len(big_training_dataset))
print('80% of Training images is: ', eighty_pct_training)
training_dataset = big_training_dataset[:eighty_pct_training]
validation_dataset = big_training_dataset[eighty_pct_training:]

# Write the training, testing, and validation data sets to a file
# with open("training_dataset.txt", "w") as f:
#     f.writelines(map("{}\n".format, training_dataset))

# with open("validation_dataset.txt", "w") as f:
#     f.writelines(map("{}\n".format, validation_dataset))

# with open("testing_dataset.txt", "w") as f:
#     f.writelines(map("{}\n".format, testing_dataset))

# Loop over the images in training and if it has an CATID of person move it to the person folder.
# move all of the training data set into a folder. then change the directory of coco to that directory.
# then in that file split the directory into person and furniture data.
image_file = []
for image in training_dataset:
	#image_file = './twitter_images/*' +image + '.jpg'
	#print(image, len(str(image)))
	image = str(image)
	paddedimage = image.zfill(12)
	image_name = paddedimage+'.jpg'
	image_file.append(image_name)
	#print(image, len(str(image)), " ", image_name)

source = "./Coco_Dataset/val2017/"
dest = "./training"

for image in image_file:
	shutil.move(source+image, dest)
	# move each image to the training directory
# catId_person = coco.getCatIds(catNms=['person']);
# catId_furniture = coco.getCatIds(catNms=['furniture']);


# for iImage in range(len(training_dataset)):
# 	print(coco.getCatIds(imgIds = [training_dataset[iImage]]))


