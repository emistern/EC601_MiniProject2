# The purpose of this program is to go into the downloaded cocodataset and get the desired Categories to use
# for training an image classifier.

# This follows the cocodatasetAPI Python Demo found here: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoDemo.ipynb

# Import Needed Libraries
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab

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
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))


# get all images containing given categories, select one at random
catIds = coco.getCatIds(catNms=['person','chair']);
imgIds = coco.getImgIds(catIds=catIds );
#imgIds = coco.getImgIds(imgIds = [241668])
img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]

print(imgIds)

# I = io.imread(img['coco_url'])
# plt.axis('off')
# plt.imshow(I)
# plt.show()