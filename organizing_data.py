# The purpose of this program is to go into the downloaded cocodataset and get the desired Categories to use
# for training an image classifier.

# This follows the cocodatasetAPI Python Demo found here: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoDemo.ipynb

# Import Needed Libraries
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
