Boston University EC601 Miniproject 2:

Objectives: Retrain two models to classify two classes of objects. 

Dataset used: 2017 Coco Validation Images (http://cocodataset.org/#download)

Items to install:
1. Tensorflow (https://www.tensorflow.org/install/)
2. CocoAPI (https://github.com/cocodataset/cocoapi)

Classes:
1. Toilet
2. Chair

Models:
1. Tensorflow: image feature extraction module Inception V3 trained on ImageNet (took approximately 15 minutes)
	Final test accuracy = 96.3% (N=82)
	Documentation: https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1	
	Cmd line prompt for retraining this model:
	python retrain.py --image_dir Users/Emiannstern/Documents/EC601_Prod_Design/EC601_MiniProject2/training/ --saved_model_dir ./tmp/inception_v3 --how_many_training_steps 3000 --summaries_dir ./log/inception_v3

2. Tensorflow: ResNet V2 50 trained on ImageNet (took approximately 7 minutes)
	Final test accuracy = 89.0% (N=82)
	Documentation: https://tfhub.dev/google/imagenet/resnet_v2_50/classification/1
	Cmd line prompt for retraining this model:
	python retrain.py --image_dir /Users/Emiannstern/Documents/EC601_Prod_Design/EC601_MiniProject2/training/ --tfhub_module https://tfhub.dev/google/imagenet/resnet_v2_50/classification/1 --saved_model_dir ./tmp/resnet_v2 --how_many_training_steps 300 --summaries_dir = ./log/resnet_v2


Files:
1. organizing_data.py
	This file looks into the specified Cocodataset directory gets the images for the desired classes. It then either:
	- splits the specified images of given classes into training, testing, and validation datasets. 80/20 training/testing of full dataset. then the training dataset is split 80/20 training/validation. It moves the training images into the approriate directories.
	- Moves all of the images into separate folders based on their class name
2. example_code/retrain.py
	This file is used to retrain various tensorflow models. (https://www.tensorflow.org/hub/tutorials/image_retraining). 
3. example_code/label_image.py
	This file is used to pass a test image to the retrained network from file (2) and return the probabilities that each class is in the picture. (https://www.tensorflow.org/hub/tutorials/image_retraining)
4. tmp_inceptionv3/output_graph.pb
	This file contains the updated weights for the newly trained inceptionv3
5. tmp_resnet_v2/output_graph.pb
	This file contains the updated weights for the newly trained resnet v2 50


Retraining:
1. Resnet V2
	- Ran Resnet V2 with 3000 steps. Checked tensorboard output. Looked like the model started to overfit 
	at around 60 steps. 
2. Inception 
	- Ran model with 3000 steps. Checked tensorboard output. Looked like the model started to overfit at around 100 steps.
	python label_image.py --image=/Users/Emiannstern/Downloads/toilet1.jpg verified that it works (you can check it with any picture of a chair or toilet.)


