# USAGE
# python quantize.py --dataset ../datasets/dog-breeds \
# 	--model models/dogbreed_model.h5 --tflite-model models/dogbreed_model.tflite

from imutils import paths
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import tensorflow as tf
import numpy as np
import argparse
import os
import tensorflow_datasets as tfds

from pdb import set_trace as pb

# make the args dictionary global
global args

# preprocessing function
def map_image(image, label):
  image = tf.cast(image, dtype=tf.float32)
  image = image / 255.0

  return image, image # dataset label is not used. replaced with the same image input.

# parameters
BATCH_SIZE = 100
SHUFFLE_BUFFER_SIZE = 1024

def representative_dataset():
	# grab the list of images, shuffle them and choose the first 100
	# print("[INFO] loading images...")
	# imagePaths = list(paths.list_images(args["dataset"]))
	# np.random.shuffle(imagePaths)
	# imagePaths = imagePaths[:100]

	# # initialize the image preprocessors
	# aap = AspectAwarePreprocessor(299, 299)
	# iap = ImageToArrayPreprocessor()

	# # load the dataset from disk
	# sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
	# (data, _) = sdl.load(imagePaths, verbose=100)
	
	# use tfds.load() to fetch the 'train' split of CIFAR-10
	train_dataset = tfds.load('cifar10', as_supervised = True, split = 'test')

	# preprocess the dataset with the `map_image()` function above
	data = train_dataset.map(map_image) 

	# shuffle and batch the dataset
	# data = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

	# pb()

	# loop over the image data
	for image in data:
		# change the data type of the image to float32 (required by
		# the converter), reshape the image, pre-process, and yield it
		image = np.array(image, dtype="float32")
		# image = np.reshape(image, (1, 32, 32, 3))
		image = preprocess_input(image)
		yield [image]

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=False,
	help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
	help="path to input keras model")
ap.add_argument("-t", "--tflite-model", required=True,
	help="path to output tflite model")
args = vars(ap.parse_args())

# instantiate a converter object used to perform full integer
# quantization of weights and activations and set the optimization
# flag
converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(
	args["model"])
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# set the representative dataset using the custom generator defined
# above (this is used to get the dynamic range of activations)
converter.representative_dataset = representative_dataset

# set the inference input and output type to UINT8
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

# set the target operation and convert the keras model to quantized
# TFLite model
print("[INFO] converting the keras model to quantized TFLite format...")
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
pb()
quantizedModel = converter.convert()

# serialize the quantized TFLite model to disk
print("[INFO] serializing the TFLite model...")
f = open(args["tflite_model"], "wb")
f.write(quantizedModel)
f.close()