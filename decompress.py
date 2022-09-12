import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import tensorflow as tf
# import tensorflow_datasets as tfds
from tensorflow.keras.utils import plot_model
import numpy as np
import matplotlib.pyplot as plt

from pdb import set_trace

from fileinput import filename
import numpy as np
from patchify import patchify, unpatchify

# import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import glob

from pdb import set_trace 

def compress_patches(patches_img, imgs, convolutional_decoder_model):

    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):
            # set_trace()
            single_patch_img = patches_img[i, j, :, :, :]
            # get the encoder ouput
            decoded = convolutional_decoder_model.predict(single_patch_img.reshape(1, 32, 32, 256))
            # append to tuple
            imgs[i,j,:,:] = decoded.reshape(256, 256)


# load the model
convolutional_decoder_model = tf.keras.models.load_model('biros_dec_mse_256.h5')

filename = 'compressed_patchs.npz'

patches_img = np.load(filename)['arr_0']
set_trace()
print(f"{filename} shape:", patches_img.shape)

# make emty array to store the compressed patches same size as patches_img
output = np.empty((patches_img.shape[0], patches_img.shape[1], 256, 256))

compress_patches(patches_img, output, convolutional_decoder_model)

# set_trace()

# get a prediction for some values in the dataset
# predicted = convolutional_model.predict(img)

np.save("output.npy", output)
