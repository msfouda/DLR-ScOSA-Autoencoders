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

def compress_patches(patches_img, imgs, convolutional_encoder_model):

    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):
            # set_trace()
            single_patch_img = patches_img[i, j, :, :]
            # if not cv2.imwrite('patches/images_5000/' + 'image_' + '_'+ str(i)+str(j)+'.jpg', single_patch_img):
            #     raise Exception("Could not write the image")
            single_patch_img = single_patch_img.reshape(1, 256, 256, 1)
            # set_trace()
            # get the encoder ouput
            encoded = convolutional_encoder_model.predict(single_patch_img)
            # append to tuple
            imgs[i,j,:, :, :] = encoded.reshape(32, 32, 256)


# load the model
# convolutional_model = tf.keras.models.load_model('biros_model_mse_256.h5')
convolutional_encoder_model = tf.keras.models.load_model('biros_enc_mse_256.h5')

filename = 'patches_img_test.npy'

patches_img = np.load(filename)
print(f"{filename} shape:", patches_img.shape)

# make emty array to store the compressed patches same size as patches_img
compressed_patchs = np.empty((patches_img.shape[0], patches_img.shape[1], 32, 32, 256))

compress_patches(patches_img, compressed_patchs, convolutional_encoder_model)

# set_trace()

# get a prediction for some values in the dataset
# predicted = convolutional_model.predict(img)

np.save("compressed_patchs.npy", compressed_patchs)
