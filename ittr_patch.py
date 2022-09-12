from fileinput import filename
import numpy as np
from patchify import patchify, unpatchify

# import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import glob

from pdb import set_trace 

def prepare_dataset(filename, imgs):
    patches_img = np.load(filename)

    print(f"{filename} shape:", patches_img.shape)

    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):
            single_patch_img = patches_img[i, j, :, :]
            imgs.append(single_patch_img.reshape(256, 256, 1))


# create empty image array of size None x 256 x 260 x 1
data = []

# for each file in the patches_biros_256_128 folder run the prepare_dataset function
for filename in glob.glob('patches_biros_256_128/*.npy'):
    prepare_dataset(filename, data)

data = np.array(data)
print("Total dataset shape:", data.shape)

# set_trace()
np.save("biros_dataset.npy", data)

