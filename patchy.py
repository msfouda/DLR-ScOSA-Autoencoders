import numpy as np
from patchify import patchify, unpatchify

# import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import cv2

from pdb import set_trace 

# img = cv2.imread("GeoEye_Ikonos_1m_8bit_RGB_DRA_Oil_2005NOV25_8bits_r_1.png")
# img = cv2.imread("image_152.jpg")
# resize img to 5000 x 5000
# img = cv2.resize(img, (4000, 3000))

# x = cv2.imread("GeoEye_Ikonos_1m_8bit_RGB_DRA_Oil_2005NOV25_8bits_r_1.png")
# patches_img = patchify(img, (400,500,3), step=100)

# # print patches_img.shape
# print(patches_img.shape)

# sace patches as numpy arraynpy
# np.save("patches_img.npy", patches_img)

# for i in range(patches_img.shape[0]):
#     for j in range(patches_img.shape[1]):
#         single_patch_img = patches_img[i, j, 0, :, :, :]
#         if not cv2.imwrite('patches/images/' + 'image_' + '_'+ str(i)+str(j)+'.jpg', single_patch_img):
#             raise Exception("Could not write the image")

# # load patches as numpy array
# patches_img = np.load("patches_img.npy")
# patches_img = np.load("patches_biros_5_500_100.npy")
patches_img = np.load("output.npy")

print(patches_img.shape)

# create empty image array of size 500 x 500
# imgs = []

# for i in range(patches_img.shape[0]):
#     for j in range(patches_img.shape[1]):
#         # set_trace()
#         single_patch_img = patches_img[i, j, :, :]
#         # if not cv2.imwrite('patches/images_5000/' + 'image_' + '_'+ str(i)+str(j)+'.jpg', single_patch_img):
#         #     raise Exception("Could not write the image")
#         imgs.append(single_patch_img.reshape(500, 500, 1))

# imgs = np.array(imgs)

# set_trace()
# np.save("biros_data.npy", imgs)
# np.save("external_data.npy", imgs)

# # unpatchify
img = unpatchify(patches_img, (1024, 1024))
img = cv2.resize(img, (1024, 1651))

# show image
plt.imshow(img)
plt.show()

