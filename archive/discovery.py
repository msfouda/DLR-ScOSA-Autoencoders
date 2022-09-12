import cv2
import numpy as np
import h5py
import pdb
from pathlib import Path
from osgeo import gdal
import matplotlib.pyplot as plt

import tensorflow as tf
from PIL import Image
from patchify import patchify, unpatchify

class discovery:

    def patch_image(self, img):
        # plt.imshow(img)
        # plt.show()

        print(img.shape)
        img = cv2.resize(img, (1024, 1024))

        patches_img = patchify(img, (256,256), step=128)

        # # print patches_img.shape
        # print(patches_img.shape)

        # sace patches as numpy arraynpy
        np.save("patches_img_test.npy", patches_img)

        # for i in range(patches_img.shape[0]):
        #     for j in range(patches_img.shape[1]):
        #         single_patch_img = patches_img[i, j, 0, :, :, :]
        #         if not cv2.imwrite('patches/images/' + 'image_' + '_'+ str(i)+str(j)+'.jpg', single_patch_img):
        #             raise Exception("Could not write the image")

        # load patches as numpy array
        patches_img = np.load("patches_img_test.npy")

        # for i in range(patches_img.shape[0]):
        #     for j in range(patches_img.shape[1]):
        #         single_patch_img = patches_img[i, j, :, :]
        #         plt.imshow(single_patch_img, cmap='afmhot')
        #         plt.savefig('patches/images/' + 'image_' + '_'+ str(i)+str(j)+'.jpg')
        #         # if not cv2.imwrite('patches/images/' + 'image_' + '_'+ str(i)+str(j)+'.jpg', single_patch_img):
        #         #     raise Exception("Could not write the image")

        # unpatchify
        img = unpatchify(patches_img, (1024,1024))

        # show image
        plt.imshow(img, cmap='afmhot')
        plt.show()

    def get_location_map(self, hdf_ds):
        lat = gdal.Open(hdf_ds.GetSubDatasets()[3][0], gdal.GA_ReadOnly)
        long = gdal.Open(hdf_ds.GetSubDatasets()[4][0], gdal.GA_ReadOnly)
        return lat, long

    def run(self):
        # Load the image
        filename = 'img.he5'
        hdf_ds = gdal.Open(filename, gdal.GA_ReadOnly)
        band_ds = gdal.Open(hdf_ds.GetSubDatasets()[1][0], gdal.GA_ReadOnly)

        lat, long = self.get_location_map(hdf_ds)
        band_array = band_ds.ReadAsArray()
        gt = hdf_ds.GetGeoTransform()
        proj = hdf_ds.GetProjection()

        plt.figure()
        plt.imshow(band_array)
        plt.show()


        self.patch_image(band_array)
        # apply the mask
        # bimask = np.where(band_array >= np.mean(band_array), 1, 0)
        # plt.figure()
        # plt.imshow(bimask)
        # plt.show()

        # driver = gdal.GetDriverByName('GTiff')
        # driver.Register()
        # out_ds = driver.Create('bimask.tif', xsize=bimask.shape[1], ysize=bimask.shape[0],
        #                        bands=1)
        # out_ds.SetGeoTransform(gt)
        # out_ds.SetProjection(proj)
        # out = out_ds.GetRasterBand(1)
        # out.WriteArray(bimask)
        # out.SetNoDataValue(np.nan)
        # out.FlushCache()
        # out_ds = None
        # out = None


class test:

    def load_subdataset(self, filename, dataset=1, subdataset=0):
        hdf_ds = gdal.Open(filename, gdal.GA_ReadOnly)
        # print(hdf_ds.GetDatasets())
        # pdb.set_trace()
        band_ds = gdal.Open(hdf_ds.GetSubDatasets()[dataset][subdataset], gdal.GA_ReadOnly)

        band_array = band_ds.ReadAsArray()
        # plot image to check if it is correct
        plt.imshow(band_array)
        # wait for user input
        plt.show()

        # cv2.imshow('image', band_array)
        # cv2.waitKey(0)
        print(band_array)

    def run(self):
        # for each image in the current folder path folder run the function load_subdataset(filename, dataset, subdataset)
        current_path = str(Path(__file__).parent)
        for filename in Path(current_path).glob('*.he5'):
            self.load_subdataset(str(filename))

class viewer:
    def read_tif(self, filename):
        # Load the image
        ds = cv2.imread(filename)

        # plot image to check if it is correct
        plt.imshow(ds)
        # wait for user input
        plt.show()

    def run(self):
        self.read_tif('bimask.tif')

class visualizer:
    def show(self, img_1, img_2):
        # stack the two images side by side on same figure
        plt.subplot(1, 2, 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img_1, cmap='afmhot')
        plt.subplot(1, 2, 2)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img_2, cmap='afmhot')
        plt.show()


    def run(self):
        img_2 = cv2.imread('compressed/decompressed.png')
        img_1 = cv2.imread('compressed/original.png')
        self.show(img_1, img_2)

if __name__ == '__main__':
    # test().run()
    # discovery().run()
    # viewer().run()
    visualizer().run()
