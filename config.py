import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt

class Config:

    def __init__(self, mode="train"):
        '''The configurator constructor'''
        self.mode = mode
        self.set_environment_params()
        self.set_params()

    def set_environment_params(self):
        '''Set environment parameters'''
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        # os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
        self.PATH = os.path.dirname(os.path.abspath(__file__))
        self.OUTPUT_PATH = os.path.join(self.PATH, "output")
        self.CHECKPOINT_PATH = os.path.join(self.OUTPUT_PATH, "checkpoints")
        self.LOG_PATH = os.path.join(self.OUTPUT_PATH, "logs")

    def set_params(self):
        '''Function to set the training parameters'''
        if self.mode == "train":
            self.set_training_params()
        elif self.mode == "infer":
            self.set_training_params()
        else:
            raise ValueError("Invalid mode")
            

    def set_training_params(self):
        '''Set training parameters'''
        self.BATCH_SIZE = 12
        self.EPOCHS = 500
        self.SHUFFLE_BUFFER_SIZE = 10
        self.set_dataset_params()


    def set_inference_params(self):
        '''Set inference parameters'''
        self.BATCH_SIZE = 1


    def set_dataset_params(self):
        '''Set dataset parameters'''
        dataset_file = "biros_dataset.npy"
        self.DATASET_PATH = os.path.join(self.PATH, dataset_file)
        self.SPLIT_RATIO = 0.8