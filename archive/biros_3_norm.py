import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import tensorflow as tf
# import tensorflow_datasets as tfds
from tensorflow.keras.utils import plot_model
import numpy as np
import matplotlib.pyplot as plt

from pdb import set_trace


def map_image(image):
  '''Normalizes the image. Returns image as input and label.'''
#   image = tf.cast(image, dtype=tf.float32)
#   image = image / 255.0

  return image, image

BATCH_SIZE = 4
SHUFFLE_BUFFER_SIZE = 10

# load npy file
biros_data= np.load("biros_dataset.npy")

print(biros_data.shape)
# split biros_data into train and test data
train_dataset = biros_data[:int(0.8*len(biros_data))]
# train_dataset = np.array([train_dataset, train_dataset])
test_dataset = biros_data[int(0.8*len(biros_data)):]

np.random.shuffle(train_dataset)
np.random.shuffle(test_dataset)
# set_trace()

print(train_dataset.shape)
print(test_dataset.shape)

# # convert the dataset to a tf.data.Dataset
# train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset)
# train_dataset = train_dataset.map(map_image)


# test_dataset = tf.data.Dataset.from_tensor_slices(test_dataset)

# train_dataset = test_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE).repeat()
# test_dataset = test_dataset.batch(BATCH_SIZE).repeat()


def encoder(inputs):
  '''Defines the encoder with two Conv2D and max pooling layers.'''
  
  # conv = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same')(inputs)
  # max_pool = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(conv)

  # conv_0 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(max_pool)
  conv_0 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(inputs)
  
  BNd = tf.keras.layers.BatchNormalization()(conv_0)

  max_pool_0 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(BNd)
  # max_pool_0 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(conv_0)

  conv_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same')(max_pool_0)
  max_pool_1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(conv_1)

  BNd = tf.keras.layers.BatchNormalization()(max_pool_1)

  # conv_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same')(max_pool_1)
  conv_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same')(BNd)
  max_pool_2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(conv_2)

  return max_pool_2

def bottle_neck(inputs):
  '''Defines the bottleneck.'''
  bottle_neck = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same')(inputs)
  encoder_visualization = tf.keras.layers.Conv2D(filters=1, kernel_size=(3,3), activation='sigmoid', padding='same')(bottle_neck)

  return bottle_neck, encoder_visualization

def decoder(inputs):
  '''Defines the decoder path to upsample back to the original image size.'''
  
  conv = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same')(inputs)
  
  BNd = tf.keras.layers.BatchNormalization()(conv)

  # up_sample = tf.keras.layers.UpSampling2D(size=(2,2))(conv)
  up_sample = tf.keras.layers.UpSampling2D(size=(2,2))(BNd)

  conv_0 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same')(up_sample)
  up_sample_0 = tf.keras.layers.UpSampling2D(size=(2,2))(conv_0)

  conv_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(up_sample_0)
  up_sample_1 = tf.keras.layers.UpSampling2D(size=(2,2))(conv_1)

  # conv_2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same')(up_sample_1)
  # up_sample_2 = tf.keras.layers.UpSampling2D(size=(2,2))(conv_2)

  BNd = tf.keras.layers.BatchNormalization()(up_sample_1)

  # conv_3 = tf.keras.layers.Conv2D(filters=1, kernel_size=(3,3), activation='sigmoid', padding='same')(up_sample_2)
  # conv_3 = tf.keras.layers.Conv2D(filters=1, kernel_size=(3,3), activation='sigmoid', padding='same')(up_sample_1)
  conv_3 = tf.keras.layers.Conv2D(filters=1, kernel_size=(3,3), activation='sigmoid', padding='same')(BNd)

  return conv_3

def convolutional_auto_encoder():
  '''Builds the entire autoencoder model.'''
  inputs = tf.keras.layers.Input(shape=(256, 256, 1,))
  encoder_output = encoder(inputs)
  bottleneck_output, encoder_visualization = bottle_neck(encoder_output)
  decoder_output = decoder(bottleneck_output)
  
  model = tf.keras.Model(inputs =inputs, outputs=decoder_output)
  encoder_model = tf.keras.Model(inputs=inputs, outputs=encoder_visualization)
  return model, encoder_model

convolutional_model, convolutional_encoder_model = convolutional_auto_encoder()
convolutional_model.summary()

train_steps = 336 // BATCH_SIZE
valid_steps = 336 // BATCH_SIZE
EPOCHS = 50

# convolutional_model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy')
# convolutional_model.compile(optimizer=tf.keras.optimizers.Adam(), loss='crossentropy')
convolutional_model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')

conv_model_history = convolutional_model.fit(train_dataset, train_dataset,shuffle=True, steps_per_epoch=train_steps, validation_data=(test_dataset, test_dataset), validation_steps=valid_steps, epochs=EPOCHS)
# conv_model_history = convolutional_model.fit(train_dataset, train_dataset, steps_per_epoch=train_steps, validation_data=(test_dataset, test_dataset), epochs=EPOCHS)
# conv_model_history = convolutional_model.fit(train_dataset, train_dataset, steps_per_epoch=train_steps, epochs=EPOCHS)
# conv_model_history = convolutional_model.fit(train_dataset, train_dataset, steps_per_epoch=train_steps, epochs=EPOCHS)

# # # save the model
convolutional_model.save('biros_model_mse_256.h5')
convolutional_encoder_model.save('biros_encoder_model_mse_256.h5')

convolutional_model.summary()

# load the model
# convolutional_model = tf.keras.models.load_model('biros_model_mse.h5')
# convolutional_encoder_model = tf.keras.models.load_model('biros_encoder_model_mse.h5')


plot_model(convolutional_model, show_shapes=True, show_layer_names=True, to_file='outer-model.png')

plt.plot(conv_model_history.history['loss'])
plt.plot(conv_model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

def display_one_row(disp_images, offset, shape=(256, 256)):
  '''Display sample outputs in one row.'''
  for idx, test_image in enumerate(disp_images):
    plt.subplot(3, 10, offset + idx + 1)
    plt.xticks([])
    plt.yticks([])
    test_image = np.reshape(test_image, shape)
    plt.imshow(test_image, cmap='gray')


def display_results(disp_input_images, disp_encoded, disp_predicted, enc_shape=(8,4)):
  '''Displays the input, encoded, and decoded output values.'''
  plt.figure(figsize=(15, 5))
  display_one_row(disp_input_images, 0, shape=(256,256,))
  display_one_row(disp_encoded, 10, shape=enc_shape)
  display_one_row(disp_predicted, 20, shape=(256,256,))

  # take 1 batch of the dataset
# test_dataset = test_dataset.take(1)

# take the input images and put them in a list
# output_samples = []
# for i in range(10):
#       output_samples = input_image

# pick 10 indices
idxs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# get 10 random images from the test dataset
conv_output_samples = test_dataset[:10]
# set_trace()

# prepare test samples as a batch of 10 images
# conv_output_samples = np.array(output_samples[idxs])
# conv_output_samples = np.reshape(conv_output_samples, (10, 500, 500, 1))
# conv_output_samples = test_dataset[1].reshape((1, 500, 500, 1))

# get the encoder ouput
encoded = convolutional_encoder_model.predict(conv_output_samples)

# get a prediction for some values in the dataset
predicted = convolutional_model.predict(conv_output_samples)

# display the samples, encodings and decoded values!
# display_results(conv_output_samples, encoded, predicted, enc_shape=(16,16))
display_results(conv_output_samples, encoded, predicted, enc_shape=(32,32))

plt.show()