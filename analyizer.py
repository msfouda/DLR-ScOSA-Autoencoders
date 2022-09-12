# TODO

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

enc = e_c.predict(conv_output_samples)
dec = d_c.predict(enc)

predicted = convolutional_model.predict(conv_output_samples)

# display the samples, encodings and decoded values!
# display_results(conv_output_samples, encoded, predicted, enc_shape=(16,16))
display_results(conv_output_samples, encoded, dec, enc_shape=(32,32))

plt.show()