# TODO
# load the model
convolutional_model = tf.keras.models.load_model('biros_model_mse_256_final.h5')
convolutional_encoder_model = tf.keras.models.load_model('biros_encoder_model_mse_256_final.h5')
enc_m = tf.keras.models.load_model('biros_enc_mse_256_final.h5')
dec_m = tf.keras.models.load_model('biros_dec_mse_256_final.h5')

convolutional_model.summary()

plot_model(convolutional_model, show_shapes=True, show_layer_names=True, to_file='outer-model_256_32_final_2.png')
plot_model(e_c, show_shapes=True, show_layer_names=True, to_file='outer-encoder_256_32_final_2.png')
plot_model(d_c, show_shapes=True, show_layer_names=True, to_file='outer-decoder_256_32_final_2.png')

plt.plot(conv_model_history.history['loss'])
plt.plot(conv_model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('loss_256_32_final_12.png')
plt.show()

# save plot as png
