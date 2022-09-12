import config

class Autoencoder:
    def __init__(self) -> None:
        self.encoder = None
        self.decoder = None
        self.bottle_neck = None
        self.autoencoder = None

    def encoder(self, inputs):
        '''Defines the encoder with two Conv2D and max pooling layers.'''
        
        # conv = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same')(inputs)
        # max_pool = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(conv)

        # conv_0 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(max_pool)
        conv_0 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(inputs)
        max_pool_0 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(conv_0)

        conv_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same')(max_pool_0)
        max_pool_1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(conv_1)
        # max_pool_1 = tf.keras.layers.AveragePooling2D(pool_size=(2,2))(conv_1)

        conv_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same')(max_pool_1)
        max_pool_2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(conv_2)
        # max_pool_2 = tf.keras.layers.AveragePooling2D(pool_size=(2,2))(conv_2)

        return max_pool_2

    def bottle_neck(self, inputs):
        '''Defines the bottleneck.'''
        bottle_neck = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same')(inputs)
        encoder_visualization = tf.keras.layers.Conv2D(filters=1, kernel_size=(3,3), activation='sigmoid', padding='same')(bottle_neck)

        return bottle_neck, encoder_visualization

    def decoder(self, inputs):
        '''Defines the decoder path to upsample back to the original image size.'''
        
        conv = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same')(inputs)
        up_sample = tf.keras.layers.UpSampling2D(size=(2,2))(conv)

        conv_0 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same')(up_sample)
        up_sample_0 = tf.keras.layers.UpSampling2D(size=(2,2))(conv_0)

        conv_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(up_sample_0)
        up_sample_1 = tf.keras.layers.UpSampling2D(size=(2,2))(conv_1)

        # conv_2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same')(up_sample_1)
        # up_sample_2 = tf.keras.layers.UpSampling2D(size=(2,2))(conv_2)

        # conv_3 = tf.keras.layers.Conv2D(filters=1, kernel_size=(3,3), activation='sigmoid', padding='same')(up_sample_2)
        conv_3 = tf.keras.layers.Conv2D(filters=1, kernel_size=(3,3), activation='sigmoid', padding='same')(up_sample_1)

        return conv_3

    def convolutional_auto_encoder(self):
        '''Builds the entire autoencoder model.'''
        inputs = tf.keras.layers.Input(shape=(256, 256, 1,))
        encoder_output = encoder(inputs)
        bottleneck_output, encoder_visualization = bottle_neck(encoder_output)
        decoder_output = decoder(bottleneck_output)
        
        model = tf.keras.Model(inputs =inputs, outputs=decoder_output)
        bottleneck_visualization = tf.keras.Model(inputs=inputs, outputs=encoder_visualization)
        decoder_model = tf.keras.Model(inputs=bottleneck_output, outputs=decoder_output)
        encoder_model = tf.keras.Model(inputs=inputs, outputs=bottleneck_output)
        
        return model, bottleneck_visualization, encoder_model, decoder_model