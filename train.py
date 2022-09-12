# In progress
import config
from autoencoder import Autoencoder

class Train:
    def __init__(self, config):
        self.config = config
        self.data_driver = DataDriver(config)
        self.data_driver.load_dataset()
        self.data_driver.split_dataset()
        self.data_driver.prepare_dataset()
        self.build_model()
        self.train()
        self.save_model()
        self.test()

    def build_model(self):
        '''Builds the model.'''
        autoencoder = Autoencoder(config)
        self.model = autoencoder.model
        self.bottleneck_visualization = autoencoder.bottleneck_visualization
        self.encoder_model = autoencoder.encoder_model
        self.decoder_model = autoencoder.decoder_model
        self.model.summary()

    def train(self):
        '''Trains the model.'''
        train_steps = train_dataset.shape[0] // BATCH_SIZE
        # valid_steps = train_dataset.shape[0] // BATCH_SIZE
        valid_steps = (train_dataset.shape[0] / 10) // BATCH_SIZE

        self.model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')

        self.model_history = self.model.fit(train_dataset, train_dataset,shuffle=True, steps_per_epoch=train_steps, validation_data=(test_dataset, test_dataset), validation_steps=valid_steps, epochs=EPOCHS)

        self.model.compile(optimizer=self.config.OPTIMIZER, loss=self.config.LOSS)
        self.model.fit(self.data_driver.train_dataset, epochs=self.config.EPOCHS)

    def save_model(self):
        '''Saves the model.'''
        self.model.save(self.config.MODEL_PATH)
        self.model.save('biros_model_mse_256_final_12.h5')
        self.bottleneck_visualization.save('biros_encoder_model_mse_256_final_12.h5')
        self.encoder_model.save('biros_enc_mse_256_final_12.h5')
        self.decoder_model.save('biros_dec_mse_256_final_12.h5')


    def test(self):
        '''Tests the model.'''
        self.model.evaluate(self.data_driver.test_dataset)


if __name__ == "__main__":
    train = Train(config)