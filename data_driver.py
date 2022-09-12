import config

class DataDriver:
    def __init__(self, config):
        self.dataset_path = config.DATASET_PATH
        self.config = config
            

    def map_image(self, image):
        '''Normalizes the image.'''
        image = tf.cast(image, dtype=tf.float32)
        image = image / 255.0

        return image, image

    def load_dataset(self):
        '''Loads the dataset from the given path.'''
        self.dataset = np.load(self.dataset_path)
        print(f"Dataset shape: {self.dataset.shape}")


    def split_dataset(self):
        '''Splits the dataset into train and test sets.'''
        self.train_dataset = self.dataset[:int(self.config.SPLIT_RATIO*len[self.dataset])]
        self.test_dataset = self.dataset[int(self.config.SPLIT_RATIO*len[self.dataset]):]

        # Shuffle the dataset
        np.random.shuffle(self.train_dataset)
        np.random.shuffle(self.test_dataset)

        print(f"Train dataset shape: {self.train_dataset.shape}")
        print(f"Test dataset shape: {self.test_dataset.shape}")

    def prepare_dataset(self):
        '''Prepares the dataset for training.'''
        self.train_dataset = tf.data.Dataset.from_tensor_slices(self.train_dataset)
        self.train_dataset = self.train_dataset.map(self.map_image)
        self.train_dataset = self.train_dataset.shuffle(self.config.SHUFFLE_BUFFER_SIZE)
        self.train_dataset = self.train_dataset.batch(self.config.BATCH_SIZE)

        self.test_dataset = tf.data.Dataset.from_tensor_slices(self.test_dataset)
        self.test_dataset = self.test_dataset.map(self.map_image)
        self.test_dataset = self.test_dataset.batch(self.config.BATCH_SIZE)
