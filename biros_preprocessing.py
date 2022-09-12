import numpy as np

# load npy file
biros_data= np.load("biros_data.npy")

print(biros_data.shape)
# split biros_data into train and test data
train_data = biros_data[:int(0.8*len(biros_data))]
test_data = biros_data[int(0.8*len(biros_data)):]

print(train_data.shape)
print(test_data.shape)