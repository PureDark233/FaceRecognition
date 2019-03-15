import numpy as np
import os
from sklearn.model_selection import train_test_split
from PIL import Image
import diy
np.random.seed(1337)

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils

nb_classes = 3
nb_epoch = 3
batch_size = 3

size = 64
nb_filters1, nb_filters2 = 5, 10
nb_pool = 2
nb_conv = 3
face=np.empty((3,12288))
itr=0
def load_data(dataset_path):
	global itr
	for filename in os.listdir(dataset_path):
		if filename.endswith('.jpg'):
			filename = dataset_path + '/' + filename
			img = Image.open(filename)
			img_array = np.asarray(img, dtype='float64') / 256
			img_array = np.ndarray.flatten(img_array)
			face[itr] = img_array
			itr += 1

load_data("ft")
model=diy.Net_model()
model.load_weights('mw.h5')
test_X = face.reshape(face.shape[0], 3, size, size)
prediction = model.predict_classes(test_X, verbose=0)
print(prediction)
