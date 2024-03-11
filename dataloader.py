import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow.keras as k


num_classes = 10
random.seed(42)
# Load the data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
img_rows, img_cols, channels = 32, 32, 3
# set up image augmentation
datagen = ImageDataGenerator(
    # randomly rotate images in the range (degrees, 0 to 180)
    rotation_range=40,
    horizontal_flip=True,  # randomly flip images
    # randomly shift images horizontally (fraction of total width)
    width_shift_range=0.1,
    # randomly shift images vertically (fraction of total height)
    height_shift_range=0.1,
    # zoom_range=0.3 # randomly zoom images
    fill_mode='nearest'  # fill in missing pixels with the nearest filled value
)

datagen.fit(x_train)

augmented_data_gen = datagen.flow(x_train, y_train)

# y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
# y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

# reshape into images
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)
input_shape = (img_rows, img_cols, 1)

# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')


# Check that it's actuall images
# plt.imshow(x_train[0], cmap=plt.cm.binary)


# Normalizing
x_train = x_train/255
x_test = x_test/255
# One-Hot-Encoding
Y_train_en = to_categorical(y_train, 10)
Y_test_en = to_categorical(y_test, 10)

# convert integers to float; normalise and center the mean
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
mean = np.mean(x_train)
std = np.std(x_train)
x_test = (x_test-mean)/std
x_train = (x_train-mean)/std

y_train = k.utils.to_categorical(y_train, num_classes)
y_test = k.utils.to_categorical(y_test, num_classes)
# Check the shape of the data
# print(len(x_train))
# print(len(x_test))
# plt.imshow(x_train[0], cmap=plt.cm.binary)
# plt.show()
