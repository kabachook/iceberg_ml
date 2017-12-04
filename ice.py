import os
os.environ['PATH'] = f'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\bin;{os.environ["PATH"]}'

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LeakyReLU
from keras.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

from model import callbacks, file_path, MyNetv2, MyNetv3


# Constants
batch_size = 64
epochs = 100
predict = False

seed = 4444

data = pd.read_json("input\\train.json")
data.inc_angle = data.inc_angle.replace('na', 0)
data.inc_angle = data.inc_angle.astype(float).fillna(0.0)


def gen_flow_for_two_inputs(X1, X2, y, gen):
    genX1 = gen.flow(X1, y,  batch_size=batch_size, seed=seed)
    genX2 = gen.flow(X1, X2, batch_size=batch_size, seed=seed)
    while True:
        X1i = genX1.next()
        X2i = genX2.next()
        # Assert arrays are equal - this was for peace of mind, but slows down training
        # np.testing.assert_array_equal(X1i[0],X2i[0])
        yield [X1i[0], X2i[1]], X1i[1]


# Add band 3 = (band_1 + band_2)/2
x_band_1 = np.array([np.array(band).astype(
    np.float32).reshape(75, 75) for band in data["band_1"]])
x_band_2 = np.array([np.array(band).astype(
    np.float32).reshape(75, 75) for band in data["band_2"]])
x_bands = np.concatenate([x_band_1[:, :, :, np.newaxis], x_band_2[:, :, :, np.newaxis], ((
    x_band_1 + x_band_2) / 2)[:, :, :, np.newaxis]], axis=-1)
data_angle = np.array(data.inc_angle)
data_angle = data_angle / np.max(data_angle)



# Split data
data_y = data['is_iceberg']
x_train, x_test, x_train_angle, x_test_angle,  y_train, y_test = train_test_split(
    x_bands, data_angle, data_y, random_state=seed, train_size=0.75)

# Image generator
augmentation = ImageDataGenerator(horizontal_flip=True,
                                  vertical_flip=True,
                                  width_shift_range=0.1,
                                  height_shift_range=0.1,
                                  zoom_range=0.1,
                                  rotation_range=360)

gen_flow = gen_flow_for_two_inputs(
    x_train, x_train_angle, y_train, augmentation)


# model = MyNetv3(75,75)
model = MyNetv2()
model.fit_generator(gen_flow, epochs=epochs,
                    verbose=1, validation_data=([x_test, x_test_angle], y_test), callbacks=callbacks,
                    steps_per_epoch=len(x_train) / batch_size)


model.load_weights(filepath=file_path)  # Load best result
score = model.evaluate([x_test, x_test_angle], y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
