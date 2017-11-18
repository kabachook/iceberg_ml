import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, ELU
from keras.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os

from model import MyNet, callbacks, file_path

# Constants
batch_size = 64
epochs = 10
predict = True

train = pd.read_json("input\\train.json")


# Add band 3 = (band_1 + band_2)/2
X_band_1 = np.array([np.array(band).astype(
    np.float32).reshape(75, 75) for band in train["band_1"]])
X_band_2 = np.array([np.array(band).astype(
    np.float32).reshape(75, 75) for band in train["band_2"]])
X_train = np.concatenate([X_band_1[:, :, :, np.newaxis], X_band_2[:, :, :, np.newaxis], ((
    X_band_1 + X_band_2) / 2)[:, :, :, np.newaxis]], axis=-1)

model = MyNet()

target_train = train['is_iceberg']
X_train_cv, X_valid, y_train_cv, y_valid = train_test_split(
    X_train, target_train, random_state=1, train_size=0.75)

#   model.fit(X_train_cv, y_train_cv, batch_size=batch_size, epochs=epochs,
#            verbose=1, validation_data=(X_valid, y_valid), callbacks=callbacks)


if predict:
    model.load_weights(filepath=file_path)
    
    score = model.evaluate(X_valid, y_valid, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    exit()

    test = pd.read_json("input\\test.json")    
    
    X_band_test_1 = np.array([np.array(band).astype(
        np.float32).reshape(75, 75) for band in test["band_1"]])
    X_band_test_2 = np.array([np.array(band).astype(
        np.float32).reshape(75, 75) for band in test["band_2"]])
    X_test = np.concatenate([X_band_test_1[:, :, :, np.newaxis], X_band_test_2[:, :, :, np.newaxis], ((
        X_band_test_1 + X_band_test_2) / 2)[:, :, :, np.newaxis]], axis=-1)
    predicted_test = model.predict_proba(X_test)

    submission = pd.DataFrame()
    submission['id']=test['id']
    submission['is_iceberg']=predicted_test.reshape((predicted_test.shape[0]))
    submission.to_csv('sub.csv', index=False)