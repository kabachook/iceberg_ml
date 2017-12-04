import os
os.environ[
    'PATH'] = f'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\bin;{os.environ["PATH"]}'

import pandas as pd
import numpy as np
from model import MyNetv2, file_path

model = MyNetv2()

model.load_weights(filepath=file_path)

test = pd.read_json("input\\test.json")

X_band_test_1 = np.array([np.array(band).astype(
    np.float32).reshape(75, 75) for band in test["band_1"]])
X_band_test_2 = np.array([np.array(band).astype(
    np.float32).reshape(75, 75) for band in test["band_2"]])
X_test = np.concatenate([X_band_test_1[:, :, :, np.newaxis], X_band_test_2[:, :, :, np.newaxis], ((
    X_band_test_1 + X_band_test_2) / 2)[:, :, :, np.newaxis]], axis=-1)

X_test_angle = test.inc_angle.replace('na', 0).astype(float).fillna(0.0)

prediction = model.predict(
    [X_test, X_test_angle], verbose=1, batch_size=32)
print('Submitting')
submission = pd.DataFrame(
    {'id': test["id"], 'is_iceberg': prediction.reshape((prediction.shape[0]))})
submission.to_csv('sub.csv', index=False)
