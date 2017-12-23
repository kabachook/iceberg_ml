import os
os.environ[
    'PATH'] = f'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\bin;{os.environ["PATH"]}'

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LeakyReLU, Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard, LearningRateScheduler
from keras.optimizers import Adam
from keras.models import model_from_json
from sklearn.model_selection import train_test_split
from scipy.ndimage import rotate as rot
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


from model import file_path, MyNetv2, MyNetv3, MyNet


# Constants
batch_size = 128
epochs = 100
predict = False

seed = 123124
np.random.seed(seed)

submit_only = False

train_file = os.path.join(os.getcwd(), "./input/train.json")
test_file = os.path.join(os.getcwd(), "./input/test.json")
model_file = os.path.join(os.getcwd(), "./model_checkpoint.hdf5")
model_json_file = os.path.join(os.getcwd(), "./model.json")
logs_dir = os.path.join(os.getcwd(), "./logs/")




class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        plt.ion()
        plt.show()
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []

        self.fig = plt.figure()

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):

        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1

        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        # plt.legend()
        plt.draw()
        plt.pause(0.01)


def transform(df):
    images = []
    for i, row in df.iterrows():
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)
        band_3 = band_1 + band_2

        band_1_norm = (band_1 - band_1.mean()) / (band_1.max() - band_1.min())
        band_2_norm = (band_2 - band_2. mean()) / (band_2.max() - band_2.min())
        band_3_norm = (band_3 - band_3.mean()) / (band_3.max() - band_3.min())

        images.append(np.dstack((band_1_norm, band_2_norm, band_3_norm)))

    return np.array(images)


def augment(images):
    image_mirror_lr = []
    image_mirror_ud = []
    image_rotate = []
    for i in range(0, images.shape[0]):
        band_1 = images[i, :, :, 0]
        band_2 = images[i, :, :, 1]
        band_3 = images[i, :, :, 2]

        # mirror left-right
        band_1_mirror_lr = np.flip(band_1, 0)
        band_2_mirror_lr = np.flip(band_2, 0)
        band_3_mirror_lr = np.flip(band_3, 0)
        image_mirror_lr.append(
            np.dstack((band_1_mirror_lr, band_2_mirror_lr, band_3_mirror_lr)))

        # mirror up-down
        band_1_mirror_ud = np.flip(band_1, 1)
        band_2_mirror_ud = np.flip(band_2, 1)
        band_3_mirror_ud = np.flip(band_3, 1)
        image_mirror_ud.append(
            np.dstack((band_1_mirror_ud, band_2_mirror_ud, band_3_mirror_ud)))

        # rotate
        band_1_rotate = rot(band_1, 30, reshape=False)
        band_2_rotate = rot(band_2, 30, reshape=False)
        band_3_rotate = rot(band_3, 30, reshape=False)
        image_rotate.append(
            np.dstack((band_1_rotate, band_2_rotate, band_3_rotate)))

    mirrorlr = np.array(image_mirror_lr)
    mirrorud = np.array(image_mirror_ud)
    rotated = np.array(image_rotate)
    images = np.concatenate((images, mirrorlr, mirrorud, rotated))
    return images


if not submit_only:
    data = pd.read_json(train_file)
    data.inc_angle = data.inc_angle.map(lambda x: 0.0 if x == "na" else x)

    train_X = transform(data)
    train_y = np.array(data['is_iceberg'])

    indx_tr = np.where(data.inc_angle > 0)

    train_y = train_y[indx_tr[0]]
    train_X = train_X[indx_tr[0], ...]
    train_angle = data.inc_angle[indx_tr[0]]

    train_X = augment(train_X)
    train_y = np.concatenate((train_y, train_y, train_y, train_y))
    train_angle = np.concatenate(
        (train_angle, train_angle, train_angle, train_angle))

    print(len(train_X), batch_size)

    lr = 0.01
    opt = Adam(lr=0.01,decay=lr/epochs)
    model = MyNet(opt)
    callbacks = [ModelCheckpoint(model_file, save_best_only=True),
                 ReduceLROnPlateau(),
                 #PlotLosses(),
                 TensorBoard(log_dir=logs_dir,batch_size=batch_size,write_images=True),]
                 #LearningRateScheduler(schedule)]
    try:
        history = model.fit(x=[train_X, train_angle], y=train_y, epochs=epochs, verbose=1, validation_split=0.125,
                        callbacks=callbacks, batch_size=batch_size)
    except KeyboardInterrupt:
        pass

    plt.show()
    

    print(history.history.keys())
    fig = plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()

    model_json = model.to_json()
    with open(model_json_file, "w") as json_file:
        json_file.write(model_json)

# load json and create model
json_file = open(
    model_json_file, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(
    model_file)
print("Loaded model from disk")
loaded_model.compile(loss='binary_crossentropy',
                     optimizer=Adam(lr=0.01, decay=0), metrics=['accuracy'])

test = pd.read_json(test_file)
test.inc_angle = test.inc_angle.replace('na', 0)
test_X = transform(test)
print(test_X.shape)

pred_test = loaded_model.predict(test_X, verbose=1)
submission = pd.DataFrame(
    {'id': test["id"], 'is_iceberg': pred_test.reshape((pred_test.shape[0]))})
submission.to_csv(
    'submission.csv', index=False)


# def gen_flow_for_two_inputs(X1, X2, y, gen):
#     genX1 = gen.flow(X1, y,  batch_size=batch_size, seed=seed)
#     genX2 = gen.flow(X1, X2, batch_size=batch_size, seed=seed)
#     while True:
#         X1i = genX1.next()
#         X2i = genX2.next()
#         # Assert arrays are equal - this was for peace of mind, but slows down training
#         # np.testing.assert_array_equal(X1i[0],X2i[0])
#         yield [X1i[0], X2i[1]], X1i[1]


# # Add band 3 = (band_1 + band_2)/2
# x_band_1 = np.array([np.array(band).astype(
#     np.float32).reshape(75, 75) for band in data["band_1"]])
# x_band_2 = np.array([np.array(band).astype(
#     np.float32).reshape(75, 75) for band in data["band_2"]])
# x_bands = np.concatenate([x_band_1[:, :, :, np.newaxis], x_band_2[:, :, :, np.newaxis], ((
#     x_band_1 + x_band_2) / 2)[:, :, :, np.newaxis]], axis=-1)
# data_angle = np.array(data.inc_angle)
# data_angle = data_angle / np.max(data_angle)


# # Split data
# data_y = data['is_iceberg']
# x_train, x_test, x_train_angle, x_test_angle,  y_train, y_test = train_test_split(
#     x_bands, data_angle, data_y, random_state=seed, train_size=0.75)

# # Image generator
# augmentation = ImageDataGenerator(horizontal_flip=True,
#                                   vertical_flip=True,
#                                   width_shift_range=0.1,
#                                   height_shift_range=0.1,
#                                   zoom_range=0.1,
#                                   rotation_range=360,
#                                   )

# gen_flow = gen_flow_for_two_inputs(
#     x_train, x_train_angle, y_train, augmentation)

# model = MyNetv2()
# model.fit_generator(gen_flow, epochs=epochs,
#                     verbose=1, validation_data=([x_test, x_test_angle], y_test), callbacks=callbacks,
#                     steps_per_epoch=len(x_train) / batch_size)


# model.load_weights(filepath=file_path)  # Load best result
# score = model.evaluate([x_test, x_test_angle], y_test, verbose=1)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
