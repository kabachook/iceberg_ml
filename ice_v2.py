import pandas as pd
import numpy as np
import cv2  # Used to manipulated the images
seed = 251736
# The seed I used - pick your own or comment out for a random seed. A constant seed allows for better comparisons though
np.random.seed(seed)

# Kfold
from sklearn.model_selection import StratifiedKFold

# Import Keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, MaxPool2D, GlobalMaxPooling2D, AvgPool2D
from keras.layers import Input, concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1, l2, l1_l2
from keras.optimizers import Adam
from keras.applications import VGG16

import os


def get_scaled_imgs(df):
    """
    basic function for reshaping and rescaling data as images
    """
    imgs = []

    for i, row in df.iterrows():
        # make 75x75 image
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)
        band_3 = band_1 + band_2  # plus since log(x*y) = log(x) + log(y)

        # Rescale
        a = (band_1 - band_1.mean()) / (band_1.max() - band_1.min())
        b = (band_2 - band_2.mean()) / (band_2.max() - band_2.min())
        c = (band_3 - band_3.mean()) / (band_3.max() - band_3.min())

        imgs.append(np.dstack((a, b, c)))

    return np.array(imgs)


def central_crop(imgs, cropsize):
    """
    central crop for images
    """

    x = imgs.shape[1]
    y = imgs.shape[2]
    startx = x // 2 - (cropsize // 2)
    starty = y // 2 - (cropsize // 2)
    return imgs[:, startx:startx + cropsize, starty:starty + cropsize, :]


def get_more_images(imgs):
    """
    augmentation for more data
    """

    more_images = []
    vert_flip_imgs = []
    hori_flip_imgs = []

    # Central crop
    # imgs = central_crop(imgs, cropsize)

    for i in range(0, imgs.shape[0]):
        a = imgs[i, :, :, 0]
        b = imgs[i, :, :, 1]
        c = imgs[i, :, :, 2]

        av = cv2.flip(a, 1)
        ah = cv2.flip(a, 0)
        bv = cv2.flip(b, 1)
        bh = cv2.flip(b, 0)
        cv = cv2.flip(c, 1)
        ch = cv2.flip(c, 0)

        vert_flip_imgs.append(np.dstack((av, bv, cv)))
        hori_flip_imgs.append(np.dstack((ah, bh, ch)))

    v = np.array(vert_flip_imgs)
    h = np.array(hori_flip_imgs)

    more_images = np.concatenate((imgs, v, h))

    return more_images


def get_model():
    """
    Keras Sequential model

    """
    model = Sequential()

    # Conv block 1
    model.add(Conv2D(64, kernel_size=(3, 3),
                     activation='relu', input_shape=(75, 75, 3)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    # Conv block 2
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Conv block 3
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Conv block 4
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Flatten before dense
    model.add(Flatten())

    # Dense 1
    model.add(Dense(1024, activation='relu', kernel_regularizer=l2()))
    model.add(Dropout(0.4))

    # Dense 2
    model.add(Dense(512, activation='relu', kernel_regularizer=l2()))
    model.add(Dropout(0.4))

    # Dense 3
    model.add(Dense(256, activation='relu', kernel_regularizer=l2()))
    model.add(Dropout(0.2))

    # Output
    model.add(Dense(1, activation="sigmoid"))

    optimizer = Adam(lr=0.0001, decay=0.0)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'],
                  )

    return model


def get_model2():
    """
    Keras Sequential model

    """
    model = Sequential()

    # Conv block 1
    model.add(Conv2D(32, kernel_size=(7, 7),
                     activation='relu', input_shape=(75, 75, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Conv block 2
    model.add(Conv2D(64, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Conv block 3
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Flatten before dense
    model.add(Flatten())

    # Dense 1
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))

    # Dense 2
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    # Dense 3
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    # Output
    model.add(Dense(1, activation="sigmoid"))

    optimizer = Adam(lr=0.0001, decay=0.0)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'],
                  )

    return model


def get_model3():
    inputs = Input((75, 75, 3))
    conv1 = Conv2D(64, (9, 9), padding='valid', activation='elu')(inputs)
    conv1 = BatchNormalization(momentum=0.99)(conv1)
    pool1 = MaxPool2D(pool_size=(2, 2))(conv1)
    drop1 = Dropout(0.1)(pool1)

    conv2 = Conv2D(64, (5, 5), padding='valid', activation='elu')(drop1)
    conv2 = BatchNormalization(momentum=0.95)(conv2)
    pool2 = AvgPool2D(pool_size=(2, 2))(conv2)
    drop2 = Dropout(0.1)(pool2)

    conv3 = Conv2D(64, (3, 3), padding='valid', activation='elu')(drop2)
    conv3 = BatchNormalization(momentum=0.95)(conv3)
    pool3 = MaxPool2D(pool_size=(2, 2))(conv3)
    drop3 = Dropout(0.1)(pool3)

    conv4 = Conv2D(64, (3, 3), padding='valid', activation='elu')(drop3)
    pool4 = AvgPool2D(pool_size=(2, 2))(conv4)

    gp = GlobalMaxPooling2D()(pool4)

    out = Dense(1, activation='sigmoid')(gp)
    optimizer = Adam(lr=1e-3, decay=1e-5)
    model = Model(input=inputs, outputs=out)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer, metrics=['accuracy'])
    return model


def get_model_vgg16_pretrained(input_shape=(75, 75, 3)):
    dropout = 0.25
    optimizer = Adam(lr=1e-3, decay=1e-5)
    # optimizer = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    #Building the model

    base_model = VGG16(weights='imagenet', include_top=False, 
                 input_shape=input_shape, classes=1)

    # input_meta = Input(shape=[inputs_meta], name='meta')
    # input_meta_norm = BatchNormalization()(input_meta)

    x = base_model.get_layer('block5_pool').output

    x = GlobalMaxPooling2D()(x)
    # concat = concatenate([x, input_meta_norm], name='features_layer')
    fc1 = Dense(512, activation='relu', name='fc2')(x)
    fc1 = Dropout(dropout)(fc1)
    fc2 = Dense(512, activation='relu', name='fc3')(fc1)
    fc2 = Dropout(dropout)(fc2)

    # Sigmoid Layer
    output = Dense(1)(fc2)
    output = Activation('sigmoid')(output)
    
    model = Model(inputs=[base_model.input], outputs=output)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model


# Training Data
df_train = pd.read_json('./input/train.json')  # this is a dataframe


Xtrain = get_scaled_imgs(df_train)
Ytrain = np.array(df_train['is_iceberg'])
df_train.inc_angle = df_train.inc_angle.replace('na', 0)
idx_tr = np.where(df_train.inc_angle > 0)

Ytrain = Ytrain[idx_tr[0]]
Xtrain = Xtrain[idx_tr[0], ...]
# Xangle = df_train.inc_angle[idx_tr[0]]

Xtr_more = get_more_images(Xtrain)
# Xangle_more = np.concatenate((Xangle, Xangle, Xangle, Xangle))
Ytr_more = np.concatenate((Ytrain, Ytrain, Ytrain))

# Test data
df_test = pd.read_json('./input/test.json')
df_test.inc_angle = df_test.inc_angle.replace('na', 0)
Xtest = (get_scaled_imgs(df_test))
# Xtest = central_crop(Xtest, 50)


expname = 'v6_5fold'
folds = 5
batch_size = 32
epochs = 50

dirs = {'logs': f'./logs/{expname}',
        'model': f'./checkpoints/{expname}', 'result': f'./result/{expname}'}
for i in dirs.values():
    if not os.path.exists(i):
        os.makedirs(i)

# K fold CV training
kfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
for fold_n, (train, test) in enumerate(kfold.split(Xtr_more, Ytr_more)):
    print(f"FOLD {fold_n}: ")
    # Since we use augmentation, angle changes on augmented images, so we do not pass angle to model
    model = get_model_vgg16_pretrained()
    

    MODEL_FILE = f'{dirs["model"]}/mdl_simple_k{fold_n}_wght.hdf5'

    mcp_save = ModelCheckpoint(
        MODEL_FILE, save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')
    tb = TensorBoard(log_dir=f'{dirs["logs"]}/{fold_n}')
    es = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='min')

    model.fit(Xtr_more[train], Ytr_more[train],
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(
                  Xtr_more[test], Ytr_more[test]),
              callbacks=[mcp_save, reduce_lr_loss, tb, es])

    model.load_weights(filepath=MODEL_FILE)

    score = model.evaluate(
        Xtr_more[test], Ytr_more[test], verbose=1)
    print('\n Val score:', score[0])
    print('\n Val accuracy:', score[1])

    SUBMISSION = f'{dirs["result"]}/sub_part{fold_n}.csv'

    pred_test = model.predict(Xtest)

    submission = pd.DataFrame(
        {'id': df_test["id"], 'is_iceberg': pred_test.reshape((pred_test.shape[0]))})
    print(submission.head(10))

    submission.to_csv(SUBMISSION, index=False)
    print("submission saved")



stacked = [pd.read_csv(dirs['result'] + f'/sub_part{i}.csv')
           for i in range(folds)]

sub = pd.DataFrame()
sub['id'] = stacked[1]['id']
sub['is_iceberg'] = np.exp(
    np.mean([i['is_iceberg'].apply(lambda x: np.log(x)) for i in stacked], axis=0))

sub.to_csv(dirs['result'] + '/final_ensemble.csv',
           index=False, float_format='%.6f')
