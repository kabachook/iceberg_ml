from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, Activation, concatenate, AveragePooling2D, GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras import initializers, applications
from keras.optimizers import Adam, Adagrad
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, TensorBoard
import keras.backend as K


def MyNet():
    model = Sequential()

    model.add(BatchNormalization(input_shape=(75, 75, 3)))

    # Input 75x75x3
    # Layer 1
    model.add(Conv2D(256, (5, 5), activation='relu', input_shape=(75, 75, 3)))
    model.add(Conv2D(128, (5, 5), activation='relu'))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    # Layer 2
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    # Layer 3
    model.add(BatchNormalization())
    model.add(Conv2D(64, (1, 1), activation='relu'))
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-8)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt, metrics=['accuracy'])
    model.summary()
    return model


def MyNetv2(activation='relu'):
    pic_input = Input(shape=(75, 75, 3))
    ang_input = Input(shape=(1,))

    cnn = BatchNormalization()(pic_input)
    cnn = Conv2D(32, (3, 3), activation=activation)(cnn)
    cnn = AveragePooling2D()(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = Conv2D(64, (3, 3), activation=activation)(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = Conv2D(128, (5, 5), activation=activation)(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = Conv2D(256, (5, 5), activation=activation)(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = Conv2D(512, (1, 1), activation=activation)(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = MaxPooling2D()(cnn)
    cnn = Conv2D(512, (9, 9), activation=activation)(cnn)

    # cnn = GlobalAveragePooling2D()(cnn)
    cnn = Flatten()(cnn)
    cnn = concatenate([cnn, ang_input])
    cnn = BatchNormalization()(cnn)
    cnn = Dense(2048, activation=activation)(cnn)
    cnn = Dropout(0.5)(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = Dense(1024, activation=activation)(cnn)
    cnn = Dropout(0.5)(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = Dense(512, activation=activation)(cnn)
    cnn = Dropout(0.5)(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = Dense(256, activation=activation)(cnn)
    cnn = Dropout(0.5)(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = Dense(128, activation=activation)(cnn)
    cnn = Dropout(0.5)(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = Dense(1, activation='sigmoid')(cnn)

    simple_cnn = Model(inputs=[pic_input, ang_input], outputs=cnn)
    # opt = Adagrad(lr=0.001,epsilon=1e-6,decay=0.)
    opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.)
    simple_cnn.compile(
        optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    simple_cnn.summary()
    return simple_cnn


def MyNetv3(img_width, img_height):
    vgg = applications.VGG19(weights='imagenet', include_top=False,
                             input_shape=(img_width, img_height, 3), pooling=None)

    ang_input = Input(shape=(1,))
    model = vgg.output
    model = Flatten()(model)
    model = concatenate([model, ang_input])
    model = BatchNormalization()(model)
    model = Dense(1024, activation='relu')(model)
    model = Dropout(0.5)(model)
    model = BatchNormalization()(model)
    model = Dense(512, activation='relu')(model)
    model = Dropout(0.5)(model)
    model = BatchNormalization()(model)
    model = Dense(256, activation='relu')(model)
    model = Dropout(0.5)(model)
    prediction = Dense(1, activation='sigmoid')(model)

    model_final = Model(inputs=[vgg.input, ang_input], outputs=prediction)
    model_final.summary()
    model_final.compile(
        optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model_final


def get_callbacks(filepath, patience=2, early_stopping=False):
    callbacks = []
    if early_stopping:
        callbacks.append(EarlyStopping(
            'val_loss', patience=patience, mode="auto"))
    callbacks.append(ModelCheckpoint(filepath, save_best_only=True))
    callbacks.append(TensorBoard(log_dir='./logs', batch_size=32))
    return callbacks


file_path = ".model_weights.hdf5"
callbacks = get_callbacks(filepath=file_path, patience=20, early_stopping=True)

if __name__ == "__main__":
    MyNetv2()
    # MyNetv3(75, 75)
