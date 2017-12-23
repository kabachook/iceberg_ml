from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, Activation, concatenate, AveragePooling2D, GlobalAveragePooling2D, Cropping2D
from keras.layers import GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras import initializers, applications
from keras.optimizers import Adam, Adagrad
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, TensorBoard, LearningRateScheduler
import keras.backend as K


def MyNet(opt):
    # model = Sequential()
    # # model.add(BatchNormalization())

    # model.add(Conv2D(
    #     64, kernel_size=(3, 3), input_shape=(75, 75, 3)))
    # model.add(Activation('relu'))
    # model.add(BatchNormalization())
    # model.add(MaxPooling2D(
    #     pool_size=(3, 3), strides=(2, 2)))
    # model.add(Dropout(0.2))

    # model.add(Conv2D(128, kernel_size=(3, 3)))
    # model.add(Activation('relu'))
    # model.add(BatchNormalization())
    # model.add(MaxPooling2D(
    #     pool_size=(2, 2), strides=(2, 2)))
    # model.add(Dropout(0.2))

    # model.add(Conv2D(128, kernel_size=(3, 3)))
    # model.add(Activation('relu'))
    # model.add(BatchNormalization())
    # model.add(MaxPooling2D(
    #     pool_size=(2, 2), strides=(2, 2)))
    # model.add(Dropout(0.3))

    # model.add(Conv2D(64, kernel_size=(3, 3)))
    # model.add(Activation('relu'))
    # model.add(BatchNormalization())
    # model.add(MaxPooling2D(
    #     pool_size=(2, 2), strides=(2, 2)))
    # model.add(Dropout(0.3))

    # model.add(Flatten())

    # angle_input = Input(shape=(1,))
    # angle_input = BatchNormalization()(angle_input)

    # model.add(concatenate([model, angle_input]))

    # model.add(Dense(512))
    # model.add(Activation('relu'))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.2))

    # model.add(Dense(256))
    # model.add(Activation('relu'))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.2))

    # model.add(Dense(1))
    # model.add(Activation('sigmoid'))

    # mypotim = Adam(lr=0.01, decay=0.0)
    # model.compile(inputs=[model, angle_input], outputs=model, loss='binary_crossentropy',
    #               optimizer=mypotim, metrics=['accuracy'])

    # model.summary()
    img_input = Input(shape=(75, 75, 3))
    angle_input = Input(shape=(1,))

    model = Conv2D(
        64, kernel_size=(3, 3), input_shape=(75, 75, 3))(img_input)
    model = Activation('relu')(model)
    model = BatchNormalization()(model)
    model = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2))(model)
    model = Dropout(0.4)(model)

    model = Conv2D(128, kernel_size=(3, 3))(model)
    model = Activation('relu')(model)
    model = BatchNormalization()(model)
    model = MaxPooling2D(
        pool_size=(2, 2), strides=(2, 2))(model)
    model = Dropout(0.4)(model)

    model = Conv2D(128, kernel_size=(3, 3))(model)
    model = Activation('relu')(model)
    model = BatchNormalization()(model)
    model = MaxPooling2D(
        pool_size=(2, 2), strides=(2, 2))(model)
    model = Dropout(0.5)(model)

    model = Conv2D(64, kernel_size=(3, 3))(model)
    model = Activation('relu')(model)
    model = BatchNormalization()(model)
    model = MaxPooling2D(
        pool_size=(2, 2), strides=(2, 2))(model)
    model = Dropout(0.5)(model)

    model = Flatten()(model)

    model = concatenate([model, angle_input])

    model = Dense(512)(model)
    model = Activation('relu')(model)
    model = BatchNormalization()(model)
    model = Dropout(0.5)(model)

    model = Dense(256)(model)
    model = Activation('relu')(model)
    model = BatchNormalization()(model)
    model = Dropout(0.5)(model)

    model = Dense(1)(model)
    model = Activation('sigmoid')(model)

    # mypotim = Adam(lr=0.01, decay=0.0)
    final_model = Model(inputs=[img_input, angle_input], outputs=model)
    final_model.compile(loss='binary_crossentropy',
                        optimizer=opt, metrics=['accuracy'])

    final_model.summary()

    return final_model


def MyNetv2(activation='relu'):
    pic_input = Input(shape=(75, 75, 3))
    ang_input = Input(shape=(1,))

    cnn = Cropping2D(20)(pic_input)
    cnn = BatchNormalization()(cnn)
    cnn = Conv2D(32, (3, 3), activation=activation)(cnn)
    # cnn = AveragePooling2D()(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = Conv2D(64, (3, 3), activation=activation)(cnn)
    cnn = BatchNormalization()(cnn)
    # cnn = Conv2D(128, (5, 5), activation=activation)(cnn)
    # cnn = BatchNormalization()(cnn)
    # cnn = Conv2D(256, (5, 5), activation=activation)(cnn)
    # cnn = BatchNormalization()(cnn)
    # cnn = Conv2D(512, (1, 1), activation=activation)(cnn)
    # cnn = BatchNormalization()(cnn)
    cnn = MaxPooling2D()(cnn)
    cnn = Conv2D(512, (9, 9), activation=activation)(cnn)

    cnn = GlobalAveragePooling2D()(cnn)
    # cnn = Flatten()(cnn)
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
    opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.001)
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
        optimizer=Adam(lr=0.0005), loss='binary_crossentropy', metrics=['accuracy'])
    return model_final


def get_callbacks(filepath, patience=2, early_stopping=False):
    es = EarlyStopping('val_loss', patience=patience, mode="auto")
    checkpoint = ModelCheckpoint(filepath, save_best_only=True)
    tb = TensorBoard(log_dir='./logs', batch_size=32)
    if early_stopping:
        return [es, checkpoint, tb]
    return [checkpoint, tb]


file_path = "model_weights.hdf5"
callbacks = get_callbacks(filepath=file_path, patience=50, early_stopping=True)

if __name__ == "__main__":
    MyNet()
