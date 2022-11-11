import matplotlib.pyplot as plt 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, DepthwiseConv2D, SpatialDropout2D, RandomRotation, Flatten, Rescaling, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Activation, ZeroPadding2D
from tensorflow import keras

def newModel():
    model = Sequential()
    model.add(RandomRotation(factor=(-0.3,0.3), fill_mode='constant', fill_value=0.0, input_shape=(224,224,3)))
    model.add(Rescaling(scale=1./255))
    model.add(BatchNormalization())

    model.add(DepthwiseConv2D(depth_multiplier=25, kernel_size=7, kernel_regularizer=keras.regularizers.L1(l1=1e-5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters=20, kernel_size=4, kernel_regularizer=keras.regularizers.L1(l1=1e-4), padding='same'))
    model.add(SpatialDropout2D(0.2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters=25, kernel_size=4, kernel_regularizer=keras.regularizers.L1(l1=1e-4), padding='same')) 
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters=1, kernel_size=4, kernel_regularizer=keras.regularizers.L1(l1=1e-5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(ZeroPadding2D(2))

    model.add(Conv2D(filters=20, kernel_size=4, kernel_regularizer=keras.regularizers.L1(l1=1e-5), padding='same'))
    model.add(SpatialDropout2D(0.2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(filters=25, kernel_size=4, kernel_regularizer=keras.regularizers.L1(l1=1e-4), padding='same')) 
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(filters=20, kernel_size=4, kernel_regularizer=keras.regularizers.L1(l1=1e-5), padding='same'))
    model.add(SpatialDropout2D(0.2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(filters=25, kernel_size=4, kernel_regularizer=keras.regularizers.L1(l1=1e-5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(filters=25, kernel_size=4, kernel_regularizer=keras.regularizers.L1(l1=1e-4), padding='same'))
    model.add(SpatialDropout2D(0.2))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(2))

    model.add(Conv2D(filters=30, kernel_size=4, kernel_regularizer=keras.regularizers.L1(l1=1e-5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(2)) 

    model.add(Conv2D(filters=35, kernel_size=4, kernel_regularizer=keras.regularizers.L1(l1=1e-4), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(2))

    model.add(Flatten())
    model.add(Dense(units=784, activation='tanh'))
    model.add(Dropout(0.15)) 
    model.add(Dense(units=200, activation='tanh'))
    model.add(Dropout(0.15))
    model.add(Dense(units=5, activation='softmax'))

    return model

def visualiz(history, epochs, num):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(10,7))
    plt.plot(range(epochs), acc, label='Точность на обучении')
    plt.plot(range(epochs), val_acc, label='Точность на валидации')
    plt.plot(range(epochs), loss, label='Ошибка на обучении')
    plt.plot(range(epochs), val_loss, label='Ошибка на валидации')
    plt.legend(loc='lower left')
    plt.title('График обучения')

    plt.savefig(f'/content/drive/MyDrive/history{num}.png')
    plt.show()