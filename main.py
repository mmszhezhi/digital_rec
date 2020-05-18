import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Dense,Activation,Lambda,Flatten,Conv2D,BatchNormalization,Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

class helloworld:
    def __init__(self):
        print("init")
        self.model = None
        # self.load()
        self.datagen = ImageDataGenerator(
            rotation_range=10,
            zoom_range=0.10,
            width_shift_range=0.1,
            height_shift_range=0.1)
        self.build_model()
        print("hello ")



    def load(self):
        self.train = pd.read_csv("train.csv")
        self.test = pd.read_csv("test.csv")
    def build_model(self):
        nets = 15
        model = [0] * nets
        for j in range(nets):
            model[j] = Sequential()

            model[j].add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
            model[j].add(BatchNormalization())
            model[j].add(Conv2D(32, kernel_size=3, activation='relu'))
            model[j].add(BatchNormalization())
            model[j].add(Conv2D(32, kernel_size=5, strides=2, padding='same', activation='relu'))
            model[j].add(BatchNormalization())
            model[j].add(Dropout(0.4))

            model[j].add(Conv2D(64, kernel_size=3, activation='relu'))
            model[j].add(BatchNormalization())
            model[j].add(Conv2D(64, kernel_size=3, activation='relu'))
            model[j].add(BatchNormalization())
            model[j].add(Conv2D(64, kernel_size=5, strides=2, padding='same', activation='relu'))
            model[j].add(BatchNormalization())
            model[j].add(Dropout(0.4))

            model[j].add(Conv2D(128, kernel_size=4, activation='relu'))
            model[j].add(BatchNormalization())
            model[j].add(Flatten())
            model[j].add(Dropout(0.4))
            model[j].add(Dense(10, activation='softmax'))

            # COMPILE WITH ADAM OPTIMIZER AND CROSS ENTROPY COST
            model[j].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        print(model)
        self.model = model
    def train(self):
        # DECREASE LEARNING RATE EACH EPOCH
        nets = 10
        annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)
        # TRAIN NETWORKS
        history = [0] * nets
        epochs = 45
        train = pd.read_csv("train.csv")
        test = pd.read_csv("test.csv")
        # PREPARE DATA FOR NEURAL NETWORK
        Y_train = train["label"]
        X_train = train.drop(labels=["label"], axis=1)
        X_train = X_train / 255.0
        X_test = test / 255.0
        X_train = X_train.values.reshape(-1, 28, 28, 1)
        X_test = X_test.values.reshape(-1, 28, 28, 1)
        Y_train = to_categorical(Y_train, num_classes=10)



        test = pd.read_csv("test.csv")
        for j in range(nets):
            X_train2, X_val2, Y_train2, Y_val2 = train_test_split(X_train, Y_train, test_size=0.1)
            history[j] = self.model[j].fit_generator(self.datagen.flow(X_train2, Y_train2, batch_size=64),
                                                epochs=epochs, steps_per_epoch=X_train2.shape[0] // 64,
                                                validation_data=(X_val2, Y_val2), callbacks=[annealer], verbose=0)
            print("CNN {0:d}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(
                j + 1, epochs, max(history[j].history['acc']), max(history[j].history['val_acc'])))

if __name__ == '__main__':
    h = helloworld()
    h.train()
