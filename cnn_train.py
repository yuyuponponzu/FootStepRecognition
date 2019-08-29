import keras
import numpy as np
import os
import random
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn import model_selection
from sklearn import preprocessing
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation
from keras.layers import Conv2D, GlobalAveragePooling2D
from keras.layers import BatchNormalization, Add
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
import argparse

parser = argparse.ArgumentParser(description='CNNの学習を行うプログラム')
parser.add_argument('--aug', '-a', required=True,choices=['yes', 'no'] help='Augmentationありのデータを使うか否か')
parser.add_argument('--type', '-t', required=True,choices=['kutusita_npz_data/', 'slip_npz_data/'] help='靴下とスリッパどっちを使っているか')
args = parser.parse_args()
augkey = args.aug

if augkey == "yes":
    # dataset files
    train_files = [args.type+"esc_melsp_train_raw.npz",
                   args.type+"esc_melsp_train_ss.npz",
                   args.type+"esc_melsp_train_st.npz",
                   args.type+"esc_melsp_train_wn.npz",
                   args.type+"esc_melsp_train_com.npz"]
    test_file = args.type+"esc_melsp_test.npz"

else :
    # dataset files
    train_files = [args.type+"esc_melsp_train_raw.npz"]
    test_file = args.type+"esc_melsp_test.npz"

n = [0 for i in range(len(train_files)+1)]
m = 0
for i in range(len(train_files)):
    (a, b, c) = np.load(train_files[i])["x"].shape
    n[i+1] =n[i] + a
    m = m + a
freq = b
time = c
train_num = m

# define dataset placeholders
x_train = np.zeros(freq*time*train_num).reshape(train_num, freq, time)
y_train = np.zeros(train_num)

# load dataset
for i in range(len(train_files)):
    data = np.load(train_files[i])
    x_train[n[i]:n[i+1]] = data["x"]
    y_train[n[i]:n[i+1]] = data["y"]

# load test dataset
test_data = np.load(test_file)
x_test = test_data["x"]
y_test = test_data["y"]
test_num = len(y_test)

# redefine target data into one hot vector
classes = 4
y_train = keras.utils.to_categorical(y_train, classes)
y_test = keras.utils.to_categorical(y_test, classes)

# reshape training dataset
x_train = x_train.reshape(train_num, freq, time, 1)
x_test = x_test.reshape(test_num, freq, time, 1)

print("x train:{0}\ny train:{1}\nx test:{2}\ny test:{3}".format(x_train.shape,
                                                                y_train.shape, 
                                                                x_test.shape, 
                                                                y_test.shape))

# ## Define convolutional neural network
def cba(inputs, filters, kernel_size, strides):
    x = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

# define CNN
inputs = Input(shape=(x_train.shape[1:]))

x_1 = cba(inputs, filters=32, kernel_size=(1,8), strides=(1,2))
x_1 = cba(x_1, filters=32, kernel_size=(8,1), strides=(2,1))
x_1 = cba(x_1, filters=64, kernel_size=(1,8), strides=(1,2))
x_1 = cba(x_1, filters=64, kernel_size=(8,1), strides=(2,1))

x_2 = cba(inputs, filters=32, kernel_size=(1,16), strides=(1,2))
x_2 = cba(x_2, filters=32, kernel_size=(16,1), strides=(2,1))
x_2 = cba(x_2, filters=64, kernel_size=(1,16), strides=(1,2))
x_2 = cba(x_2, filters=64, kernel_size=(16,1), strides=(2,1))

x_3 = cba(inputs, filters=32, kernel_size=(1,32), strides=(1,2))
x_3 = cba(x_3, filters=32, kernel_size=(32,1), strides=(2,1))
x_3 = cba(x_3, filters=64, kernel_size=(1,32), strides=(1,2))
x_3 = cba(x_3, filters=64, kernel_size=(32,1), strides=(2,1))

x_4 = cba(inputs, filters=32, kernel_size=(1,64), strides=(1,2))
x_4 = cba(x_4, filters=32, kernel_size=(64,1), strides=(2,1))
x_4 = cba(x_4, filters=64, kernel_size=(1,64), strides=(1,2))
x_4 = cba(x_4, filters=64, kernel_size=(64,1), strides=(2,1))

x = Add()([x_1, x_2, x_3, x_4])

x = cba(x, filters=128, kernel_size=(1,16), strides=(1,2))
x = cba(x, filters=128, kernel_size=(16,1), strides=(2,1))

x = GlobalAveragePooling2D()(x)
x = Dense(classes)(x)
x = Activation("softmax")(x)

model = Model(inputs, x)
model.summary()


# ## Optimization and callbacks
# initiate Adam optimizer
opt = keras.optimizers.adam(lr=0.00001, decay=1e-6)#, amsgrad=True)

# Let's train the model using Adam with amsgrad
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

if augkey == "yes":
    # directory for model checkpoints
    model_dir = "./cnn_models_aug"
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
else:
    model_dir = "./cnn_models"
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

# early stopping and model checkpoint# early  
#es_cb = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
chkpt = os.path.join(model_dir, 'esc50_.{epoch:02d}_{val_loss:.4f}_{val_acc:.4f}.hdf5')
cp_cb = ModelCheckpoint(filepath = chkpt, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

# ## Train CNN model with between class dataset
# between class data generator
class MixupGenerator():
    def __init__(self, x_train, y_train, batch_size=16, alpha=0.2, shuffle=True):
        self.x_train = x_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.sample_num = len(x_train)

    def __call__(self):
        while True:
            indexes = self.__get_exploration_order()
            itr_num = int(len(indexes) // (self.batch_size * 2))

            for i in range(itr_num):
                batch_ids = indexes[i * self.batch_size * 2:(i + 1) * self.batch_size * 2]
                x, y = self.__data_generation(batch_ids)

                yield x, y

    def __get_exploration_order(self):
        indexes = np.arange(self.sample_num)

        if self.shuffle:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, batch_ids):
        _, h, w, c = self.x_train.shape
        _, class_num = self.y_train.shape
        x1 = self.x_train[batch_ids[:self.batch_size]]
        x2 = self.x_train[batch_ids[self.batch_size:]]
        y1 = self.y_train[batch_ids[:self.batch_size]]
        y2 = self.y_train[batch_ids[self.batch_size:]]
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        x_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l.reshape(self.batch_size, 1)

        x = x1 * x_l + x2 * (1 - x_l)
        y = y1 * y_l + y2 * (1 - y_l)

        return x, y

# train model
batch_size = 16
epochs = 50

training_generator = MixupGenerator(x_train, y_train)()
model.fit_generator(generator=training_generator,
                    steps_per_epoch=x_train.shape[0] // batch_size,
                    validation_data=(x_test, y_test),
                    epochs=epochs, 
                    verbose=1,
                    shuffle=True,
                    callbacks=[cp_cb])



# ## Evaluate model

#model = load_model("./cnn_models/asioto01.hdf5")

#evaluation = model.evaluate(x_test, y_test)
#print(evaluation)

