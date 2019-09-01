import argparse
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
from scipy.io.wavfile import read
from scipy.io.wavfile import write
from sklearn import model_selection
from sklearn import preprocessing
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation
from keras.layers import Conv2D, GlobalAveragePooling2D
from keras.layers import BatchNormalization, Add
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import json

def identify_outliers(ys):
    quartile_1, quartile_3 = np.percentile(ys, [25, 75])
    iqr = quartile_3 - quartile_1
    # 下限
    lower_bound = quartile_1 - (iqr * 1.5)
    # 上限
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.array(ys)[((ys > upper_bound) | (ys < lower_bound))]

#外れ値を除いた中で最長のlengthを算出
def lengthcheck(wavdir):
    length_list = []
    for c in os.listdir(wavdir):
        d = os.path.join(wavdir, c)
        if ".DS_Store" in d:
            print("This is .DS_Store. passing")
            continue
        elif ".npz" in d:
            print("This is npz file. passing.")
            continue
        elif ".json" in d:
            print("This is json. passing.")
            continue
        wavs = os.listdir(d)
        for i in [f for f in wavs if ('wav' in f)]:
            fs, x = read(os.path.join(d, i)) # wavファイルの読み込み            
            length_list.append(len(x))
    a = identify_outliers(length_list)
    length = a.max()
    return length

"""
def lengthcheck(wavdir):
    length = 0
    for c in os.listdir(wavdir):
        d = os.path.join(wavdir, c)
        wavs = os.listdir(d)
        for i in [f for f in wavs if ('wav' in f)]:
            fs, x = read(os.path.join(d, i)) # wavファイルの読み込み            
            if length <= len(x):
                length = len(x)
    return length
"""

def to_mono(x):
    wav_l = x[:, 0]
    wav_r = x[:, 1]
    xs = (0.5 * wav_l) + (0.5 * wav_r)
    return xs

def alignx(x, length):
    x = np.pad(x,(0,length-len(x)),"constant")
    return x

def calculate_melsp(x, n_fft=1024, hop_length=128):
    stft = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length))**2
    log_stft = librosa.power_to_db(stft)
    melsp = librosa.feature.melspectrogram(S=log_stft,n_mels=128)
    return melsp


def main(augkey, _, wavdir, testdir, type_, model_):
    #labelname = {"0":"andosan","1":"akagawakun","2":"hori","3":"huziisan"}
    wavdir = os.path.join(wavdir, type_)
    testdir = os.path.join(testdir, type_)
    f = open('./class.json', 'r')
    labelname = json.load(f)
    print('Loading dataset ...')
    train = []
    train2 = []
    labelarr = []
    labelarr2 = []
    label = 0
    length = lengthcheck(wavdir)
    for label,key in labelname.items():
        label = int(label)
        d = os.path.join(testdir, key)
        wavs = os.listdir(d)
        for i in [f for f in wavs if ('wav' in f)]:
            fs, x = read(os.path.join(d, i)) # wavファイルの読み込み
            if len(x[1]) != 1:
                x = to_mono(x) #ステレオからモノラルへ
            if length < len(x): 
                print("This is outlier. Excluding...")
                continue #外れ値を算出（大抵は１歩以上の足音が入ってる）
            x = np.asarray(alignx(x,length)).astype(np.float32)
            _x = calculate_melsp(x)
            freq, time = _x.shape
            print('   {} loaded'.format(i))
            train = [np.asarray(_x).astype(np.float32)] if train == [] else np.vstack((train, [np.asarray(_x).astype(np.float32)]))
            labelarr = label if labelarr == [] else np.vstack([labelarr, label])
        print(labelarr,label)
    labelarr = labelarr.flatten()
    print(labelarr)

    classes = len(labelname.items())
    y_test = keras.utils.to_categorical(labelarr, classes)
    print("vstackのところ未確認，train.shape",train.shape)
    x_test = train.reshape(len(train), freq, time, 1)
    print("適切かどうか未確認",x_test.shape)
    if augkey == "yes":
        model_dir = "./cnn_models_aug"
        model_f = os.path.join(model_dir, model_)
        model = load_model(model_f)
    else :
        model_dir = "./cnn_models"
        model_f = os.path.join(model_dir, model_)
        model = load_model(model_f)
    model.summary()
    evaluation = model.evaluate(x_test, y_test)
    print(evaluation)
    pred = model.predict(x_test)
    report = classification_report(labelarr, np.argmax(pred, axis=1), output_dict=True)
    #report = accuracy_score(labelarr,np.argmax(pred, axis=1))
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CNNで学習済みのモデルを使って評価を行うプログラム')
    parser.add_argument('--wavdir', '-w', default='./data', help='音が保存されているディレクトリ')
    parser.add_argument('--testdir', '-t', default='./testdata', help='テストに使いたい音が保存されているディレクトリ')
    parser.add_argument('--type', required=True, choices=["kutusita","slip"], help='靴下かスリッパかの違い')
    parser.add_argument('--aug', '-a', required=True, choices=["yes","no"], help='augmentationありかなしか')
    parser.add_argument('--model', '-m', required=True, help='modelの場所')
    args = parser.parse_args()

    a=main(args.aug, "", args.wavdir, args.testdir, args.type, args.model)

