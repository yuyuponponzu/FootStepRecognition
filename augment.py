# -*- coding: utf-8 -*-
#from __future__ import print_function
import argparse
import os
import sys
import six
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
import pandas as pd
import librosa
import librosa.display
import seaborn as sn
import math
import json
import scipy.signal as signal
from scipy import ceil, complex64, float64, hamming, zeros
from scipy.fftpack import fft
from scipy import ifft
from scipy.io.wavfile import read
from scipy.io.wavfile import write
from sklearn.model_selection import train_test_split

# 外れ値算出のため4分位を算出
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

"""単純に最長のlengthに合わせる処理
def lengthcheck(wavdir):
    length = 0
    for c in os.listdir(wavdir):
        d = os.path.join(wavdir, c)
        if ".DS_Store" in d:
            print("process is passing")
            continue
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

# change wave data to mel-stft
def calculate_melsp(x, n_fft=1024, hop_length=128):
    stft = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length))**2
    log_stft = librosa.power_to_db(stft)
    melsp = librosa.feature.melspectrogram(S=log_stft,n_mels=128)
    return melsp

# display wave in plots
def show_wave(x):
    plt.plot(x)
    plt.show()

# display wave in heatmap
def show_melsp(melsp, fs):
    librosa.display.specshow(melsp, sr=fs)
    plt.colorbar()
    plt.show()

# data augmentation: add white noise
def add_white_noise(x, rate=0.02):
    return x + rate*np.random.randn(len(x))

# data augmentation: shift sound in timeframe
def shift_sound(x, rate=2):
    return np.roll(x, int(len(x)//rate))

# data augmentation: stretch sound
def stretch_sound(x, rate=1.1):
    input_length = len(x)
    x = librosa.effects.time_stretch(x, rate)
    if len(x)>input_length:
        return x[:input_length]
    else:
        return np.pad(x, (0, max(0, input_length - len(x))), "constant")

# save wave data in npz, with augmentation
def save_np_data(filename, x, y, freq, time, aug=None, rates=None):
    np_data = np.zeros(freq*time*len(x)).reshape(len(x), freq, time)
    print(np_data.shape)
    np_targets = np.zeros(len(y))
    for i in range(len(y)):
        _x = x[i]
        if aug is not None:
            _x = aug(x=_x, rate=rates[i])
        _x = calculate_melsp(_x)
        np_data[i] = _x
        np_targets[i] = y[i]
    np.savez(filename, x=np_data, y=np_targets)

#save wave data in npz, this is generating multiple for one data.
def save_np_multi_data(filename, x, y, freq, time, N, aug=None, rates=None):
    np_data = np.zeros(freq*time*len(x)*N).reshape(len(x)*N, freq, time)
    stack_x = x
    stack_y = y
    for i in range(N-1):
        stack_x = np.vstack((stack_x,x))
        stack_y = np.hstack((stack_y,y))
    print("y.shape:",y.shape,)
    print("stack_y.shape:",stack_y.shape)
    print("output_xdata.shape:",np_data.shape)
    print("stack_x.shape:",stack_x.shape)
    print("rates.shape:",rates.shape)
    print("N:",N)
    np_targets = np.zeros(len(y)*N)
    for i in range(len(y)*N):
        _x = stack_x[i]
        if aug is not None:
            _x = aug(x=_x, rate=rates[i])
        _x = calculate_melsp(_x)
        np_data[i] = _x
        np_targets[i] = stack_y[i]
    np.savez(filename, x=np_data, y=np_targets)

def main():
    parser = argparse.ArgumentParser(description='足音音声の学習データの準備用プログラム')
    parser.add_argument('--wavdir', '-w', default='.', help='音が保存されているディレクトリ')            
    args = parser.parse_args()

    #足音音声データを取得
    print('Loading dataset ...')
    train = []
    train2 = []
    labelarr = []
    labelarr2 = []
    label = 0
    length = lengthcheck(args.wavdir)
    class_ = dict([])
    ind = 0
    for c in os.listdir(args.wavdir):
        print('class: {}, class id: {}'.format(c, label))
        class_[label] = c
        d = os.path.join(args.wavdir, c)
        if ".DS_Store" in d:
            print("process is passing")
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
            if len(x[1]) != 1:
                x = to_mono(x) #ステレオからモノラルへ
            if length < len(x): 
                print("This is outlier. Excluding...")
                continue #外れ値を算出（大抵は１歩以上の足音が入ってる）
            x = alignx(x,length)
            print('   {} loaded'.format(i))
            #ind += 1
            #if ind ==2: return x, length
            train = [np.asarray(x).astype(np.float32)] if train == [] else np.vstack((train, [np.asarray(x).astype(np.float32)]))
            labelarr = label if labelarr == [] else np.vstack([labelarr, label])
        label += 1
    labelarr = labelarr.flatten()
    x_train, x_test, y_train, y_test = train_test_split(train, labelarr, test_size=0.2)
    f = os.path.join('./class.json')
    fw = open(f,'w') # classとlabelの対応づけを保存
    json.dump(class_,fw,indent=4)

    freq = 128 # 128 is frequency band. fs/N = about 43Hz
    time = math.ceil(length / 128 ) # 128 is hop length

    # save test dataset
    f = os.path.join(args.wavdir, "esc_melsp_test.npz")
    if not os.path.exists(f):
        save_np_data(f, x_test,  y_test, freq, time)

    f = os.path.join(args.wavdir, "esc_melsp_train_raw.npz")
    # save raw training dataset
    if not os.path.exists(f):
        save_np_data(f, x_train,  y_train, freq, time)

    f = os.path.join(args.wavdir, "esc_melsp_train_wn.npz")
    # save training dataset with white noise
    if not os.path.exists(f):
        #rates = np.random.randint(1,50,len(x_train))/10000
        N = int(50)
        rates = np.array([np.full(len(x_train), i/1000) for i in range(1,N+1)]).flatten()
        save_np_multi_data(f, x_train,  y_train, freq, time, N, aug=add_white_noise, rates=rates)

    f = os.path.join(args.wavdir, "esc_melsp_train_ss.npz")
    # save training dataset with sound shift
    if not os.path.exists(f):
        #rates = np.random.choice(np.arange(2,6),len(y_train)) #全てのデータに対する2~6までの乱数をセット
        N = int(30)
        rates = np.array([np.full(len(x_train),i) for i in range(2,2+N)]).flatten()
        save_np_multi_data(f, x_train,  y_train, freq, time, N, aug=shift_sound, rates=rates)

    f = os.path.join(args.wavdir, "esc_melsp_train_st.npz")
    # save training dataset with stretch
    if not os.path.exists(f):
        #rates = np.random.choice(np.arange(80,120),len(y_train))/100
        N = int(30)
        rates = np.array([np.full(len(x_train),(i+85)/100) for i in range(0,N)]).flatten()
        save_np_multi_data(f, x_train,  y_train, freq, time, N, aug=stretch_sound, rates=rates)

    f = os.path.join(args.wavdir, "esc_melsp_train_com.npz")
    if not os.path.exists(f):
        N_wn = int(100)
        np_data = np.zeros(freq*time*len(x_train)*N_wn).reshape(len(x_train)*N_wn, freq, time)
        np_targets = np.zeros(len(y_train)*N_wn)
        stack_x = x_train
        stack_y = y_train
        for i in range(N_wn-1):
            stack_x = np.vstack((stack_x,x_train))
            stack_y = np.hstack((stack_y,y_train))
        for i in range(len(y_train)*N_wn):
            x = stack_x[i]
            x = add_white_noise(x=x, rate=np.random.randint(1,N_wn*3)/1000)
            if np.random.choice((True,False)):
                x = shift_sound(x=x, rate=np.random.choice(np.arange(2,6)))
            else:
                x = stretch_sound(x=x, rate=np.random.choice(np.arange(80,120))/100)
            print("x",x.shape,"np_data",np_data.shape)
            x = calculate_melsp(x)
            np_data[i] = x
            np_targets[i] = stack_y[i]
        np.savez(f, x=np_data, y=np_targets)

if __name__ == "__main__":
    main()
