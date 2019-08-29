# -*- coding: utf-8 -*-
#from __future__ import print_function
import argparse
import pickle
import os
import sys
import six
import numpy as np

import scipy.signal as signal
import librosa
import numpy as np
import matplotlib.pyplot as plt

from scipy.fftpack import fft
from scipy import ifft
from scipy import ceil, complex64, float64, hamming, zeros
from scipy.io.wavfile import read
from scipy.io.wavfile import write
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.svm import SVR
from sklearn import model_selection
from sklearn.decomposition import NMF
from sklearn.metrics import classification_report

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

def alignx(x, length):
    x = np.pad(x,(0,length-len(x)),"constant")
    return x

def calculate_melsp(x, n_fft=1024, hop_length=128):
    stft = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length))**2
    log_stft = librosa.power_to_db(stft)
    melsp = librosa.feature.melspectrogram(S=log_stft,n_mels=128)
    return melsp


def main(augkey, nmfkey, wavdir, testdir, type_):
    wavdir = os.path.join(wavdir, type_)
    testdir = os.path.join(testdir, type_)
    K = 25
    model = NMF(n_components=K, init='random', random_state=0)
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
            x = np.asarray(alignx(x,length)).astype(np.float32)
            _x = calculate_melsp(x)
            print('   {} loaded'.format(i))
            if nmfkey == "yes" :
                print("NMF!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                _x = model.fit_transform(_x)
            P = _x.flatten()
            train = [np.asarray(P).astype(np.float32)] if train == [] else np.vstack((train, [np.asarray(P).astype(np.float32)]))
            labelarr = label if labelarr == [] else np.vstack([labelarr, label])
    labelarr = labelarr.flatten()

    traindata, testdata, trainlabels, testlabels = model_selection.train_test_split(train, labelarr, test_size=0)
    print(labelarr)
    print(trainlabels)
    if nmfkey == "yes" :
        if augkey == "yes" :
            filename = './svm_model_nmf_W/finalized_model_aug.sav'
            #log = open('./log_svm_nmf_aug.txt','w')
        else :
            filename = './svm_model_nmf_W/finalized_model.sav'
            #log = open('./log_svm_nmf.txt','w')
    else :
        if augkey == "yes" :
            filename = './svm_model_default/finalized_model_aug.sav'
            #log = open('./log_svm_aug.txt','w')
        else :
            filename = './svm_model_default/finalized_model.sav'
            #log = open('./log_svm.txt','w')

    loaded_model = pickle.load(open(filename, 'rb'))
    #print(loaded_model.best_estimator_.score(traindata,trainlabels))
    #predicted_label = loaded_model.best_estimator_.predict(testdata)
    #print(predicted_label)
    score = loaded_model.score(traindata,trainlabels)
    print(loaded_model.score(traindata,trainlabels))
    pred = loaded_model.predict(traindata)
    print(pred)
    report = classification_report(trainlabels, pred, output_dict=True)
    print(report)
    #log.write("\n")
    #log.write("scores:"+str(score*100)+"\n")
    #log.write("report:"+report+"\n")
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='音を認識するモデルの学習')
    parser.add_argument('--wavdir', '-w', default='./data', help='音が保存されているディレクトリ')
    parser.add_argument('--testdir', '-t', default='./testdata', help='音が保存されているディレクトリ')
    parser.add_argument('--type', required=True, choices=["kutusita","slip"], help='靴下かスリッパかの違い')
    parser.add_argument('--aug', required=True, choices=["yes","no"], help='augmentationありか否か')
    parser.add_argument('--nmf', required=True, choices=["yes","no"], help='NMFありか否か')
    parser.add_argument('--out', '-o', default='result', help='ログ，モデルの出力先')
    args = parser.parse_args()

    x = main(args.aug, args.nmf, args.wavdir, args.testdir, args.type)
