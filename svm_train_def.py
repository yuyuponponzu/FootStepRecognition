import numpy as np
import os
import random
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sn
import pickle
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import preprocessing
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
import argparse

parser = argparse.ArgumentParser(description='CNNの学習を行うプログラム')
parser.add_argument('--aug', '-a', required=True,choices=['yes', 'no'], help='Augmentationありのデータを使うか否か')
parser.add_argument('--type', '-t', required=True,choices=['kutusita_npzdata/', 'slip_npzdata/'], help='靴下とスリッパどっちを使っているか')
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

traindata = np.array([x_train[i].flatten() for i in range(train_num)])
testdata = np.array([x_test[i].flatten() for i in range(test_num)])
trainlabels = y_train
testlabels = y_test

"""
search_params = [
    {
        "kernel"          : ["linear"],
        #'C':np.logspace(-5, 15, 21, base=2),
        #'gamma':np.logspace(-15, 3, 19, base=2),
        "C"               : [10**i for i in range(0,3)],
        "gamma"           : [10**j for j in range(-3,0)],
        "random_state"    : [2525],
    }
]
gs = GridSearchCV(SVC(), 
              search_params,
              verbose=True, 
              n_jobs=4)
gs.fit(traindata, trainlabels)
print(gs.best_estimator_)
print(gs.best_estimator_.score(testdata,testlabels))
#predicted_label = gs.best_estimator_.predict(testdata)
"""
#for params, mean_score, all_scores in gs.grid_scores_:
#    print "{:.3f} (+/- {:.3f}) for {}".format(mean_score, all_scores.std() / 2, params)

model = SVC(C=0.001, cache_size=200, class_weight=None, coef0=0.0,
          decision_function_shape='ovr', degree=3, gamma=1e-20, kernel='linear',
            max_iter=-1, probability=False, random_state=2525, shrinking=True,
              tol=0.001, verbose=False)

model.fit(traindata,trainlabels)

if augkey == "yes" :
    filename = './svm_model_default/finalized_model_aug.sav'
else :
    filename = './svm_model_default/finalized_model.sav'


#pickle.dump(gs, open(filename, 'wb'))
pickle.dump(model, open(filename, 'wb'))
