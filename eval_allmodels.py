import svm_eval as sv
import cnn_eval as cn
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description='音を認識するモデルの学習')
parser.add_argument('--wavdir', '-w', default='./data', help='音が保存されているディレクトリ')
parser.add_argument('--testdir', '-t', default='./testdata', help='音が保存されているディレクトリ')
parser.add_argument('--type', required=True, choices=["kutusita","slip"] help='靴下かスリッパかの違い')
args = parser.parse_args()

def_ = [sv.main, sv.main, cn.main]
df_aug = [''] * 3
df_ = [] * 3

#Augmentationなし，key_2はnmfあり，nmfなし，CNNの順番で読み取る
key_1 = ["no", "no", "no"]
key_2 = ["yes", "no", ""]

for i, de in enumerate(def_):
    x = de(key_1[i], key_2[i], args.wavdir, args.testdir, args.type)
    df = pd.DataFrame(x).T
    pd.options.display.float_format = '{:.2f}'.format
    df = df.rename(index={'0':'A','1':'B','2':'C','3':'D'})
    print(df)
    df_aug[i] = df.drop('support',axis=1)

log = open('./log_.txt','w')
dff = pd.concat([df_aug[0], df_aug[1], df_aug[2]], axis=1)
log.write(dff.to_latex())


#Augmentationありの処理．SVM単体だと，学習が終わらなかったから二つのみ．

def_ = [sv.main, cn.main]
df_aug = [''] * 2
df_ = [] * 2
key_1 = ["yes", "yes"] #aug 
key_2 = ["yes", ""] #nmf

for i, de in enumerate(def_):
    x = de(key_1[i], key_2[i], args.wavdir, args.testdir, args.type)
    df = pd.DataFrame(x).T
    pd.options.display.float_format = '{:.2f}'.format
    df = df.rename(index={'0':'A','1':'B','2':'C','3':'D'})
    df_aug[i] = df.drop('support',axis=1)

log = open('./log_aug.txt','w')
dff = pd.concat([df_aug[0], df_aug[1]], axis=1)
log.write(dff.to_latex())
