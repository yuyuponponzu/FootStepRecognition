import argparse
import os
import sys
import functions as fc

parser = argparse.ArgumentParser(description='足音音声のplot用プログラム')
parser.add_argument('-w','--wavdir', choices=['data/andosan/', 'data/hori/', 'data/akagawakun/', 'data/fujiisan'], 
                    default='data/andosan/')
args = parser.parse_args()

#好きな処理を記述

#fc.plot_wavdir(args.wavdir) #指定したディレクトリの足音を全て順にplot
#fc.plot_onestep("./data/andosan/split004.wav") #指定した音源の一歩分足音をplot
#fc.plot_augment("./data/andosan/split004.wav") #指定した音源の一歩分足音を水増ししたデータをplot
fc.plot_NMF_reconstruct_error("./data/andosan/split004.wav") #指定した音源の一歩分足音をNMFで分解した再建誤差をplot

