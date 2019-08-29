import os
import sys
import numpy as np
import argparse
import soundfile as sf

parser = argparse.ArgumentParser(description='足音を一歩分で切り取るプログラム')
parser.add_argument('--wavdir', '-w', default='.', help='保存されているディレクトリ')        
args = parser.parse_args()

for c in os.listdir(args.wavdir):
    print('class: {}'.format(c))
    d = os.path.join(args.wavdir, c)
    wavs = os.listdir(d)

    for i in [f for f in wavs if ('wav' in f)]:
         j = os.path.join(d, i)
        wav, fs = sf.read(j)
        print(wav.shape)
        wav_l = wav[:, 0]
        wav_r = wav[:, 1]
        xs = (0.5 * wav_l) + (0.5 * wav_r)
        k = "mono" + i
        kk = os.path.join(d,k)
        sf.write(kk, xs, samplerate=fs)
 
        """
        if i == "huziisan01.wav":
            j = os.path.join(d, i)
            wav, fs = sf.read(j)
            print(wav.shape)
            wav_l = wav[:, 0]
            wav_r = wav[:, 1]
            xs = (0.5 * wav_l) + (0.5 * wav_r)
            k = "mono" + i
            kk = os.path.join(d,k)
            sf.write(kk, xs, samplerate=fs)
        """
