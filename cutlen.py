import argparse
import os
import sys
import pydub
import numpy as np
from pydub.silence import split_on_silence
from pydub import AudioSegment
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='閾値による一歩分切り出しプログラム')
parser.add_argument('--wavdir', '-w', required=True, help='切り出したいデータがある親ディレクトリ')
parser.add_argument('--const', '-c', default=1.25, help='切り出しの閾値の調整係数')
args = parser.parse_args()
save_dir = "{}/split/".format(args.wavdir)
CONST = args.const

for c in os.listdir(args.wavdir):
    print('class: {}'.format(c))
    if  c.find('.wav') == -1:
        d = os.path.join(args.wavdir, c)
        if ".DS_Store" in d:
            print("process is passing")
            continue
        wavs = os.listdir(d)
        for i in [f for f in wavs if ('wav' in f)]:
            k = os.path.join(d, i)
            base_sound = AudioSegment.from_file(k, format="wav")  # 音声を読み込み
            base_sound.set_channels(1)
            length_seconds = base_sound.duration_seconds  # 長さを確認
            rms = base_sound.dBFS * CONST #CONSTは調整用の定数
            chunks = split_on_silence(
                base_sound,
                # 100ms以上の無音がある箇所で分割
                min_silence_len=100,
                # 実効値の平均以下で無音とみなす
                silence_thresh=rms,
                # 分割後200msだけ、無音を残す
                keep_silence=200,
                seek_step=1
                )

                # 分割数の表示
            #print(len(chunks), length_seconds)
            g = os.path.join(save_dir, c)
            if not os.path.exists(g):
                os.makedirs(g)
            else:
                #前保存してたファイルと競合させない
                while(os.path.exists(g)):
                    g = g+'_'
                os.makedirs(g)
            for i, chunk in enumerate(chunks):
                f = os.path.join(g, "split_%03d.wav" %(i))
                chunk.export(f, format="wav")
                """2歩分なら以下の書き方
                for j, next_chunk in enumerate(chunks):
                    if i + 1 == j:
                        output = chunk + next_chunk
                        output.export(g+"00"+str(i)+".wav", format="wav")
                """
