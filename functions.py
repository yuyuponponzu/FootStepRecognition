import os
import sys
import librosa
import librosa.display
import augment
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from scipy.io.wavfile import write
from sklearn.decomposition import NMF
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

"""
dir と dirname と wavname の違いに注意
dir はディレクトリ最上層の名前だけ(基本は'./data')
dirname はディレクトリの名前だけ('./data/andosan')
wavname はwav名まで含めて一意になるように('./data/andosan/01.wav')
"""

# メルスペクトログラム を算出
def calculate_melsp(x, fs, n_fft=1024, hop_length=128):
    S = librosa.feature.melspectrogram(x, sr=fs, n_mels=128)
    log_S = librosa.amplitude_to_db(S)
    return log_S

# 任意のディレクトリの足音波形を表示
def plot_wavdir(dirname):
    wavs = os.listdir(dirname)
    try:
        while True:
            for i in [f for f in wavs if ('wav' in f)]:
                k = os.path.join(dirname, i)
                fs, x = read(k)
                plt.figure()
                plt.plot(x)
                plt.show()
    except KeyboardInterrupt:
        sys.exit

# 任意の足音一歩分をplot
def plot_onestep(wavname):
    x, fs = librosa.load(wavname) # wavファイルの読み込み
    plt.figure() # 上側にそのままの波形，下側にメルスペクトログラム 
    ax1 = plt.subplot(2,1,1)
    librosa.display.waveplot(x)
    plt.title('(a)')
    ax_pos = ax1.get_position()
    plt.text(ax_pos.x1 - 0.1, ax_pos.y1 - 0.05, "(a)")
    ax2 = plt.subplot(2,1,2,sharex=ax1)
    librosa.display.specshow(calculate_melsp(x, fs), sr=fs, x_axis='time', y_axis='mel')
    plt.title('(b)')
    ax_pos = ax2.get_position()
    plt.text(ax_pos.x1 - 0.1, ax_pos.y1 - 0.05, "(a)")
    plt.colorbar(orientation='horizontal')
    plt.tight_layout()
    plt.show()

def plot_augment(wavname):
    x, fs = librosa.load(wavname)
    plt.figure(figsize=(37,5)) # 上側にそのままの波形，下側にメルスペクトログラム
    aug = [augment.add_white_noise, augment.shift_sound, augment.stretch_sound]
    i = 0
    ax = [''] * 7
    str_ = ['(a)', '(b)', '(c)']
    for au in aug:
        i += 1
        xx = au(x)
        ax[i] = plt.subplot(2,3,i) if i == 1 else plt.subplot(2,3,i,sharey=ax[1])
        librosa.display.waveplot(xx)
        ax[i].xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        plt.title(str_[i-1])
        ax[i+3] = plt.subplot(2,3,i+3)#,sharex=ax[i])
        librosa.display.specshow(calculate_melsp(xx, fs), sr=fs, x_axis='time', y_axis='mel')
        ax[i+3].axes.get_xaxis().set_ticks([])
        plt.colorbar(orientation='horizontal')
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
    #plt.tight_layout()
    plt.show()

def plot_NMF_reconstruct_error(wavname):
    x, fs = librosa.load(wavname) 
    mel_x = calculate_melsp(x, fs)
    xx = np.arange(1,31)
    error_ = np.array([])
    print("Start!")
    for k in range(1,31,1):
        print("{}_Start!".format(k))
        model = NMF(n_components=k, init='random', random_state=0)
        W = model.fit_transform(abs(mel_x))
        error_ = np.append(error_, model.reconstruction_err_)
    plt.figure()
    plt.bar(xx, error_)
    plt.xlabel("Number of basis vectors")
    plt.ylabel("Approximation error")
    plt.tight_layout()
    plt.show()

# 一歩分に切り取り
def cut_length(dir):
    for c in os.listdir(dir):
        if "wav" in c :
            pass
        else :
            print('class: {}'.format(c))
            d = os.path.join(dir, c)
            wavs = os.listdir(d)
            for i in [f for f in wavs if ('wav' in f)]:
                k = os.path.join(d, i)
                base_sound = AudioSegment.from_file(k, format="wav")  # 音声を読み込み
                base_sound.set_channels(1)
                length_seconds = base_sound.duration_seconds  # 長さを確認
                rms = base_sound.dBFS * 1.3 #1.25 藤井さん用
                chunks = split_on_silence(
                    base_sound,
                    # 1500ms以上の無音がある箇所で分割
                    min_silence_len=100,
                    # 実効値の平均以下で無音とみなす
                    silence_thresh=rms,
                    # 分割後500msだけ、無音を残す
                    keep_silence=200,
                    seek_step=1
                    )
                g = os.path.join(d,"split")
                for i, chunk in enumerate(chunks):
                    chunk.export(g+"00"+str(i)+".wav", format="wav")

# mono音源に変換
def convert_mono(dir):
    for c in os.listdir(dir):
        print("Convert to Mono now!")    
        print('class: {}'.format(c))
        d = os.path.join(dir, c)
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
