"""
plot the feature from feature bank
------------------------------------------------------------------
Copyright: 2019 Wang,Zehui (wzehui@hotmail.com)
@author: Wang,Zehui
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from sklearn.decomposition import PCA


path = '/Users/wzehui/Documents/MA/Daten/quellcode/sounddb/T2-Tamino/T2-m034-Tami-I-a1-1_m.wav'

def mel_plot(path):

    y, fs = librosa.load(path)

    frame_size = 0.025
    frame_stride = 0.01

    frame_length = frame_size * fs
    frame_length = int(round(frame_length))

    frame_step = frame_stride * fs
    frame_step = int(round(frame_step))

    y = librosa.stft(y, n_fft=frame_length, hop_length=frame_step, win_length=None, window='hann', center=True, pad_mode='reflect')
    y = np.abs(y) ** 2
    y = librosa.feature.melspectrogram(S=y, n_mels=72)
    y = librosa.power_to_db(y)
    y = librosa.feature.mfcc(S=y, n_mfcc=32, dct_type=2, norm='ortho')

    pca = PCA(n_components=1, svd_solver='full')
    y = pca.fit(y)
    y = y.mean_
    # y = pca.fit_transform(y.T)
    # y = y.T

    # plt.figure(figsize=(10, 4))
    # librosa.display.specshow(y, y_axis='mel', x_axis='time')
    #
    # plt.colorbar(format='%+2.0f dB')
    # plt.tight_layout()
    #
    # plt.savefig('/Users/wzehui/Documents/MA/Plot/STFT.pdf')
    # plt.show()

    return y


y = mel_plot(path)