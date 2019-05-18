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


def mel_plot(path, name, label):

    y, fs = librosa.load(path)

    frame_size = 0.025
    frame_stride = 0.01

    frame_length = frame_size * fs
    frame_length = int(round(frame_length))

    frame_step = frame_stride * fs
    frame_step = int(round(frame_step))

    y = librosa.stft(y, n_fft=frame_length, hop_length=frame_step, win_length=None, window='hann', center=True, pad_mode='reflect')
    y = np.abs(y)**2
    y = librosa.feature.melspectrogram(sr=fs, S=y, n_fft=frame_length, hop_length=frame_step, power=2.0, n_mels=64)
    # y = librosa.power_to_db(y, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(y, y_axis='mel', fmax=8000, x_axis='time')

    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.tight_layout()

    if label:
        plt.savefig('/Users/wzehui/Documents/MA/Plot/1/' + str(name))
    else:
        plt.savefig('/Users/wzehui/Documents/MA/Plot/0/' + str(name))

    # plt.show()

