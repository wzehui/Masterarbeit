"""
extract feature bank
------------------------------------------------------------------
Copyright: 2019 Wang,Zehui (wzehui@hotmail.com)
@author: Wang,Zehui
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import torch


def mel_freq(y, fs, frame_size=0.025, frame_stride=0.01, nmel=72):

    frame_length = frame_size * fs
    frame_length = int(round(frame_length))

    frame_step = frame_stride * fs
    frame_step = int(round(frame_step))

    y = torch.squeeze(y)
    y = torch.Tensor.numpy(y)   # tensor to ndarray

    y = librosa.stft(y, n_fft=frame_length, hop_length=frame_step, win_length=None, window='hann', center=True,
                     pad_mode='reflect')
    y = np.abs(y) ** 2
    y = librosa.feature.melspectrogram(sr=fs, S=y, n_fft=frame_length, hop_length=frame_step, power=1.0, n_mels=nmel)

    # # plot
    # y = librosa.power_to_db(y, ref=np.max)
    # plt.figure(figsize=(10, 4))
    # librosa.display.specshow(y, y_axis='mel', fmax=8000, x_axis='time')
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Mel spectrogram')
    # plt.tight_layout()
    # plt.show()

    y = torch.from_numpy(y).float()    # ndarray to tensor
    return y
