import numpy as np
import librosa
from sklearn.decomposition import PCA


def process(file_path, path):
    y, fs = librosa.load(file_path + path + '_m.wav')
    # pre-emphasis
    pre_emphasis = 0.97
    y = np.append(y[0], (y[1:] - pre_emphasis * y[:-1]))
    # zero padding
    zero_temp = np.zeros(90000)
    if y.size < 90000:
        zero_temp[0:y.size] = y[0:]
    else:
        zero_temp[0:] = y[0:90000]
    y = zero_temp
    frame_size = 0.025
    frame_stride = 0.01
    nmel = 72
    frame_length = frame_size * fs
    frame_length = int(round(frame_length))
    frame_step = frame_stride * fs
    frame_step = int(round(frame_step))
    # Short Time Fourier Transform
    y = librosa.stft(y, n_fft=frame_length, hop_length=frame_step, win_length=None, window='hann', center=True,
                     pad_mode='reflect')
    # power spectral density
    y = np.abs(y) ** 2
    # Mel-Spectrum
    y = librosa.feature.melspectrogram(S=y, n_mels=72)
    y = librosa.power_to_db(y)
    # Mel-Frequency Cepstrum
    y = librosa.feature.mfcc(S=y, n_mfcc=32, dct_type=2, norm='ortho')
    # Principal components analysis
    pca = PCA(n_components=1)
    # Mean
    # y = pca.fit(y)
    # PCA
    y = pca.fit_transform(y.T)
    y = y.T
    return y