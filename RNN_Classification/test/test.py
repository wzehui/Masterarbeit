import wavio
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import librosa


path = '/Users/wzehui/Documents/MA/Daten/quellcode/sounddb/S1-Uebung/S1-w035-Uebu-A-a1-1_m.wav'
epsilon = np.finfo(np.float).eps

wav_struct = wavio.read(path)

frame_size =0.025; frame_stride=0.01
frame_length = frame_size * wav_struct.rate
frame_length = int(round(frame_length))

frame_step = frame_stride * wav_struct.rate
frame_step = int(round(frame_step))

wav = wav_struct.data.astype(float)/np.power(2, wav_struct.sampwidth*8-1)
wav = (wav[:, 0] + wav[:, 1]) / 2

# z = np.zeros(30000-wav_struct.data.shape[0])
# wav = np.append(wav, z)

[f, t, X] = signal.spectral.spectrogram(wav, wav_struct.rate, np.hamming(frame_length), nperseg=frame_length, noverlap=frame_step, detrend=False, return_onesided=True, mode='magnitude')

spectrum, fs, ts, fig = plt.specgram(wav+epsilon, NFFT = frame_length,Fs =wav_struct.rate,window=np.hanning(M=frame_length),noverlap=frame_step,mode='psd',scale_by_freq=True,sides='default',scale='dB',xextent=None)
# plt.plot(f,t,X)
# plt.imshow(X)
plt.show()

melX = librosa.feature.melspectrogram(t, sr=wav_struct.rate, n_mels=64, S=X, n_fft=frame_length, hop_length=frame_step, power=2.0)
# melW = librosa.filters.mel(22050, n_fft=551,n_mels=64)
# melW /= np.max(melW,axis=-1)[:,None]
# melX = np.dot(X, melW.T)
