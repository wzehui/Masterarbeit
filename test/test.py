import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.gridspec as gridspec

# Fixing random state for reproducibility
np.random.seed(19680801)

dt = 5
t = np.arange(0, 1000, dt)
nse = 0.2*np.random.randn(len(t))

s = np.sin(2 * np.pi * t)

a = s+nse

plt.subplot(312)
plt.title
plt.plot(t, s)
plt.subplot(313)
plt.plot(t, nse)
plt.subplot(311)
plt.plot(t, a)

ax1 = plt.subplot(311)
ax2 = plt.subplot(312)
ax3 = plt.subplot(313)

ax3.set_xlabel('Frequenz [Hz]')
ax1.set_ylabel('log X[k]')
ax2.set_ylabel('log H[k]')
ax3.set_ylabel('log E[k]')

plt.savefig("cepstral.pdf")

plt.show()

# from matplotlib.pyplot import specgram
# import librosa
#
# from scipy.io import wavfile
# # sample_rate, X = wavfile.read('hello.mp3')
# y, sr = librosa.load(librosa.util.example_audio_file(), duration=0.5)
# librosa.output.write_wav('file.wav', y, sr)
# sample_rate, X = wavfile.read('file.wav')
#
# print (sample_rate, X.shape)
# S = specgram(X, Fs=sample_rate, xextent=(0,192000))
