"""
Load raw audio files in frequency domain with help of Short Time Fourier
Transform and Mel frequency.
------------------------------------------------------------------------
Copyright: 2019 Wang,Zehui (wzehui@hotmail.com)
@author: Wang,Zehui
"""

import numpy as np
import torch


def calc_spectrum(signal, sample_rate=22050, frame_size=0.025, frame_stride=0.01):

    # Calculate the number of frames from the signal
    frame_length = frame_size * sample_rate
    frame_length = int(round(frame_length))

    frame_step = frame_stride * sample_rate
    frame_step = int(round(frame_step))

    win = torch.hann_window(frame_length)
    n_fft = frame_length + frame_step

    # Returns the real and the imaginary parts together as one tensor of size
    # (∗×N×T×2) where ∗ is the optional batch size of input,
    # N is the number of frequencies where STFT is applied,
    # T is the total number of frames used, and each pair in the last dimension represents a complex number as the
    # real part and the imaginary part.

    a = torch.stft(signal, n_fft, hop_length=frame_step, win_length=frame_length, window=win, center=True, pad_mode='reflect', \
                   normalized=False, onesided=True)
    # signal_length = len(signal)
    # num_frames = 1 + int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))
    # zero padding
    # pad_signal_length = num_frames * frame_step + frame_length
    # z = torch.zeros((pad_signal_length - signal_length), 1)
    # Pad signal to make sure that all frames have equal number of samples
    # without truncating any samples from the original signal
    # pad_signal = torch.cat((signal, z), dim=0, out=None)

    # Slice the signal into frames from indices
    # indices = torch.arange(0, frame_length).repeat((num_frames, 1)) + \
    #           torch.arange(0, num_frames * frame_step, frame_step).repeat(frame_length, 1).permute(1, 0)

    # frames = pad_signal[indices.astype(np.int32, copy=False)]
    # Get windowed frames
    # frames *= winfunc(frame_length)
    # Compute the one-dimensional n-point discrete Fourier Transform(DFT) of
    # a real-valued array by means of an efficient algorithm called Fast Fourier Transform (FFT)
    # mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    # Compute power spectrum
    # pow_frames = (1.0 / NFFT) * ((mag_frames) ** 2)

    return a