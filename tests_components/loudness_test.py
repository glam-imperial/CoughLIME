import numpy as np
import os
import librosa
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
import libfmp.b


def compute_power_db(x, Fs, win_len_sec=0.1, power_ref=10**(-12)):
    """Computation of the signal power in dB

    Notebook: C1/C1S3_Dynamics.ipynb

    Args:
        x (np.ndarray): Signal (waveform) to be analyzed
        Fs (scalar): Sampling rate
        win_len_sec (float): Length (seconds) of the window (Default value = 0.1)
        power_ref (float): Reference power level (0 dB) (Default value = 10**(-12))

    Returns:
        power_db (np.ndarray): Signal power in dB
    """
    win_len = round(win_len_sec * Fs)
    win = np.ones(win_len) / win_len
    power_db = 10 * np.log10(np.convolve(x**2, win, mode='same') / power_ref)
    return power_db


if __name__ == '__main__':
    test_path = '/Users/anne/Documents/Uni/Robotics/Masterarbeit/MA_Code/DICOVA/DiCOVA_Train_Val_Data_Release/AUDIO/iyWdhFuN_cough.flac'
    x, Fs = librosa.load(test_path)

    win_len_sec = 0.2
    power_db = compute_power_db(x, win_len_sec=win_len_sec, Fs=Fs)

    libfmp.b.plot_signal(x, Fs=Fs, ylabel='Amplitude')
    plt.show()

    libfmp.b.plot_signal(power_db, Fs=Fs, ylabel='Power (dB)', color='red')
    plt.ylim([70, 110])
    plt.show()
    print(power_db)