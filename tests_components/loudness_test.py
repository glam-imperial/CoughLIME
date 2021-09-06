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
    print(np.min(power_db), np.max(power_db))
    print(len(x), len(power_db))
    lines = [0, 10000, 25000, 30000, 45000, 70000, np.size(x)]
    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle('Loudness decomposition')
    ax1.plot(x, color='c')
    for line in lines:
        ax1.axvline(x=line, color='m')
    ax1.set(ylabel='Amplitude', xlim=[0, np.size(x)])
    for i in range(0, len(lines) - 1, 2):
        ax1.axvspan(lines[i], lines[i+1], facecolor='m', alpha=0.1)
    ax2.plot(power_db, color='c')
    for line in lines:
        ax2.axvline(x=line, color='m')
    ax2.set(xlabel='Time', ylabel='Power (db)', xlim=[0, np.size(x)])
    for i in range(0, len(lines) - 1, 2):
        ax2.axvspan(lines[i], lines[i+1], facecolor='m', alpha=0.1)
    plt.show()
    print(power_db)