from scipy.signal import argrelextrema
import numpy as np
import itertools
import librosa
import matplotlib.pyplot as plt


class LoudnessDecomposition(object):
    def __init__(self, audio, fs, threshold=75):
        """audio: np array, (n,) for mono; (n, 2) for stereo"""
        self.audio = audio
        self.fs = fs
        self.threshold = threshold
        # segments are stored in (n, 1, num_segments)
        self.num_segments, self.segments, self.indices_segments, self.loudness = self.initialize_segments()
        self.fudged_segments = self.initialize_fudged_segments()
        """to init
        self.num_segments
        self.segments
        self.fudged_segments
        """

    def get_number_segments(self):
        return self.num_segments

    def initialize_segments(self):
        audio = self.audio
        indices_min, loudness = self.get_loudness_indices()  # np.array: indices local minima (not the first and last el)
        temp_segments = []
        previous = 0
        current_index = 0
        audio_length = np.size(audio)
        number_indices = np.size(indices_min)
        if number_indices == 0:
            print("No minimum detected, initializing one component")
            temp_segments.append(audio)
        else:
            while previous < audio_length and current_index < number_indices:
                temp_segments.append(audio[previous:indices_min[current_index]])
                previous = indices_min[current_index]
                current_index += 1
            temp_segments.append(audio[previous:])
        return len(temp_segments), temp_segments, indices_min, loudness

    def initialize_fudged_segments(self):
        temp = []
        for comp in self.segments:
            temp.append(np.zeros(np.shape(comp)))
        return temp

    def get_segments_mask(self, mask):
        # mask: array of false and true, length of num_segments
        # get segments for true and fudged for false
        temp = np.array([])
        for index, value in enumerate(mask):
            if value:
                print(self.segments[index])
                temp = np.append(temp, np.array(self.segments[index]))
            else:
                print(self.fudged_segments[index])
                temp = np.append(temp, np.array(self.fudged_segments[index]))
        return temp

    def return_segments(self, indices):
        # make mask setting true for indices
        mask = np.zeros((self.num_segments,)).astype(bool)
        mask[indices] = True
        audio = self.get_segments_mask(mask)
        return audio

    def get_loudness_indices(self):
        loudness = self.compute_power_db()
        loudness_rounded = np.around(loudness, decimals=-1)
        li = [[k, next(g)[0]] for k, g in itertools.groupby(enumerate(loudness_rounded), key=lambda k: k[1])]
        loudness_no_dups = [item[0] for item in li]
        indices = [item[1] for item in li]
        # get threshold for loudness to start no components, check minima if they are above the threshold, if so: delete
        minima = np.array((argrelextrema(np.array(loudness_no_dups), np.less))).flatten()
        to_delete = []
        for i, m in enumerate(minima):
            if loudness[int(indices[int(m)])] > self.threshold:
                to_delete.append(i)
        minima = np.delete(minima, to_delete)
        indices_min = []
        for k in minima:
            indices_min.append(int(indices[int(k)]))
        return indices_min, loudness

    def compute_power_db(self, win_len_sec=0.1, power_ref=10**(-12)):
        """Computation of the signal power in dB

        Notebook: C1/C1S3_Dynamics.ipynb

        Args:
            x (np.ndarray): Signal (waveform) to be analyzed
            fs (scalar): Sampling rate
            win_len_sec (float): Length (seconds) of the window (Default value = 0.1)
            power_ref (float): Reference power level (0 dB) (Default value = 10**(-12))

        Returns:
            power_db (np.ndarray): Signal power in dB"""
        win_len = round(win_len_sec * self.fs)
        win = np.ones(win_len) / win_len
        power_db = 10 * np.log10(np.convolve(self.audio**2, win, mode='same') / power_ref)
        """s = np.abs(librosa.stft(self.audio))
        power_db = librosa.power_to_db(s**2, ref=power_ref)"""
        power_db[np.where(power_db == -np.inf)] = 0
        power_db = np.abs(power_db)
        return power_db

    def visualize_decomp(self, save_path=None):
        audio = self.audio
        indices = [0] + self.indices_segments + [np.size(audio)]
        loudness = self.loudness
        fig, (ax1, ax2) = plt.subplots(2)
        fig.suptitle('Loudness decomposition')
        ax1.plot(audio, color='c')
        for line in indices:
            ax1.axvline(x=line, color='m')
        ax1.set(ylabel='Amplitude', xlim=[0, np.size(audio)])
        for i in range(0, len(indices) - 1, 2):
            ax1.axvspan(indices[i], indices[i+1], facecolor='m', alpha=0.1)
        ax2.plot(loudness, color='c')
        for line in indices:
            ax2.axvline(x=line, color='m')
        ax2.set(xlabel='Time', ylabel='Power (db)', xlim=[0, np.size(audio)])
        for i in range(0, len(indices) - 1, 2):
            ax2.axvspan(indices[i], indices[i+1], facecolor='m', alpha=0.1)
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()
        print("visualized :)")
