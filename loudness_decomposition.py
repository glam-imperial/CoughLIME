from scipy.signal import argrelextrema
import numpy as np
import itertools
import matplotlib.pyplot as plt


class LoudnessDecomposition(object):
    """ decomposes the cough audio array according to minima in the power curve"""
    def __init__(self, audio, fs, min_length=0, threshold=75):
        """
        Init function
        :param audio: np.array((n,)), audio to be decomposed
        :param fs: int, sample rate
        :param min_length: int, minimum length for one component
        :param threshold: int, power threshold to generate the components
        """
        self.audio = audio
        self.fs = fs
        self.threshold = threshold
        self.min_length = min_length
        self.decomposition_type = 'loudness'
        # components are stored in list of numpy arrays that hold the individual components
        self.num_components, self.components, self.indices_components, self.loudness = self.initialize_components()
        self.fudged_components = self.initialize_fudged_components()

    def get_number_components(self):
        """
        :return: int, number of components generated during the decomposition
        """
        return self.num_components

    def initialize_components(self):
        """
        initializes the interpretable components
        :return: length_components, temp_components, indices_min, loudness
                length_components: number of components generated
                temp_components: generated components
                indices_min: indices of the minima of the power array -> splitting indices of decomposition
                loudness: 1d numpy array, power array, same length as audio array
        """
        audio = self.audio
        indices_min, loudness = self.get_loudness_indices()
        temp_components = []
        previous = 0
        current_index = 0
        audio_length = np.size(audio)
        number_indices = np.size(indices_min)
        if number_indices == 0:
            temp_components.append(audio)
        else:
            while previous < audio_length and current_index < number_indices:
                temp_components.append(audio[previous:indices_min[current_index]])
                previous = indices_min[current_index]
                current_index += 1
            temp_components.append(audio[previous:])
        return len(temp_components), temp_components, indices_min, loudness

    def initialize_fudged_components(self):
        """
        initializes the fudged components that are needed for training the linear model
        :return: array of same shape as original components, all set to 0
        """
        temp = []
        for comp in self.components:
            temp.append(np.zeros(np.shape(comp)))
        return temp

    def get_components_mask(self, mask):
        """
        return components for a mask, set to original audio component for true and fudged for false
        :param mask: 1D np.array of false and true
        :return: concatenated fudged and original audio components
        """
        # mask: array of false and true, length of num_components
        # get components for true and fudged for false
        temp = np.array([])
        for index, value in enumerate(mask):
            if value:
                temp = np.append(temp, np.array(self.components[index]))
            else:
                temp = np.append(temp, np.array(self.fudged_components[index]))
        return temp

    def get_loudness_components(self, mask):
        """
        get components of the power array for mask values
        :param mask: 1D np.array of false and true
        :return: 1d np.array of original audio/power level length, 0 for false mask values,
                    original power for true mask values
        """
        temp = np.array([])
        indices = [0] + self.indices_components + [np.size(self.audio)]
        for index, value in enumerate(mask):
            if value:
                temp = np.append(temp, np.array(self.loudness[indices[index]:indices[index+1]]))
            else:
                temp = np.append(temp, np.array(self.fudged_components[index]))
        return temp

    def return_components(self, indices, loudness=False):
        """
        return audio array for given component indices, all other components set to 0
        :param indices: list of indices for which to return the original audio components
        :param loudness: bool, if true also returns the loudness values corresponding to indices
        :return: audio(, loudness)
        """
        # make mask setting true for indices
        mask = np.zeros((self.num_components,)).astype(bool)
        mask[indices] = True
        audio = self.get_components_mask(mask)
        if loudness:
            loudness = self.get_loudness_components(mask)
            return audio, loudness
        return audio

    def get_loudness_indices(self):
        """
        calculate the indices of the audio array that correspond to minima below self.threshold of the power array
        :return: indices, loudness power array
        """
        min_length = self.min_length
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
            elif (i < (len(minima) - 1)) and (int(indices[int(m + 1)]) - int(indices[int(m)]) < min_length):
                to_delete.append(i)
        minima = np.delete(minima, to_delete)
        indices_min = []
        for k in minima:
            indices_min.append(int(indices[int(k)]))
        return indices_min, loudness

    def compute_power_db(self, win_len_sec=0.1, power_ref=10**(-12)):
        """Computation of the signal power in dB

        Notebook: C1/C1S3_Dynamics.ipynb, retrieved from
                    https://www.audiolabs-erlangen.de/resources/MIR/FMP/C1/C1S3_Dynamics.html
                    Accessed 28.09.2021

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
        power_db[np.where(power_db == -np.inf)] = 0  # added, errors in calculation because of division by 0
        power_db = np.abs(power_db)
        return power_db

    def return_mask_boundaries(self, positive_indices, negative_indices):
        """
        calculates a mask for highlighting selected components in an image
        :param positive_indices: indices of components with positive weights to include
        :param negative_indices: indices of components with negative weights to include
        :return: 2d array, set to 1 for components with positive weights and to -1 for negative weights, mask to use
        with scikit.mark_boundaries
        """
        mask = np.zeros(np.shape(self.audio), dtype=np.byte)
        indices = [0] + self.indices_components + [np.size(self.audio)]
        for i in range(len(indices) - 1):
            if i in positive_indices:
                mask[indices[i] + 1:indices[i + 1] - 1] = 1
            elif i in negative_indices:
                mask[indices[i] + 1:indices[i + 1] - 1] = -1
        return mask

    def return_weighted_components(self, used_features, weights):
        """
        return audio with loudness components weighted according to their absolute importance
        :param used_features: array of indices of features to include
        :param weights: array of their corresponding weights
        :return: 1d array with weighted audio
            """
        # normalize weights
        sum_weights = np.sum(np.abs(weights))
        weights = np.abs(weights) / sum_weights
        mask_weights = np.zeros((self.num_components,))
        mask_include = np.zeros((self.num_components,)).astype(bool)
        # make weighted sum
        for index, feature in enumerate(used_features):
            mask_weights[feature] = weights[index]
            mask_include[feature] = True

        temp = np.array([])
        for index, value in enumerate(mask_include):
            if value:
                temp = np.append(temp, mask_weights[index] * np.array(self.components[index]))
            else:
                temp = np.append(temp, np.array(self.fudged_components[index]))
        return temp

    def visualize_decomp(self, save_path=None):
        """
        visualize the calculated decomposition and the loudness level
        :param save_path: if not None, path for where to save the generated figure
        """
        audio = self.audio
        indices = [0] + self.indices_components + [np.size(audio)]
        loudness = self.loudness
        fig, (ax1, ax2) = plt.subplots(2)
        fig.suptitle('Loudness Decomposition')
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
