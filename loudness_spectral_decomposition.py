from scipy.signal import argrelextrema
import numpy as np
import itertools
import librosa
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries


class LoudnessSpectralDecomposition(object):
    def __init__(self, audio, fs, threshold=75, num_spectral=5):
        """
        init function for Loudness-spectral decomposition
        :param audio: numpy array, audio signal
        :param fs: int, sample rate
        :param threshold: int, threshold in power level to generate new loudness component
        :param num_spectral: int, number of spectral components per loudness component
        """
        self.audio = audio
        self.fs = fs
        self.threshold = threshold
        self.num_spectral = num_spectral
        self.decomposition_type = 'ls'
        # components stored in a list of length num_loudness. Each element contains the corresponding spectral
        # components in a 3d numpy array. shape: (num_spectral, 128, length_spectrogram)
        self.components, self.indices_components, self.loudness = self.initialize_components()
        self.num_loudness = len(self.components)
        self.num_components = self.num_loudness * self.num_spectral
        self.fudged_components = self.initialize_fudged_components()

    def initialize_components(self):
        """
        initializes the components decomposed according to loudness and in further spectral components
        :return: list, length is number of loudness components
                    each element is a numpy array of shape (num_spectral_comp, spectrogram_length, 128) containing
                    the spectral components for the audio file
        """
        audio = self.audio
        indices_min, loudness = self.get_loudness_indices()
        temp_components = []
        previous = 0
        current_index = 0
        audio_length = np.size(audio)
        number_indices = np.size(indices_min)
        if number_indices == 0:
            print("No minimum detected, initializing one component")
            temp_components.append(self.get_spectral_comp(audio))
        else:
            while previous < audio_length and current_index < number_indices:
                temp_components.append(self.get_spectral_comp(audio[previous:indices_min[current_index]]))
                previous = indices_min[current_index]
                current_index += 1
            temp_components.append(self.get_spectral_comp(audio[previous:]))
        return temp_components, indices_min, loudness

    def get_loudness_indices(self, min_length=4096):
        """
        calculate the indices of the audio array that correspond to minima below self.threshold of the power array
        :param min_length: minimum length for each loudness component (to not break the spectrogram retransformation)
        :return: indices, loudness power array
        """
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
            elif (i < (len(minima) - 1)) and (int(indices[int(minima[i+1])]) - int(indices[int(m)]) < min_length):
                to_delete.append(i)
        minima = np.delete(minima, to_delete)
        indices_min = []
        for k in minima:
            indices_min.append(int(indices[int(k)]))
        return indices_min, loudness

    def get_spectral_comp(self, audio):
        """
        decomposes a given audio array into its spectral components
        :param audio: np.array, audio
        :return: np.array((num_spectral_comp, spectrogram_length, 128))
        """
        spectrogram = librosa.feature.melspectrogram(y=audio, sr=self.fs, n_mels=128)
        shape_components = (self.num_spectral,) + np.shape(spectrogram)
        temp = np.zeros(shape_components)
        if 128 % self.num_spectral == 0:
            length = 128 / self.num_spectral
            for i in range(self.num_spectral):
                temp[i, i*length:(i+1)*length, :] = spectrogram[i*length:(i+1)*length, :]
        else:
            length = int(128 / self.num_spectral + 1)
            for i in range(self.num_spectral - 1):
                temp[i, i*length:(i+1)*length, :] = spectrogram[i*length:(i+1)*length, :]
            # last component
            temp[self.num_spectral-1, (self.num_spectral-1)*length:, :] = spectrogram[(self.num_spectral - 1)*length, :]
        return temp

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
        power_db[np.where(power_db == -np.inf)] = 0  # accounts for errors in calculation because of division by 0
        power_db = np.abs(power_db)
        return power_db

    def initialize_fudged_components(self):
        """
        initializes the fudged components that are needed for training the linear model
        :return: array of same shape as original components, all set to 0
        """
        temp = []
        for comp in self.components:
            temp.append(np.zeros(np.shape(comp)))
        return temp

    def get_number_components(self):
        """
        :return: int, number of components generated during the decomposition
        """
        return self.num_components

    def get_components_mask(self, mask, spec=False):
        """
        return components for a mask, set to original audio component for true and fudged for false
        :param mask: 1D np.array of false and true
        :param spec: bool, return spectrogram if true, return converted audio array if false
        :return: concatenated fudged and original audio components
        """
        # mask: array of false and true, length of num_components
        # get components for true and fudged for false
        indices = list(range(1, self.num_loudness))
        current_mask = mask[0:self.num_spectral]
        temp = np.sum(self.components[0][current_mask, :, :], axis=0)
        for i in indices:
            current_mask = mask[i*self.num_spectral:(i+1)*self.num_spectral]
            combined_spec = np.sum(self.components[i][current_mask, :, :], axis=0)
            temp = np.concatenate((temp, combined_spec), axis=1)
        if spec:
            return temp
        temp_audio = librosa.feature.inverse.mel_to_audio(temp)
        return temp_audio

    def return_components(self, indices):
        """
        return audio array for given component indices, all other components set to 0
        :param indices: list of indices for which to return the original audio components
        :return: audio
        """
        # make mask setting true for indices
        mask = np.zeros((self.num_components,)).astype(bool)
        mask[indices] = True
        audio = self.get_components_mask(mask)
        return audio

    def return_mask_boundaries(self, positive_indices, negative_indices):
        """
        makes a mask with the indices marking the location of all components in positive_indices set to 1 and the
        negative components to -1
        :param positive_indices: list, contains indices of positive components
        :param negative_indices: list, contains indices of negative components
        :return: 2d numpy array, marks components in spectrogram
        """
        temp = self.components[0][0, :, :]
        indices_l = [0, np.shape(temp)[1]]
        for i in range(1, self.num_loudness):
            combined_spec = self.components[i][0, :, :]
            temp = np.concatenate((temp, combined_spec), axis=1)
            indices_l.append(np.shape(temp)[1])
        mask = np.zeros(np.shape(temp), dtype=np.byte)
        # self.components, self.indices_components
        # self.num_spectral, self.num_loudness
        if 128 % self.num_spectral == 0:
            len_s = 128 / self.num_spectral
        else:
            len_s = int(128 / self.num_spectral + 1)

        for i in range(self.num_components):
            if i in positive_indices:
                i_s = i % self.num_spectral
                i_l = int(i / self.num_spectral)

                if (i_s == (self.num_spectral - 1)) and (i_l == (self.num_loudness - 1)):
                    # last comps
                    mask[(i_s*len_s+1):-1, indices_l[i_l]+1:-1] = 1
                elif i_s == (self.num_spectral - 1):
                    mask[i_s*len_s+1:-1, indices_l[i_l]+1:indices_l[i_l+1]-1] = 1
                elif i_l == (self.num_loudness - 1):
                    mask[i_s*len_s+1:((i_s+1)*len_s-1), indices_l[i_l]+1:-1] = 1
                else:
                    mask[i_s*len_s+1:((i_s+1)*len_s-1), indices_l[i_l]+1:indices_l[i_l+1]-1] = 1
            elif i in negative_indices:
                i_s = i % self.num_spectral
                i_l = int(i / self.num_spectral)

                if (i_s == (self.num_spectral - 1)) and (i_l == (self.num_loudness - 1)):
                    # last comps
                    mask[(i_s*len_s+1):-1, indices_l[i_l]+1:-1] = 1
                elif i_s == (self.num_spectral - 1):
                    mask[i_s*len_s+1:-1, indices_l[i_l]+1:indices_l[i_l+1]-1] = -1
                elif i_l == (self.num_loudness - 1):
                    mask[i_s*len_s+1:((i_s+1)*len_s-1), indices_l[i_l]+1:-1] = -1
                else:
                    mask[i_s*len_s+1:((i_s+1)*len_s-1), indices_l[i_l]+1:indices_l[i_l+1]-1] = -1

        return mask

    def visualize_decomp(self, save_path=None):
        """
        visualizes the generated components in a spectrogram image
        :param save_path: if not None, path to save the generated image
        :return: nothing, shows and possibly saves image
        """
        indices = [0] + self.indices_components + [np.size(self.audio)]
        loudness = self.loudness
        fig, (ax1, ax2) = plt.subplots(2, figsize=(6, 6), gridspec_kw={'height_ratios': [2, 1]})
        # get combined spectrogram
        # get a mask for the spectrogram
        spectrogram_indices = range(self.num_components)
        mask = self.return_mask_boundaries(spectrogram_indices, [])
        mask_s = np.ones(self.num_components).astype(bool)
        spectrogram = self.get_components_mask(mask_s, spec=True)
        spec_db = librosa.power_to_db(spectrogram, ref=np.max)

        marked = mark_boundaries(spec_db, mask)
        img = ax1.imshow(marked[:, :, 2], origin="lower", cmap=plt.get_cmap("magma"))
        fig.colorbar(img, ax=ax1)
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Frequency")
        ax1.axes.xaxis.set_ticks([])
        ax1.axes.yaxis.set_ticks([])
        fig.suptitle('Spectral-Loudness Decomposition with 5\nspectral components and threshold=75')

        ax2.plot(loudness, color='c')
        for line in indices:
            ax2.axvline(x=line, color='m')
        ax2.set(xlabel='Time', ylabel='Power (db)', xlim=[0, np.size(self.audio)])
        for i in range(0, len(indices) - 1, 2):
            ax2.axvspan(indices[i], indices[i+1], facecolor='m', alpha=0.1)
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()
        print("visualized :)")

    def return_weighted_components(self, used_features, weights):
        """
        return audio with ls components weighted according to their absolute importance
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

        indices = list(range(1, self.num_loudness))
        current_mask = mask_include[0:self.num_spectral]
        temp = np.zeros(np.shape(np.sum(self.components[0][current_mask, :, :], axis=0)))

        for comp in range(self.num_spectral):
            if mask_weights[comp] != 0:
                temp += abs(mask_weights[comp]) * self.components[0][comp, :, :]
        for i in indices:
            current_mask = mask_include[i*self.num_spectral:(i+1)*self.num_spectral]
            combined_spec = np.zeros(np.shape(np.sum(self.components[i][current_mask, :, :], axis=0)))
            for comp in range(self.num_spectral):
                if mask_weights[comp] != 0:
                    combined_spec += abs(mask_weights[comp]) * self.components[i][comp, :, :]
            temp = np.concatenate((temp, combined_spec), axis=1)
        temp_audio = librosa.feature.inverse.mel_to_audio(temp)
        return temp_audio
