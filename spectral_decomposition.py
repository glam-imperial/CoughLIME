import numpy as np
import librosa
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries


class SpectralDecomposition(object):
    """ decomposes the cough audio array into equally sized spectral components"""
    def __init__(self, audio, sample_rate, num_components):
        """
        Init function
        :param audio: np.array((n,)), audio to be decomposed
        :param sample_rate: int, sample rate of audio
        :param num_components: int, number of components to be generated
        """
        self.num_components = num_components
        self.audio = audio
        self.sample_rate = sample_rate
        self.decomposition_type = 'spectral'
        # components are stored in 3d numpy array of shape (num_components, 128, length_spectrogram)
        self.initialize_components()

    def get_number_components(self):
        """
        :return: int, number of components generated during the decomposition
        """
        return self.num_components

    def initialize_components(self):
        """
        caculates the spectrogram of the audio and divides audio into components along spectral axis
        store the spectogram in components array of size (num_components, 128, n) with for each [num_component, :, :]
        :return: nothing, stores components in self.components
        """
        spectrogram = librosa.feature.melspectrogram(y=self.audio, sr=self.sample_rate, n_mels=128)
        shape_components = (self.num_components,) + np.shape(spectrogram)
        self.components = np.zeros(shape_components)
        if 128 % self.num_components == 0:
            len_component = 128 / self.num_components
            for i in range(self.num_components):
                self.components[i, i*len_component:(i+1)*len_component, :] = spectrogram[i*len_component:(i+1)*len_component, :]
        else:
            len_component = int(128 / self.num_components + 1)
            for i in range(self.num_components - 1):
                self.components[i, i*len_component:(i+1)*len_component, :] = spectrogram[i*len_component:(i+1)*len_component, :]
            # last component
            self.components[self.num_components-1, (self.num_components-1)*len_component:, :] = spectrogram[(self.num_components-1)*len_component, :]

    # make function that returns the combined array for a mask input
    def get_components_mask(self, mask):
        """
        return components for a mask, set to original audio component for true and fudged for false
        :param mask: 1D np.array of false and true
        :return: concatenated fudged and original audio components
        """
        # mask: array of false and true, length of num_components
        # get components for true and fudged for false
        if len(mask) != self.num_components:
            print('Error: mask has incorrect length')
        mask = np.array(mask)
        combined_spec = np.sum(self.components[mask, :, :], axis=0)
        reconstructed_audio = librosa.feature.inverse.mel_to_audio(combined_spec, sr=self.sample_rate)
        return reconstructed_audio

    def return_components(self, indices):
        """
        return audio array for given component indices, all other components set to 0
        :param indices: list of indices for which to return the original audio components
        :return: 1d np audio array
        """
        # make mask setting true for indices
        mask = np.zeros((self.num_components,)).astype(bool)
        mask[indices] = True
        audio = self.get_components_mask(mask)
        return audio

    def return_spectrogram_indices(self, indices):
        """
        returns the spectrogram with only the components indices specified
        :param indices: list of indices for which to return the original spectral components
        :return: 2d array, combined spectrogram with selected components only
        """
        mask = np.zeros((self.num_components,)).astype(bool)
        mask[indices] = True
        combined_spec = np.sum(self.components[mask, :, :], axis=0)
        return combined_spec

    def return_mask_boundaries(self, positive_indices, negative_indices):
        """
        calculates a mask for highlighting selected components in an image
        :param positive_indices: indices of components with positive weights
        :param negative_indices: indices of components with negative weights
        :return: 2d array, set to 1 for components with positive weights and to -1 for negative weights
        """
        mask = np.zeros(np.shape(self.components[0, :, :]), dtype=np.byte)
        if 128 % self.num_components == 0:
            len_component = 128 / self.num_components
        else:
            len_component = int(128 / self.num_components + 1)

        for i in range(self.num_components):
            if i in positive_indices:
                mask[(i*len_component+1):((i+1)*len_component-1), 1:-1] = 1
            elif i in negative_indices:
                mask[(i*len_component+1):((i+1)*len_component-1), 1:-1] = -1
        if 128 % self.num_components != 0:
            # last component
            if (self.num_components - 1) in positive_indices:
                mask[((self.num_components-1)*len_component+1):-1, 1:-1] = 1
            elif (self.num_components - 1) in negative_indices:
                mask[((self.num_components-1)*len_component+1):-1, 1:-1] = -1
        return mask

    def return_weighted_components(self, used_features, weights):
        """
        return audio with spectral components weighted according to their absolute importance
        :param used_features: array of indices of features to include
        :param weights: array of their corresponding weights
        :return: 1d array with weighted audio
        """
        # normalize weights
        sum_weights = np.sum(np.abs(weights))
        weights = weights / sum_weights
        mask_weights = np.zeros((self.num_components,))
        # make weighted sum
        for index, feature in enumerate(used_features):
            mask_weights[feature] = weights[index]
        weighted_spectrogram = np.zeros(np.shape(self.components[0, :, :]))
        for comp in range(self.num_components):
            if mask_weights[comp] != 0:
                weighted_spectrogram += abs(mask_weights[comp]) * self.components[comp, :, :]
        reconstructed_audio = librosa.feature.inverse.mel_to_audio(weighted_spectrogram, sr=self.sample_rate)
        return reconstructed_audio

    def visualize_decomp(self, save_path=None):
        """
        visualizes the generated spectral decomposition
        :param save_path: if not None, path to save the generated figure
        """
        spectrogram_indices = range(self.num_components)
        mask = self.return_mask_boundaries(spectrogram_indices, [])
        spectrogram = self.return_spectrogram_indices(spectrogram_indices)
        spec_db = librosa.power_to_db(spectrogram, ref=np.max)

        marked = mark_boundaries(spec_db, mask)
        plt.imshow(marked[:, :, 2], origin="lower", cmap=plt.get_cmap("magma"))
        plt.colorbar(format='%+2.0f dB')

        plt.xlabel("Time")
        plt.ylabel("Frequency")
        ax = plt.gca()
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        plt.title("Spectral Decomposition")
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()
        plt.close()
