import numpy as np
import librosa
import matplotlib.pyplot as plt


class NMFDecomposition(object):
    """ decomposes the cough audio array according Non-negative matrix factorization components"""
    def __init__(self, audio, fs, num_components=6):
        """
        Init function
        :param audio: np.array((n,)), audio to be decomposed
        :param fs: int, sample rate
        :param num_components: int, number of components to generate with NMF
        """
        # TODO
        self.audio = audio
        self.fs = fs
        self.decomposition_type = 'nmf'
        # components are stored in (n, 1, num_components)
        self.num_components = num_components
        self.spectral, self.temporal, self.phase = self.initialize_components()

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
        s = librosa.stft(self.audio)
        x, x_phase = librosa.magphase(s)
        spectral, temporal = librosa.decompose.decompose(x, n_components=self.num_components)
        return spectral, temporal, x_phase

    def get_components_mask(self, mask):
        """
        return components for a mask, set to original audio component for true and fudged for false
        :param mask: 1D np.array of false and true
        :return: concatenated fudged and original audio components
        """
        # TODO
        # mask: array of false and true, length of num_components
        # get components for true and fudged for false
        # check if only one value is set to True
        if mask.sum() == 1:
            # only one component, need to calculate slighty different
            reconstructed_x = np.outer(self.spectral[:, mask], self.temporal[mask, :]) * self.phase
        else:
            reconstructed_x = np.dot(self.spectral[:, mask], self.temporal[mask, :]) * self.phase
        reconstructed_audio = librosa.istft(reconstructed_x)
        return reconstructed_audio

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

    def return_weighted_components(self, used_features, weights):
        """
        return audio with loudness components weighted according to their absolute importance
        :param used_features: array of indices of features to include
        :param weights: array of their corresponding weights
        :return: 1d array with weighted audio
            """
        # TODO
        """# normalize weights
        sum_weights = np.sum(np.abs(weights))
        weights = np.abs(weights) / sum_weights
        mask_weights = np.zeros((self.num_components,))
        mask_include = np.zeros((self.num_components,)).astype(bool)
        # make weighted sum
        for index, feature in enumerate(used_features):
            mask_weights[feature] = weights[index]
            mask_include[feature] = True"""
        print("Still needs to be implemented")

    def visualize_decomp(self, save_path=None):
        """
        visualize the calculated decomposition and the loudness level
        :param save_path: if not None, path for where to save the generated figure
        """
        # spectral profiles
        plt.figure(figsize=(13, 7))
        logw = np.log10(self.spectral)
        for n in range(self.num_components):
            plt.subplot(np.ceil(self.num_components/2.0), 2, n + 1)
            plt.plot(logw[:, n])
            plt.ylim(-3, logw.max())
            plt.xlim(0, self.spectral.shape[0])
            plt.ylabel(f'Component {n}')
        if save_path is not None:
            plt.savefig(f'{save_path}/nmf_spectral.png')
        plt.show()
        # temporal activations
        plt.figure(figsize=(13, 7))
        for n in range(self.num_components):
            plt.subplot(np.ceil(self.num_components/2.0), 2, n + 1)
            plt.plot(self.temporal[n])
            plt.ylim(0, self.temporal.max())
            plt.xlim(0, self.temporal.shape[1])
            plt.ylabel(f'Component {n}')
        if save_path is not None:
            plt.savefig(f"{save_path}/nmf_temporal.png")
