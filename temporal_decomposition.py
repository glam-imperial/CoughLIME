import numpy as np
import matplotlib.pyplot as plt


class TemporalDecomposition(object):
    """ decomposes the cough audio array into equally sized temporal components"""
    def __init__(self, audio, num_components):
        """
        Init function
        :param audio: np.array((n,)), audio to be decomposed
        :param num_components: int, number of components to be generated
        """
        self.num_components = num_components
        self.audio = audio
        self.decomposition_type = 'temporal'
        length_audio = np.shape(audio)[0]
        if length_audio % num_components == 0:
            shape_components = (int(length_audio/num_components), num_components)
        else:
            shape_components = (int(length_audio/num_components) + 1, num_components)
        self.components = np.zeros(shape_components)
        self.fudged_components = np.zeros(shape_components)
        # components are stored in (length_audio/num_components (+1), num_components)
        self.initialize_components()

    def get_number_components(self):
        """
        :return: int, number of components generated during the decomposition
        """
        return self.num_components

    def initialize_components(self):
        """
        initializes the components of the audio array
        :return: nothing, directly stores initialized components in self.components
        """
        # works only for mono audio for now, adapt for stereo as well
        length_audio = np.shape(self.audio)[0]
        pad_size = np.prod(np.shape(self.components)) - length_audio
        self.components = np.pad(self.audio, (0, pad_size)).reshape(np.shape(self.components), order='F')

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
        temp = np.array(self.fudged_components, copy=True)
        temp[:, mask] = self.components[:, mask]
        component_shape = np.shape(self.components)
        temp_flattened = temp.reshape((component_shape[0] * self.num_components, ), order='F')
        return temp_flattened

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

    def return_mask_boundaries(self, positive_indices, negative_indices):
        """
        calculates a mask for highlighting selected components in an image
        :param positive_indices: indices of components with positive weights
        :param negative_indices: indices of components with negative weights
        :return: 2d array, set to 1 for components with positive weights and to -1 for negative weights
        """
        audio = self.audio
        length_audio = np.shape(audio)[0]
        distance = int(length_audio/self.num_components)
        indices = np.array(range(self.num_components))
        indices = indices * distance
        indices = np.append(indices, [length_audio])
        mask = np.zeros(np.shape(self.audio), dtype=np.byte)
        for i in range(len(indices) - 1):
            if i in positive_indices:
                mask[indices[i] + 1:indices[i + 1] - 1] = 1
            elif i in negative_indices:
                mask[indices[i] + 1:indices[i + 1] - 1] = -1
        return mask

    def return_weighted_components(self, used_features, weights):
        """
        return audio with temporal components weighted according to their absolute importance
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

        temp = np.array(self.fudged_components, copy=True)
        for index, val in enumerate(mask_include):
            if val:
                temp[:, index] = mask_weights[index] * self.components[:, index]
        component_shape = np.shape(self.components)
        temp_flattened = temp.reshape((component_shape[0] * self.num_components, ), order='F')
        return temp_flattened

    def visualize_decomp(self, save_path=None):
        """
        visualizes the calculated decomposition
        :param save_path: if not None, path for where to save the generated figure
        """
        audio = self.audio
        length_audio = np.shape(audio)[0]
        distance = int(length_audio/self.num_components)
        indices = np.array(range(self.num_components))
        indices = indices * distance
        indices = np.append(indices, [length_audio])
        plt.rcParams["figure.figsize"] = (8, 5)
        plt.plot(audio, color='c')
        for line in indices:
            plt.axvline(x=line, color='m')
        plt.xlim([0, np.size(audio)])
        plt.title(f"Temporal Decomposition into {self.num_components} Components")
        for i in range(0, len(indices) - 1, 2):
            plt.axvspan(indices[i], indices[i+1], facecolor='m', alpha=0.1)
        plt.ylabel('Amplitude')
        plt.xlabel('Time')
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()
        print("visualized :)")
        plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]
        plt.close()