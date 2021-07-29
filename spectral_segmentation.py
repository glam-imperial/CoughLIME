import numpy as np
import librosa


class SpectralSegmentation(object):
    def __init__(self, audio, sample_rate, num_segments, segmentation_type='mel_spectrogram'):
        """audio:np array of shape (n,) -> mono audio
        """
        self.num_segments = num_segments
        self.audio = audio
        self.sample_rate = sample_rate
        self.segmentation_type = segmentation_type
        self.initialize_segments()

    def get_number_segments(self):
        return self.num_segments

    def initialize_segments(self):
        """
        get the spectrogram, and make num_segments out of 128 (which is the used n_mels)
        store the spectogram in segments array of size (num_segments, 128, n) with for each [num_segment, :, :]
        everything but segment set to 0
        store this in self.segments
        """
        spectrogram = librosa.feature.melspectrogram(y=self.audio, sr=self.sample_rate, n_mels=128)
        shape_segments = (self.num_segments,) + np.shape(spectrogram)
        self.segments = np.zeros(shape_segments)
        if 128 % self.num_segments == 0:
            len_component = 128 / self.num_segments
            for i in range(self.num_segments):
                self.segments[i, i*len_component:(i+1)*len_component, :] = spectrogram[i*len_component:(i+1)*len_component, :]
        else:
            len_component = int(128 / self.num_segments + 1)
            for i in range(self.num_segments - 1):
                self.segments[i, i*len_component:(i+1)*len_component, :] = spectrogram[i*len_component:(i+1)*len_component, :]
            # last component
            self.segments[self.num_segments-1, (self.num_segments-1)*len_component:, :] = spectrogram[(self.num_segments-1)*len_component, :]

    def initialize_fudged_segments(self):
        print("still needs to be implemented")

    # make function that returns the combined array for a mask input
    def get_segments_mask(self, mask):
        # mask: array of false and true, length of num_segments
        # get segments for true and fudged for false
        if len(mask) != self.num_segments:
            print('Error: mask has incorrect length')
        mask = np.array(mask)
        combined_spec = np.sum(self.segments[mask, :, :], axis=0)
        reconstructed_audio = librosa.feature.inverse.mel_to_audio(combined_spec, sr=self.sample_rate)
        return reconstructed_audio

    def return_segments(self, indices):
        # make mask setting true for indices
        mask = np.zeros((self.num_segments,)).astype(bool)
        mask[indices] = True
        audio = self.get_segments_mask(mask)
        return audio

    def return_spectrogram_indices(self, indices):
        mask = np.zeros((self.num_segments,)).astype(bool)
        mask[indices] = True
        combined_spec = np.sum(self.segments[mask, :, :], axis=0)
        return combined_spec

    def return_mask_boundaries(self, positive_indices, negative_indices):
        mask = np.zeros(np.shape(self.segments[0, :, :]), dtype=np.byte)
        if 128 % self.num_segments == 0:
            len_component = 128 / self.num_segments
            for i in range(self.num_segments):
                if i in positive_indices:
                    mask[(i*len_component+1):((i+1)*len_component-1), 1:-1] = 1
                elif i in negative_indices:
                    mask[(i*len_component+1):((i+1)*len_component-1), 1:-1] = 2
        else:
            len_component = int(128 / self.num_segments + 1)
            for i in range(self.num_segments - 1):
                if i in positive_indices:
                    mask[(i*len_component+1):((i+1)*len_component-1), 1:-1] = 1
                elif i in negative_indices:
                    mask[(i*len_component+1):((i+1)*len_component-1), 1:-1] = -1
            # last component
            if (self.num_segments - 1) in positive_indices:
                mask[((self.num_segments-1)*len_component+1):-1, 1:-1] = 1
            elif (self.num_segments - 1) in negative_indices:
                mask[((self.num_segments-1)*len_component+1):-1, 1:-1] = -1

        return mask

    def return_weighted_segments(self, used_features, weights):
        """
        used_features: array of indices of features to include. weights: array of their corresponding weights
        weights[i] is weight of feature[i]
        """
        # normalize weights
        sum_weights = np.sum(np.abs(weights))
        weights = weights / sum_weights
        mask_weights = np.zeros((self.num_segments,))
        # make weighted sum
        for index, feature in enumerate(used_features):
            mask_weights[feature] = weights[index]
        weighted_spectrogram = np.zeros(np.shape(self.segments[0, :, :]))
        for comp in range(self.num_segments):
            if mask_weights[comp] != 0:
                weighted_spectrogram += abs(mask_weights[comp]) * self.segments[comp, :, :]
        reconstructed_audio = librosa.feature.inverse.mel_to_audio(weighted_spectrogram, sr=self.sample_rate)
        return reconstructed_audio
