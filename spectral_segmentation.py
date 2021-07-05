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
        # sum(arr1[mask, :, :])
        # retransform to audio with librosa         reconstructed_audio = librosa.feature.inverse.mel_to_audio(spectrogram, sr=sample_rate)
        if len(mask) != self.num_segments:
            print('Error: mask has incorrect length')
        mask = np.array(mask)
        combined_spec = sum(self.num_segments[mask, :, :])
        reconstructed_audio = librosa.feature.inverse.mel_to_audio(combined_spec, sr=self.sample_rate)
        return reconstructed_audio

    def return_segments(self, indices):
        # make mask setting true for indices
        mask = np.zeros((self.num_segments,)).astype(bool)
        mask[indices] = True
        audio = self.get_segments_mask(mask)
        return audio
