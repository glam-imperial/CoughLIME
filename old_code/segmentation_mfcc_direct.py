import numpy as np
import librosa


class MFCCDirectSegmentation(object):
    def __init__(self, audio, sample_rate, num_segments, segmentation_type='mfcc', config=None):
        """audio:np array of shape (n,) -> mono audio
        """
        self.num_segments = num_segments
        self.audio = audio
        self.sample_rate = sample_rate
        self.segmentation_type = segmentation_type
        self.config = config
        self.initialize_segments()

    def get_number_segments(self):
        return self.num_segments

    def initialize_segments(self):
        """
        get the mfcc, and make num_segments out of 20 (which is the used n_mels)
        store the spectogram in segments array of size (num_segments, 128, n) with for each [num_segment, :, :]
        everything but segment set to 0
        store this in self.segments
        """
        #don't forget to add the deltas, how big is array with deltas?
        mfcc = librosa.feature.mfcc(self.audio, sr=int(self.config['default']['sampling_rate']),
                                    n_mfcc=int(self.config['mfcc']['n_mfcc']),
                                    n_fft=int(self.config['default']['window_size']),
                                    hop_length=int(self.config['default']['window_shift']),
                                    n_mels=int(self.config['mfcc']['n_mels']),
                                    fmax=int(self.config['mfcc']['fmax']))
        self.num_mfcc = np.shape(mfcc)[0]
        if self.config['mfcc']['add_deltas'] in ['True', 'true', 'TRUE', '1']:
            deltas = librosa.feature.delta(mfcc)
            features = np.concatenate((mfcc, deltas), axis=0)

        if self.config['mfcc']['add_delta_deltas'] in ['True', '∞true', 'TRUE', '1']:
            ddeltas = librosa.feature.delta(mfcc, order=2)
            mfcc = np.concatenate((features, ddeltas), axis=0)

        shape_segments = (self.num_segments,) + np.shape(mfcc)
        self.segments = np.zeros(shape_segments)
        if self.num_mfcc % self.num_segments == 0:
            len_component = self.num_mfcc / self.num_segments
            for i in range(self.num_segments):
                self.segments[i, i*len_component:(i+1)*len_component, :] = mfcc[i*len_component:(i+1)*len_component, :]
                if self.config['mfcc']['add_deltas'] in ['True', 'true', 'TRUE', '1']:
                    self.segments[i, i*len_component+39:(i+1)*len_component+39, :] = mfcc[i*len_component+39:(i+1)*len_component+39, :]
                if self.config['mfcc']['add_delta_deltas'] in ['True', '∞true', 'TRUE', '1']:
                    self.segments[i, i*len_component+78:(i+1)*len_component+78, :] = mfcc[i*len_component+78:(i+1)*len_component+78, :]
        else:
            len_component = int(self.num_mfcc / self.num_segments + 1)
            for i in range(self.num_segments - 1):
                self.segments[i, i*len_component:(i+1)*len_component, :] = mfcc[i*len_component:(i+1)*len_component, :]
                if self.config['mfcc']['add_deltas'] in ['True', 'true', 'TRUE', '1']:
                    self.segments[i, i*len_component+39:(i+1)*len_component+39, :] = mfcc[i*len_component+39:(i+1)*len_component+39, :]
                if self.config['mfcc']['add_delta_deltas'] in ['True', '∞true', 'TRUE', '1']:
                    self.segments[i, i*len_component+78:(i+1)*len_component+78, :] = mfcc[i*len_component+78:(i+1)*len_component+78, :]
            # last component
            self.segments[self.num_segments-1, (self.num_segments-1)*len_component:-78, :] = mfcc[(self.num_segments-1)*len_component:-78, :]
            if self.config['mfcc']['add_deltas'] in ['True', 'true', 'TRUE', '1']:
                self.segments[self.num_segments-1, (self.num_segments-1)*len_component+39:-39, :] = mfcc[(self.num_segments-1)*len_component+39:-39, :]
            if self.config['mfcc']['add_delta_deltas'] in ['True', '∞true', 'TRUE', '1']:
                self.segments[self.num_segments-1, (self.num_segments-1)*len_component+78:, :] = mfcc[(self.num_segments-1)*len_component+78:, :]

    def initialize_fudged_segments(self):
        print("still needs to be implemented")

    # make function that returns the combined array for a mask input
    def get_segments_mask(self, mask):
        # mask: array of false and true, length of num_segments
        # get segments for true and fudged for false
        # sum(arr1[mask, :, :])
        # retransform to audio with librosa
        if len(mask) != self.num_segments:
            print('Error: mask has incorrect length')
        mask = np.array(mask)
        combined_mfcc = np.sum(self.segments[mask, :, :], axis=0)
        return combined_mfcc

    def return_segments(self, indices):
        # make mask setting true for indices
        mask = np.zeros((self.num_segments,)).astype(bool)
        mask[indices] = True
        mfcc = self.get_segments_mask(mask)
        return mfcc

    def return_audio_from_indices(self, indices):
        mfcc = self.return_segments(indices)
        audio = librosa.feature.inverse.mfcc_to_audio(mfcc, sr=self.sample_rate)
        return audio

    def return_spectrogram_indices(self, indices):
        mask = np.zeros((self.num_segments,)).astype(bool)
        mask[indices] = True
        combined_mfcc = np.sum(self.segments[mask, :, :], axis=0)
        return combined_mfcc

    def return_mask_boundaries(self, positive_indices, negative_indices):
        mask = np.zeros(np.shape(self.segments[0, :, :]), dtype=np.byte)
        if self.num_mfcc % self.num_segments == 0:
            len_component = self.num_mfcc / self.num_segments
            for i in range(self.num_segments):
                if i in positive_indices:
                    mask[(i*len_component+1):((i+1)*len_component-1), 1:-1] = 1
                    if self.config['mfcc']['add_deltas'] in ['True', 'true', 'TRUE', '1']:
                        mask[i*len_component+40:(i+1)*len_component+38, 1:-1] = 1
                    if self.config['mfcc']['add_delta_deltas'] in ['True', '∞true', 'TRUE', '1']:
                        mask[i*len_component+79:(i+1)*len_component+77, 1:-1] = 1
                elif i in negative_indices:
                    mask[(i*len_component+1):((i+1)*len_component-1), 1:-1] = -1
                    if self.config['mfcc']['add_deltas'] in ['True', 'true', 'TRUE', '1']:
                        mask[i*len_component+40:(i+1)*len_component+38, 1:-1] = -1
                    if self.config['mfcc']['add_delta_deltas'] in ['True', '∞true', 'TRUE', '1']:
                        mask[i*len_component+79:(i+1)*len_component+77, 1:-1] = -1
        else:
            len_component = int(self.num_mfcc / self.num_segments + 1)
            for i in range(self.num_segments - 1):
                if i in positive_indices:
                    mask[(i*len_component+1):((i+1)*len_component-1), 1:-1] = 1
                    if self.config['mfcc']['add_deltas'] in ['True', 'true', 'TRUE', '1']:
                        mask[i*len_component+40:(i+1)*len_component+38, 1:-1] = 1
                    if self.config['mfcc']['add_delta_deltas'] in ['True', '∞true', 'TRUE', '1']:
                        mask[i*len_component+79:(i+1)*len_component+77, 1:-1] = 1
                elif i in negative_indices:
                    mask[(i*len_component+1):((i+1)*len_component-1), 1:-1] = -1
                    if self.config['mfcc']['add_deltas'] in ['True', 'true', 'TRUE', '1']:
                        mask[i*len_component+40:(i+1)*len_component+38, 1:-1] = -1
                    if self.config['mfcc']['add_delta_deltas'] in ['True', '∞true', 'TRUE', '1']:
                        mask[i*len_component+79:(i+1)*len_component+77, 1:-1] = -1
            # last component
            if (self.num_segments - 1) in positive_indices:
                mask[((self.num_segments-1)*len_component+1):-1, 1:-1] = 1
                if self.config['mfcc']['add_deltas'] in ['True', 'true', 'TRUE', '1']:
                    mask[(self.num_segments-1)*len_component+39:-40, 1:-1] = 1
                if self.config['mfcc']['add_delta_deltas'] in ['True', '∞true', 'TRUE', '1']:
                    mask[(self.num_segments-1)*len_component+79:-1, 1:-1] = 1
            elif (self.num_segments - 1) in negative_indices:
                mask[((self.num_segments-1)*len_component+1):-79, 1:-1] = -1
                if self.config['mfcc']['add_deltas'] in ['True', 'true', 'TRUE', '1']:
                    mask[(self.num_segments-1)*len_component+39:-40, 1:-1] = -1
                if self.config['mfcc']['add_delta_deltas'] in ['True', '∞true', 'TRUE', '1']:
                    mask[(self.num_segments-1)*len_component+79:-1, 1:-1] = -1

        return mask
