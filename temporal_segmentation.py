import numpy as np


class TemporalSegmentation(object):
    def __init__(self, audio, num_segments, segmentation_type='zero'):
        """audio: np array, (n,) for mono; (n, 2) for stereo
        number_components: integer with number of temporal components
        type: zero for adding zeros to unused components, mean for taking the mean value"""
        self.num_segments = num_segments
        self.audio = audio
        self.segmentation_type = segmentation_type
        length_audio = np.shape(audio)[0]
        if length_audio % num_segments == 0:
            shape_segments = (int(length_audio/num_segments), num_segments)
        else:
            shape_segments = (int(length_audio/num_segments) + 1, num_segments)
        self.segments = np.zeros(shape_segments)
        self.fudged_segments = np.zeros(shape_segments)
        # segments are stored in (n, 1, num_segments)
        self.initialize_segments()
        if segmentation_type == 'fudged':
            self.initialize_fudged_segments()

    def get_number_segments(self):
        return self.num_segments

    def initialize_segments(self):
        # works only for mono audio for now, adapt for stereo as well
        length_audio = np.shape(self.audio)[0]
        pad_size = np.prod(np.shape(self.segments)) - length_audio
        self.segments = np.pad(self.audio, (0, pad_size)).reshape(np.shape(self.segments), order='F')

    def initialize_fudged_segments(self):
        print("still needs to be implemented")

    # make function that returns the combined array for a mask input
    def get_segments_mask(self, mask):
        # mask: array of false and true, length of num_segments
        # get segments for true and fudged for false
        if len(mask) != self.num_segments:
            print('Error: mask has incorrect length')
        mask = np.array(mask)
        temp = np.array(self.fudged_segments, copy=True)
        temp[:, mask] = self.segments[:, mask]
        # TODO: reshape
        segment_shape = np.shape(self.segments)
        temp_flattened = temp.reshape((segment_shape[0] * self.num_segments, ), order='F')
        return temp_flattened

    def return_segments(self, indices):
        # make mask setting true for indices
        mask = np.zeros((self.num_segments,)).astype(bool)
        mask[indices] = True
        audio = self.get_segments_mask(mask)
        return audio
