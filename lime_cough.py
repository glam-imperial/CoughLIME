"""
Functions for explaining classifiers that use Image data.
"""
from functools import partial

import numpy as np
import sklearn
from sklearn.utils import check_random_state
from tqdm.auto import tqdm
import librosa
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries


from lime import lime_base

# TODO: update all comments


class CoughExplanation(object):
    def __init__(self, segmentation):
        """Init function.

        Args:
            image: 3d numpy array
            segments: 2d numpy array, with the output from skimage.segmentation
        """

        self.segmentation = segmentation
        self.intercept = {}
        self.local_exp = {}
        self.local_pred = {}
        self.score = {}

    # TODO: function that returns, prints outputs
    #  -> possibly sorted components as in lime audio,
    #  but some graphics as in lime text would be way nicer

    def get_exp_components(self, label, positive_components=True, negative_components=True, num_components='all',
                              min_abs_weight=0.0, return_indeces=False):
        """
        :param label:
        :param positive_components:
        :param negative_components:
        :param num_components:
        :param min_abs_weight:
        :param return_indeces:
        :return: audio that is made of the most important num_components for the explanation
        """
        if label not in self.local_exp:
            raise KeyError('Label not in explanation')
        if positive_components is False and negative_components is False:
            raise ValueError('positive_components, negative_components or both must be True')

        exp = self.local_exp[label]

        w = [[x[0], x[1]] for x in exp]
        used_features, weights = np.array(w, dtype=int)[:, 0], np.array(w)[:, 1]

        if not negative_components:
            pos_weights = np.argwhere(weights > 0)[:, 0]
            used_features = used_features[pos_weights]
            weights = weights[pos_weights]
        elif not positive_components:
            neg_weights = np.argwhere(weights < 0)[:, 0]
            used_features = used_features[neg_weights]
            weights = weights[neg_weights]
        if min_abs_weight != 0.0:
            abs_weights = np.argwhere(abs(weights) >= min_abs_weight)[:, 0]
            used_features = used_features[abs_weights]
            weights = weights[abs_weights]

        if num_components == 'all':
            num_components = len(used_features)
        else:
            assert(isinstance(num_components, int))
            # max_components = used_features[:num_components]

        used_features = used_features[:num_components]
        audio = self.segmentation.return_segments(used_features)
        if return_indeces:
            return audio, used_features
        return audio

    def weighted_audio(self, label, positive_components=True, negative_components=False, num_components='all',
                            min_abs_weight=0.0, return_indeces=False):
        # returns weighted audio (weighted by abs value of components)
        if label not in self.local_exp:
            raise KeyError('Label not in explanation')
        if positive_components is False and negative_components is False:
            raise ValueError('positive_components, negative_components or both must be True')

        exp = self.local_exp[label]

        w = [[x[0], x[1]] for x in exp]
        used_features, weights = np.array(w, dtype=int)[:, 0], np.array(w)[:, 1] # used features is array of components, with array of weights of same length

        if not negative_components:
            pos_weights = np.argwhere(weights > 0)[:, 0]
            used_features = used_features[pos_weights]
            weights = weights[pos_weights]
        elif not positive_components:
            neg_weights = np.argwhere(weights < 0)[:, 0]
            used_features = used_features[neg_weights]
            weights = weights[neg_weights]
        if min_abs_weight != 0.0:
            abs_weights = np.argwhere(abs(weights) >= min_abs_weight)[:, 0]
            used_features = used_features[abs_weights]
            weights = weights[abs_weights]

        if num_components == 'all':
            num_components = len(used_features)
        else:
            assert(isinstance(num_components, int))
            # max_components = used_features[:num_components]

        used_features = used_features[:num_components]
        weights = weights[:num_components]
        audio = self.segmentation.return_weighted_segments(used_features, weights)
        if return_indeces:
            return audio, used_features
        return audio

    def normalize(self, weights):
        abs_weights = np.abs(np.array(weights))
        minimum = min(abs_weights) - 0.2 * max(abs_weights)
        maximum = max(abs_weights) + 0.4 * max(abs_weights)
        normalized = np.zeros(np.shape(abs_weights))
        for i, w in enumerate(abs_weights):
            normalized[i] = (abs_weights[i] - minimum) / (maximum - minimum)  # zi = (xi – min(x)) / (max(x) – min(x))
        return normalized

    def show_image_mask_spectrogram(self, label, positive_only=True, negative_only=False, hide_rest=True, num_features=5, min_weight=0., save_path=None, show_colors=False):
        """
        This only works for spectral decomposition!
        # TODO: implement check if spectral decomposition is selected (probably via self.factorization.argument)
        Args:
            label: label to explain
            positive_only: if True, only take components that positively contribute to
                the prediction of the label.
            negative_only: if True, only take components that negatively contribute to
                the prediction of the label. If false, and so is positive_only, then both
                negative and positive contributions will be taken.
                Both can't be True at the same time
            hide_rest: if True, make the non-explanation part of the return
                image gray
            num_features: number of components to include in explanation
            min_weight: minimum weight of the superpixels to include in explanation
            save_path: path under which to save the obtained image for the explanation
        """
        if label not in self.local_exp:
            raise KeyError('Label not in explanation')
        if positive_only and negative_only:
            raise ValueError("Positive_only and negative_only cannot be true at the same time.")
        segmentation = self.segmentation
        explanation = self.local_exp[label]

        if positive_only:
            indices_comp = [x[0] for x in explanation if x[1] > 0 and x[1] > min_weight][:num_features]
            weights = [x[1] for x in explanation if x[1] > 0 and x[1] > min_weight][:num_features]
            mask = segmentation.return_mask_boundaries(indices_comp, [])
        if negative_only:
            indices_comp = [x[0] for x in explanation if x[1] < 0 and abs(x[1]) > min_weight][:num_features]
            weights = [x[1] for x in explanation if x[1] < 0 and abs(x[1]) > min_weight][:num_features]
            mask = segmentation.return_mask_boundaries([], indices_comp)
        if positive_only or negative_only:
            if hide_rest:
                spectrogram_indices = indices_comp
            else:
                spectrogram_indices = range(segmentation.get_number_segments())
        else:
            comp_pos, comp_neg = [], []
            indices_comp, weights = [], []
            for x in explanation[:num_features]:
                indices_comp.append(x[0])
                weights.append(x[1])
                if x[1] > 0 and x[1] > min_weight:
                    comp_pos.append(x[0])
                elif x[1] < 0 and np.abs(x[1]) > min_weight:
                    comp_neg.append(x[0])
            mask = segmentation.return_mask_boundaries(comp_pos, comp_neg)
            if hide_rest:
                spectrogram_indices = comp_pos + comp_neg
            else:
                spectrogram_indices = range(segmentation.get_number_segments())

        spectrogram = segmentation.return_spectrogram_indices(spectrogram_indices)
        spec_db = librosa.power_to_db(spectrogram, ref=np.max)
        marked = mark_boundaries(spec_db, mask)
        plt.imshow(marked[:, :, 2], origin="lower", cmap=plt.get_cmap("magma"))
        plt.colorbar(format='%+2.0f dB')
        if show_colors:
            normalized_weights = self.normalize(weights)
            for index, comp in enumerate(indices_comp):
                image_array = np.ones(np.shape(mask) + (4,))
                if weights[index] < 0:
                    mask = segmentation.return_mask_boundaries([], [comp])
                else:
                    mask = segmentation.return_mask_boundaries([comp], [])
                mask_negative = np.zeros(np.shape(mask))
                mask_negative[np.where(mask == 0)] = 1
                mask_negative_green = np.ones(np.shape(mask))
                mask_negative_green[np.where(mask == -1)] = 0
                mask_negative_red = np.ones(np.shape(mask))
                mask_negative_red[np.where(mask == 1)] = 0
                image_array[:, :, 0] = mask_negative_red #0 for red, 1 for green
                image_array[:, :, 1] = mask_negative_green
                image_array[:, :, 2] = mask_negative
                image_array[:, :, 3] = np.abs(mask)
                plt.imshow(image_array, origin="lower", interpolation="nearest", alpha=normalized_weights[index])

            # old starting
            """
            image_array = np.ones(np.shape(mask) + (4,))
            mask_negative = np.zeros(np.shape(mask))
            mask_negative[np.where(mask == 0)] = 1
            mask_negative_green = np.ones(np.shape(mask))
            mask_negative_green[np.where(mask == -1)] = 0
            mask_negative_red = np.ones(np.shape(mask))
            mask_negative_red[np.where(mask == 1)] = 0
            image_array[:, :, 0] = mask_negative_red #0 for red, 1 for green
            image_array[:, :, 1] = mask_negative_green
            image_array[:, :, 2] = mask_negative
            image_array[:, :, 3] = np.abs(mask)
            plt.imshow(image_array, origin="lower", interpolation="nearest", alpha=0.5)"""
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        ax = plt.gca()
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        plt.title("Most important components for local\nprediction of class COVID-positive")
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()
        plt.close()


class LimeCoughExplainer(object):
    """Explains predictions on Cough (i.e. matrix) data.
    For numerical features, perturb them by sampling from a Normal(0,1) and
    doing the inverse operation of mean-centering and scaling, according to the
    means and stds in the training data. For categorical features, perturb by
    sampling according to the training distribution, and making a binary
    feature that is 1 when the value is the same as the instance being
    explained."""

    def __init__(self, kernel_width=.25, kernel=None, verbose=False,
                 feature_selection='auto', random_state=None):
        """Init function.

        Args:
            kernel_width: kernel width for the exponential kernel.
            If None, defaults to sqrt(number of columns) * 0.75.
            kernel: similarity kernel that takes euclidean distances and kernel
                width as input and outputs weights in (0,1). If None, defaults to
                an exponential kernel.
            verbose: if true, print local prediction values from linear model
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
                See function 'explain_instance_with_data' in lime_base.py for
                details on what each of the options does.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """
        kernel_width = float(kernel_width)

        if kernel is None:
            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        kernel_fn = partial(kernel, kernel_width=kernel_width)

        self.random_state = check_random_state(random_state)
        self.feature_selection = feature_selection
        self.base = lime_base.LimeBase(kernel_fn, verbose, random_state=self.random_state)

    # TODO: check what feature selection options still work

    def explain_instance(self, segmentation, classifier_fn, labels=(1,), num_samples=1000,
                         batch_size=10,
                         distance_metric='cosine',
                         model_regressor=None,
                         random_seed=None,
                         progress_bar=False):
        """Generates explanations for a prediction.

        First, we generate neighborhood data by randomly perturbing features
        from the instance (see __data_inverse). We then learn locally weighted
        linear models on this neighborhood data to explain each of the classes
        in an interpretable way (see lime_base.py).

        Args:
            image: 3 dimension RGB image. If this is only two dimensional,
                we will assume it's a grayscale image and call gray2rgb.
            classifier_fn: classifier prediction probability function, which
                takes a numpy array and outputs prediction probabilities.  For
                ScikitClassifiers , this is classifier.predict_proba.
            labels: iterable with labels to be explained.
            hide_color: TODO
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            batch_size: TODO
            distance_metric: the distance metric to use for weights.
            model_regressor: sklearn regressor to use in explanation. Defaults
            to Ridge regression in LimeBase. Must have model_regressor.coef_
            and 'sample_weight' as a parameter to model_regressor.fit()
            segmentation_fn: SegmentationAlgorithm, wrapped skimage
            segmentation function
            random_seed: integer used as random seed for the segmentation
                algorithm. If None, a random integer, between 0 and 1000,
                will be generated using the internal random number generator.
            progress_bar: if True, show tqdm progress bar.

        Returns:
            An ImageExplanation object (see lime_image.py) with the corresponding
            explanations.
        """

        if random_seed is None:
            random_seed = self.random_state.randint(0, high=1000)

        top = labels

        num_features = segmentation.get_number_segments()

        data, labels = self.data_labels(segmentation,
                                        classifier_fn, num_samples,
                                        batch_size=batch_size,
                                        progress_bar=progress_bar)

        distances = sklearn.metrics.pairwise_distances(
            data,
            data[0].reshape(1, -1),
            metric=distance_metric
        ).ravel()

        ret_exp = CoughExplanation(segmentation)

        for label in top:  # same in image and audio
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score[label],
             ret_exp.local_pred[label]) = self.base.explain_instance_with_data(
                data, labels, distances, label, num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection)
        return ret_exp

    def data_labels(self,
                    segmentation,
                    classifier_fn,
                    num_samples,
                    batch_size=10,
                    progress_bar=True):
        """Generates images and predictions in the neighborhood of this image.

        Args:
            image: 3d numpy array, the image
            fudged_image: 3d numpy array, image to replace original image when
                superpixel is turned off
            segments: segmentation of the image
            classifier_fn: function that takes a list of images and returns a
                matrix of prediction probabilities
            num_samples: size of the neighborhood to learn the linear model
            batch_size: classifier_fn will be called on batches of this size.
            progress_bar: if True, show tqdm progress bar.

        Returns:
            A tuple (data, labels), where:
                data: dense num_samples * num_superpixels
                labels: prediction probabilities matrix
        """

        n_features = segmentation.get_number_segments()

        data = self.random_state.randint(0, 2, num_samples * n_features) \
            .reshape((num_samples, n_features))

        labels = []
        data[0, :] = 1
        audios = []
        rows = tqdm(data) if progress_bar else data
        for row in rows:
            non_zeros = np.where(row != 0)[0]
            mask = np.zeros((n_features,)).astype(bool)
            mask[non_zeros] = True
            temp = segmentation.get_segments_mask(mask)
            audios.append(temp)
            if len(audios) == batch_size:
                predictions = classifier_fn(np.array(audios))
                labels.extend(predictions)
                audios = []
        if len(audios) > 0:
            predictions = classifier_fn(np.array(audios))
            labels.extend(predictions)
        return data, np.array(labels)


