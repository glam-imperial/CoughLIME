"""
Functions for explaining classifiers that use Image data.
"""
from functools import partial

import numpy as np
import sklearn
from sklearn.utils import check_random_state
from tqdm.auto import tqdm


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

    def as_pyplot_figure(self, label=0):
        """Returns the explanation as a pyplot figure.

        Will throw an error if you don't have matplotlib installed
        Args:
            label: desired label. If you ask for a label for which an
                   explanation wasn't computed, will throw an exception.
                   Will be ignored for regression explanations.
            kwargs: keyword arguments, passed to domain_mapper

        Returns:
            pyplot figure (barchart).
        """
        import matplotlib.pyplot as plt
        exp = self.local_exp[label]
        fig = plt.figure()
        vals = [x[1] for x in exp]
        names = [x[0] for x in exp]
        vals.reverse()
        names.reverse()
        colors = ['green' if x > 0 else 'red' for x in vals]
        pos = np.arange(len(exp)) + .5
        plt.barh(pos, vals, align='center', color=colors)
        plt.yticks(pos, names)
        title = 'Local explanation for class COVID-positive'
        plt.title(title)
        return fig


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


