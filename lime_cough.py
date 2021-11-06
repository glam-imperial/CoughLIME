from functools import partial
import numpy as np
import sklearn
from sklearn.utils import check_random_state
from tqdm.auto import tqdm
import librosa
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
import matplotlib.patches as mpatches
from lime import lime_base
import sys


class CoughExplanation(object):
    def __init__(self, decomposition):
        """
        init function for Cough Explanation object
        :param decomposition: object, chosen decomposition for the explanation, possibilities:
                                temporal, spectral, loudness, ls (loudness-spectral), nmf
        """

        self.decomposition = decomposition
        self.intercept = {}
        self.local_exp = {}
        self.local_pred = {}
        self.score = {}

    def get_exp_components(self, label, positive_components=True, negative_components=True, num_components='all',
                           min_abs_weight=0.0, return_indices=False):
        """
        function that returns the audio made of the num_components most important components
        :param label: class for which to explain the prediction
        :param positive_components: bool, whether to include components with positive weights
        :param negative_components: bool, whether to include components with negative weights
        :param num_components: int, how many components to return
        :param min_abs_weight: float, min abs weight that the components needs to have in order to be included in return
        :param return_indices: bool, whether to also return the indices
        :return: audio that is made of the most important num_components for the explanation, possibly also indices
        """
        used_features, _ = self.get_used_indices(label, positive_components, negative_components,
                                                 num_components, min_abs_weight)

        audio = self.decomposition.return_components(used_features)
        if return_indices:
            return audio, used_features
        return audio

    def get_used_indices(self, label, positive_components=True, negative_components=False, num_components='all',
                         min_abs_weight=0.0):
        """
        returns the indices of the num_components most important components of the explanation and their weights in the
        explanation
        :param label: class for which to explain the prediction
        :param positive_components: bool, whether to include components with positive weights
        :param negative_components: bool, whether to include components with negative weights
        :param num_components: int, how many components to return
        :param min_abs_weight: float, min abs weight that the components needs to have in order to be included in return
        :param return_indices: bool, whether to also return the indices
        :return: audio that is made of the most important num_components for the explanation, possibly also indices
        """
        if label not in self.local_exp:
            print('Error: Label not in explanation')
            sys.exit()
        if positive_components is False and negative_components is False:
            print('Error: positive_components, negative_components or both must be True')
            sys.exit()

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

        used_features = used_features[:num_components]
        return used_features, weights

    def weighted_audio(self, label, positive_components=True, negative_components=False, num_components='all',
                            min_abs_weight=0.0, return_indices=False):
        """
        return weighted audio made of num_components most important components, weighted by importance of components
        :param label: class for which to explain the prediction
        :param positive_components: bool, whether to include components with positive weights
        :param negative_components: bool, whether to include components with negative weights
        :param num_components: int, how many components to return
        :param min_abs_weight: float, min abs weight that the components needs to have in order to be included in return
        :param return_indices: bool, whether to also return the indices
        :return: array, weighted audio of the most important num_components for the explanation,
             possibly also indices as list
        """
        # returns weighted audio (weighted by abs value of components)
        used_features, weights = self.get_used_indices(label, positive_components, negative_components,
                                                       num_components, min_abs_weight)
        used_features = used_features[:num_components]
        weights = weights[:num_components]
        audio = self.decomposition.return_weighted_components(used_features, weights)
        if return_indices:
            return audio, used_features
        return audio

    def normalize(self, weights):
        """
        normalizes an array of weights to be in a certain range to obtain better transparency values for the images
        :param weights: array of weights to be normalized
        :return: array of normalized weights
        """
        abs_weights = np.abs(np.array(weights))
        minimum = min(abs_weights) - 0.2 * max(abs_weights)
        maximum = max(abs_weights) + 0.4 * max(abs_weights)
        normalized = np.zeros(np.shape(abs_weights))
        for i, _ in enumerate(abs_weights):
            normalized[i] = (abs_weights[i] - minimum) / (maximum - minimum)  # zi = (xi – min(x)) / (max(x) – min(x))
        return normalized

    def show_image_mask_spectrogram(self, label, positive_only=True, negative_only=False, hide_rest=True,
                                    num_features=5, min_weight=0., save_path=None,
                                    show_colors=False, show_loudness=True):
        """
        generates an image of the decomposition with the most important components highlighted in green and red
        :param label: class for which to explain the prediction
        :param positive_only: bool, whether to include only components with positive weights
        :param negative_only: bool, whether to include only components with negative weights
        :param hide_rest: bool, whether to hide the other components that are not among the most important
        :param num_features: int, how many components to return
        :param min_weight: float, min abs weight that the components needs to have in order to be included in return
        :param save_path: if not None: path to save the generated image
        :param show_colors: bool, whether to show the components in red and green or just highlight them without colors
        :param show_loudness: bool, for loudness decomposition, whether to include an image of the power array
        :return: nothing, shows and possibly saves image
        """
        if self.decomposition.decomposition_type == 'spectral':
            self.image_spectral(label, positive_only=positive_only, negative_only=negative_only,
                                hide_rest=hide_rest, num_features=num_features, min_weight=min_weight,
                                save_path=save_path, show_colors=show_colors)

        elif self.decomposition.decomposition_type == 'loudness':
            self.image_loudness(label, positive_only=positive_only, negative_only=negative_only,
                                hide_rest=hide_rest, num_features=num_features, min_weight=min_weight,
                                save_path=save_path, show_colors=show_colors, show_loudness=show_loudness)
        
        elif self.decomposition.decomposition_type == 'temporal': 
            self.image_temporal(label, positive_only=positive_only, negative_only=negative_only, 
                                hide_rest=hide_rest, num_features=num_features, min_weight=min_weight, 
                                save_path=save_path, show_colors=show_colors)

        elif self.decomposition.decomposition_type == 'nmf':
            self.image_nmf(label, positive_only=positive_only, negative_only=negative_only,
                           hide_rest=hide_rest, num_features=num_features, min_weight=min_weight,
                           save_path=save_path, show_colors=show_colors)

        elif self.decomposition.decomposition_type == 'ls':
            self.image_ls(label, positive_only=positive_only, negative_only=negative_only,
                          hide_rest=hide_rest, num_features=num_features, min_weight=min_weight,
                          save_path=save_path, show_colors=show_colors)

    def get_indices(self, label, positive_only=True, negative_only=False, hide_rest=True,
                    num_features=5, min_weight=0., get_mask=True):
        """ helper function to return image with highlighted most important components returning the corresponding
        component indices
        :param label: class for which to explain the prediction
        :param positive_only: bool, whether to include only components with positive weights
        :param negative_only: bool, whether to include only components with negative weights
        :param hide_rest: bool, whether to hide unused features in image
        :param num_features: int, how many components to return
        :param get_mask: bool, whether to also return the generated mask  for the image (for scikit mark_boundaries)
        :param min_weight: float, min abs weight that the components needs to have in order to be included in return
        :return: indices of components to include, indices of important components (array(k,)), corresponding weights
                        (array(k,)), mask to use to highlight components
        """
        decomposition = self.decomposition
        explanation = self.local_exp[label]

        if positive_only:
            indices_comp = [x[0] for x in explanation if x[1] > 0 and x[1] > min_weight][:num_features]
            weights = [x[1] for x in explanation if x[1] > 0 and x[1] > min_weight][:num_features]
            mask = decomposition.return_mask_boundaries(indices_comp, [])
        if negative_only:
            indices_comp = [x[0] for x in explanation if x[1] < 0 and abs(x[1]) > min_weight][:num_features]
            weights = [x[1] for x in explanation if x[1] < 0 and abs(x[1]) > min_weight][:num_features]
            if get_mask:
                mask = decomposition.return_mask_boundaries([], indices_comp)
        if positive_only or negative_only:
            if hide_rest:
                indices_show = indices_comp
            else:
                indices_show = range(decomposition.get_number_components())
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
            if get_mask:
                mask = decomposition.return_mask_boundaries(comp_pos, comp_neg)
            if hide_rest:
                indices_show = comp_pos + comp_neg
            else:
                indices_show = range(decomposition.get_number_components())
        if get_mask:
            return indices_show, indices_comp, mask, weights
        else:
            return indices_show, indices_comp, weights

    def make_masked_image(self, mask):
        """
        helper function to show image with most important components highlighted
        :param mask: 2d array, where -1: show in red, where 1, show in green
        :return: image with red and green components
        """
        image = np.ones(np.shape(mask) + (4,))
        mask_negative = np.zeros(np.shape(mask))
        mask_negative[np.where(mask == 0)] = 1
        mask_negative_green = np.ones(np.shape(mask))
        mask_negative_green[np.where(mask == -1)] = 0
        mask_negative_red = np.ones(np.shape(mask))
        mask_negative_red[np.where(mask == 1)] = 0
        image[:, :, 0] = mask_negative_red  # 0 for red, 1 for green
        image[:, :, 1] = mask_negative_green
        image[:, :, 2] = mask_negative
        image[:, :, 3] = np.abs(mask)
        return image

    def image_spectral(self, label, positive_only=True, negative_only=False, hide_rest=True, num_features=5,
                       min_weight=0., save_path=None, show_colors=False):
        """
        generates the image highlighting the num_components most important components for the spectral decomposition
        :param label: class for which to explain the prediction
        :param positive_only: bool, whether to include components with positive weights
        :param negative_only: bool, whether to include components with negative weights
        :param hide_rest: bool, whether to hide or show less important components
        :param num_features: int, how many components to highlight
        :param min_weight: float, min abs weight that the components needs to have in order to be highlighted
        :param save_path: if not None: path where to save the generated image
        :param show_colors: bool, whether to highlight the components in green or red or just mark them
        :return: nothing, shows and saves image if save_path is specified
        """

        spectrogram_indices, indices_comp, mask, weights = self.get_indices(label, positive_only=positive_only,
                                                                            negative_only=negative_only,
                                                                            hide_rest=hide_rest,
                                                                            num_features=num_features,
                                                                            min_weight=min_weight)

        spectrogram = self.decomposition.return_spectrogram_indices(spectrogram_indices)
        spec_db = librosa.power_to_db(spectrogram, ref=np.max)
        marked = mark_boundaries(spec_db, mask)
        plt.imshow(marked[:, :, 2], origin="lower", cmap=plt.get_cmap("magma"))
        plt.colorbar(format='%+2.0f dB')
        if show_colors:
            normalized_weights = self.normalize(weights)
            for index, comp in enumerate(indices_comp):
                if weights[index] < 0:
                    mask = self.decomposition.return_mask_boundaries([], [comp])
                else:
                    mask = self.decomposition.return_mask_boundaries([comp], [])
                image_array = self.make_masked_image(mask)
                plt.imshow(image_array, origin="lower", interpolation="nearest", alpha=normalized_weights[index])

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

    def image_loudness(self, label, positive_only=True, negative_only=False, hide_rest=True, num_features=5,
                       min_weight=0., save_path=None, show_colors=False, show_loudness=True):
        """
        generates the image highlighting the num_components most important components for the loudness decomposition
        :param label: class for which to explain the prediction
        :param positive_only: bool, whether to include components with positive weights
        :param negative_only: bool, whether to include components with negative weights
        :param hide_rest: bool, whether to hide or show less important components
        :param num_features: int, how many components to highlight
        :param min_weight: float, min abs weight that the components needs to have in order to be highlighted
        :param save_path: if not None: path where to save the generated image
        :param show_colors: bool, whether to highlight the components in green or red or just mark them
        :param show_loudness: bool, whether to also show the components in the dB power curve
        :return: nothing, shows and saves image if save_path is specified
        """
        image_indices, indices_comp, mask, weights = self.get_indices(label, positive_only=positive_only,
                                                                      negative_only=negative_only,
                                                                      hide_rest=hide_rest,
                                                                      num_features=num_features,
                                                                      min_weight=min_weight)
        # return the loudness waveform and decibels array for the corresponding image_indices
        waveform, loudness = self.decomposition.return_components(image_indices, loudness=True)
        if hide_rest:
            waveform[np.where(waveform == 0)] = np.nan
            loudness[np.where(loudness == 0)] = np.nan
        if show_loudness:
            fig, (ax1, ax2) = plt.subplots(2)
        else:
            fig = plt.figure(figsize=(7, 4))
            ax1 = fig.add_subplot(111)
        fig.suptitle('Loudness Decomposition')
        ax1.plot(waveform, color='c')
        component_indices = [0] + self.decomposition.indices_components + [np.size(waveform)]
        # only mark the important components!!
        for i in indices_comp:
            left = component_indices[i]
            bottom = -0.98
            width = component_indices[i + 1] - component_indices[i]
            height = 1.96
            rect = mpatches.Rectangle((left, bottom), width, height,
                                      fill=False,
                                      color="purple",
                                      linewidth=2)
            ax1.add_patch(rect)
        ax1.set(xlabel='Time', ylabel='Amplitude', xlim=[0, np.size(waveform)], ylim=[-1, 1])
        if show_loudness:
            ax2.plot(loudness, color='c')
            for i in indices_comp:
                left = component_indices[i]
                bottom = 1
                width = component_indices[i + 1] - component_indices[i]
                height = 148
                rect = mpatches.Rectangle((left, bottom), width, height,
                                          fill=False,
                                          color="purple",
                                          linewidth=2)
                ax2.add_patch(rect)
            ax2.set(xlabel='Time', ylabel='Power (db)', xlim=[0, np.size(waveform)], ylim=[0, 150])
        if show_colors:
            normalized_weights = self.normalize(weights)
            for index, comp in enumerate(indices_comp):
                if weights[index] < 0:
                    ax1.axvspan(component_indices[comp], component_indices[comp+1], facecolor='red',
                                alpha=normalized_weights[index])
                    if show_loudness:
                        ax2.axvspan(component_indices[comp], component_indices[comp+1], facecolor='red',
                                    alpha=normalized_weights[index])
                else:
                    ax1.axvspan(component_indices[comp], component_indices[comp+1], facecolor='green',
                                alpha=normalized_weights[index])
                    if show_loudness:
                        ax2.axvspan(component_indices[comp], component_indices[comp+1], facecolor='green',
                                    alpha=normalized_weights[index])
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()
        plt.close()
        
    def image_temporal(self, label, positive_only=True, negative_only=False, hide_rest=False, num_features=3, 
                       min_weight=0.0, save_path=None, show_colors=True):
        """
        generates the image highlighting the num_components most important components for the temporal decomposition
        :param label: class for which to explain the prediction
        :param positive_only: bool, whether to include components with positive weights
        :param negative_only: bool, whether to include components with negative weights
        :param hide_rest: bool, whether to hide or show less important components
        :param num_features: int, how many components to highlight
        :param min_weight: float, min abs weight that the components needs to have in order to be highlighted
        :param save_path:  if not None: path where to save the generated image
        :param show_colors: bool, whether to highlight the components in green or red or just mark them
        """
        image_indices, indices_comp, mask, weights = self.get_indices(label, positive_only=positive_only,
                                                                      negative_only=negative_only,
                                                                      hide_rest=hide_rest,
                                                                      num_features=num_features,
                                                                      min_weight=min_weight)
        waveform = self.decomposition.return_components(image_indices)
        if hide_rest:
            waveform[np.where(waveform == 0)] = np.nan
        length_audio = np.shape(waveform)[0]
        distance = int(length_audio/self.decomposition.num_components)
        indices = np.array(range(self.decomposition.num_components))
        indices = indices * distance
        indices = np.append(indices, [length_audio])
        fig = plt.figure(figsize=(7, 3))
        ax1 = fig.add_subplot(111)
        fig.suptitle('Temporal Decomposition')
        ax1.plot(waveform, color='c')
        # only mark the important components!!
        for i in indices_comp:
            left = indices[i]
            bottom = -0.98
            width = indices[i + 1] - indices[i]
            height = 1.96
            rect = mpatches.Rectangle((left, bottom), width, height,
                                      fill=False,
                                      color="purple",
                                      linewidth=2)
            ax1.add_patch(rect)
        ax1.set(xlabel='Time', ylabel='Amplitude', xlim=[0, np.size(waveform)], ylim=[-1, 1])

        if show_colors:
            normalized_weights = self.normalize(weights)
            for index, comp in enumerate(indices_comp):
                if weights[index] < 0:
                    ax1.axvspan(indices[comp], indices[comp+1], facecolor='red',
                                alpha=normalized_weights[index])
                else:
                    ax1.axvspan(indices[comp], indices[comp+1], facecolor='green',
                                alpha=normalized_weights[index])
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()
        plt.close()

    def image_nmf(self, label, positive_only=False, negative_only=False,
                  hide_rest=False, num_features=3, min_weight=0.0,
                  save_path=None, show_colors=True):
        """
        generates the image highlighting the num_components most important components for the nmf decomposition
        :param label: class for which to explain the prediction
        :param positive_only: bool, whether to include components with positive weights
        :param negative_only: bool, whether to include components with negative weights
        :param hide_rest:  bool, whether to hide or show less important components
        :param num_features: int, how many components to highlight
        :param min_weight: float, min abs weight that the components needs to have in order to be highlighted
        :param save_path: if not None: path where to save the generated image
        :param show_colors: bool, whether to highlight the components in green or red or just mark them
        """
        indices_show, indices_comp, weights = self.get_indices(label, positive_only, negative_only, hide_rest,
                                                               num_features, min_weight, get_mask=False)
        num_c = self.decomposition.num_components
        w = self.decomposition.W
        h = self.decomposition.H

        fig, ax = plt.subplots(1, num_c, figsize=(7, 8))
        fig.suptitle("NMF Decomposition into 6 Components\nSpectral Profiles")
        logw = np.log10(w)
        normalized_weights = self.normalize(weights)

        for i in range(num_c):
            if i in indices_show:
                x = list(range(len(-logw[:, i])))
                ax[i].plot(logw[:, i], x)
                ax[i].set_xlabel(f"Component {i+1}", rotation=90)
                if i in indices_comp:
                    w_i = indices_comp.index(i)
                    if weights[w_i] > 0:
                        ax[i].set_facecolor((0.0, 1.0, 0.0, normalized_weights[w_i]))
                    else:
                        ax[i].set_facecolor((1.0, 0.0, 0.0, normalized_weights[w_i]))
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(f'{save_path}/nmf_spectral.png')
        plt.show()
        plt.close()
        # temporal activations

        fig, ax = plt.subplots(num_c, 1, figsize=(7, 7))
        fig.suptitle("NMF Decomposition into 6 Components\nTemporal Activations")
        for i in range(num_c):
            if i in indices_show:
                ax[i].plot(h[i])
                ax[i].set_ylabel(f"Component {i+1}", rotation=90)
                if i in indices_comp:
                    w_i = indices_comp.index(i)
                    if weights[w_i] > 0:
                        ax[i].set_facecolor((0.0, 1.0, 0.0, normalized_weights[w_i]))
                    else:
                        ax[i].set_facecolor((1.0, 0.0, 0.0, normalized_weights[w_i]))
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(f"{save_path}/nmf_temporal.png")
        plt.show()
        print("visualized :) ")
        plt.close()

    def image_ls(self, label, positive_only=False, negative_only=False,
                 hide_rest=False, num_features=3, min_weight=0.0,
                 save_path=None, show_colors=True):
        """
        generates the image highlighting the num_components most important components for the loudness-spectral
         decomposition
        :param label: class for which to explain the prediction
        :param positive_only: bool, whether to include components with positive weights
        :param negative_only: bool, whether to include components with negative weights
        :param hide_rest:  bool, whether to hide or show the other components
        :param num_features: int, how many components to show
        :param min_weight: float, min abs weight that the components needs to have in order to be highlighted
        :param save_path: if not None: path where to save the generated image
        :param show_colors:bool, whether to highlight the components in green or red or just mark them
        """
        image_indices, indices_comp, mask, weights = self.get_indices(label, positive_only=positive_only,
                                                                      negative_only=negative_only,
                                                                      hide_rest=hide_rest,
                                                                      num_features=num_features,
                                                                      min_weight=min_weight)

        fig, (ax1) = plt.subplots(1)

        mask_s = np.zeros(self.decomposition.num_components).astype(bool)
        mask_s[image_indices] = True
        spectrogram = self.decomposition.get_components_mask(mask_s, spec=True)

        spec_db = librosa.power_to_db(spectrogram, ref=np.max)

        marked = mark_boundaries(spec_db, mask)
        img = ax1.imshow(marked[:, :, 2], origin="lower", cmap=plt.get_cmap("magma"))
        fig.colorbar(img, ax=ax1)
        normalized_weights = self.normalize(weights)
        for index, comp in enumerate(indices_comp):
            if weights[index] < 0:
                mask = self.decomposition.return_mask_boundaries([], [comp])
            else:
                mask = self.decomposition.return_mask_boundaries([comp], [])
            image_array = self.make_masked_image(mask)
            ax1.imshow(image_array, origin="lower", interpolation="nearest", alpha=normalized_weights[index])

        ax1.set_xlabel("Time")
        ax1.set_ylabel("Frequency")
        ax1.axes.xaxis.set_ticks([])
        ax1.axes.yaxis.set_ticks([])
        fig.suptitle('Spectral-Loudness Decomposition')

        if save_path is not None:
            plt.savefig(save_path)
        plt.show()
        plt.close()


class LimeCoughExplainer(object):
    """Explains predictions on Cough audio (1D array) data """

    def __init__(self, kernel_width=.25, kernel=None, verbose=False,
                 feature_selection='auto', random_state=None):
        """
        Init function
        :param kernel_width: kernel width for the exponential kernel.
        :param kernel: similarity kernel that takes euclidean distances and kernel
                width as input and outputs weights in (0,1). If None, defaults to
                an exponential kernel.
        :param verbose: if true, print local prediction values from linear model
                feature_selection:
        :param feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
                See function 'explain_instance_with_data' in lime_base.py for
                details on what each of the options does.
        :param random_state: an integer or numpy.RandomState that will be used to
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

    def explain_instance(self, decomposition, classifier_fn, labels=(1,), num_samples=1000,
                         batch_size=10, distance_metric='cosine', model_regressor=None,
                         random_seed=None, progress_bar=False):
        """
        Generates explanations for a prediction
        :param decomposition: decomposition object for the chosen audio decomposition
        :param classifier_fn: classifier prediction probability function, which
                takes a numpy array and outputs prediction probabilities.  For
                ScikitClassifiers , this is classifier.predict_proba.
        :param labels: iterable with labels to be explained.
        :param num_samples: size of the neighborhood to learn the linear model for the explanation
        :param batch_size: number of samples processed in one batch in predict function
        :param distance_metric: the distance metric to use for weights.
        :param model_regressor: sklearn regressor to use in explanation. Defaults
            to Ridge regression in LimeBase. Must have model_regressor.coef_
            and 'sample_weight' as a parameter to model_regressor.fit()
            segmentation_fn: SegmentationAlgorithm, wrapped skimage
            decomposition function
        :param random_seed: integer used as random seed for the decomposition
                algorithm. If None, a random integer, between 0 and 1000,
                will be generated using the internal random number generator.
        :param progress_bar: if True, show tqdm progress bar.
        :return: CoughExplanation object
        """

        if random_seed is None:
            random_seed = self.random_state.randint(0, high=1000)

        top = labels

        num_features = decomposition.get_number_components()

        data, labels = self.data_labels(decomposition,
                                        classifier_fn, num_samples,
                                        batch_size=batch_size,
                                        progress_bar=progress_bar)

        distances = sklearn.metrics.pairwise_distances(
            data,
            data[0].reshape(1, -1),
            metric=distance_metric
        ).ravel()

        ret_exp = CoughExplanation(decomposition)

        for label in top:  # same in image and audio
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score[label],
             ret_exp.local_pred[label]) = self.base.explain_instance_with_data(
                data, labels, distances, label, num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection)
        return ret_exp

    def data_labels(self, decomposition, classifier_fn, num_samples, batch_size=10, progress_bar=True):
        """
        Generates neighborhood data and predictions for it
        :param decomposition: decomposition object, decomposition of cough audio array
        :param classifier_fn: function that takes a list of images and returns a
                matrix of prediction probabilities
        :param num_samples: size of the neighborhood to learn the linear model
        :param batch_size: classifier_fn will be called on batches of this size.
        :param progress_bar: if True, show tqdm progress bar.
        :return: data, labels:
                data: neighborhood data to train the classifier
                labels: 2d array of prediction probabilities of all labels to be explained for all data points generated
        """

        n_features = decomposition.get_number_components()

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
            temp = decomposition.get_components_mask(mask)
            audios.append(temp)
            if len(audios) == batch_size:
                predictions = classifier_fn(np.array(audios))
                labels.extend(predictions)
                audios = []
        if len(audios) > 0:
            predictions = classifier_fn(np.array(audios))
            labels.extend(predictions)
        return data, np.array(labels)


