import os
import numpy as np
import librosa
import lime_cough
from temporal_decomposition import TemporalDecomposition
from spectral_decomposition import SpectralDecomposition
from loudness_decomposition import LoudnessDecomposition
from loudness_spectral_decomposition import LoudnessSpectralDecomposition
from NMF_decomposition import NMFDecomposition
import random
import sys
import csv


def get_explanation(audio, total_components, sr, predict_fn, num_samples=64, threshold=75,
                    decomp_type='temporal'):
    """
    initializes decomposition and explanation objects and generates an explanation instance
    :param audio: np array((n,)) audio for which to generate the eplanation
    :param total_components: int, number of components to be generated (for temporal and spectral decompositions)
    :param sr: int, audio sample rate
    :param predict_fn: function prediction of classifier on the entire audio sample
    :param threshold: int, power threshold for loudness and ls decompositions, can be None for remaining decompositions
    :param num_samples: int, number of neighborhood samples to train the linear classifier
    :param decomp_type: "spectral", "temporal", "loudness", "ls" or "nmf": decomposition type for the audio
        :return: explanation and decomposition objects
    """

    if decomp_type == 'temporal':
        decomposition = TemporalDecomposition(audio, total_components)
    elif decomp_type == 'spectral':
        decomposition = SpectralDecomposition(audio, sr, total_components)
    elif decomp_type == 'loudness':
        decomposition = LoudnessDecomposition(audio, sr, threshold=threshold)
    elif decomp_type == 'ls':
        decomposition = LoudnessSpectralDecomposition(audio, sr, threshold=threshold)
    elif decomp_type == 'nmf':
        decomposition = NMFDecomposition(audio, sr, num_components=total_components)
    else:
        print("Error: decomposition type not recognized")
        sys.exit()
    explainer = lime_cough.LimeCoughExplainer()
    explanation = explainer.explain_instance(decomposition=decomposition,
                                             classifier_fn=predict_fn,
                                             labels=[0],
                                             num_samples=num_samples,
                                             batch_size=16,
                                             )
    return explanation, decomposition


def evaluate_explanation(comps, explanation, decomposition, predict_fn, decomposition_type,
                         prediction_overall):
    """evaluates the performance of an explanation by gradually adding most relevant and random components and predicting
    on the truncated audio file
    :param comps: list of ints, number of components to include and test the performance of the classifier
    :param explanation: explanation object to test
    :param decomposition: decomposition object of audio file
    :param predict_fn: function, predict function of the black-box classifier
    :param decomposition_type: type of decomposition
    :param prediction_overall: prediction of classifier on the entire audio sample
    :return: tuple (morf, rand) both are lists containing the predictions of the model on the modified audio files
    """
    morf = []
    rand = []
    if decomposition_type == 'loudness' or decomposition_type == 'ls':
        percentages = np.array(comps)
        num_comp = decomposition.get_number_components()
        comp = np.rint(percentages * num_comp).astype(int)
        comp[np.where(comp == 0)] = 1
        if comps[-1] == 1 or comp[-1] == 1.0:
            comp[-1] = num_comp
    else:
        comp = comps
        num_comp = decomposition.num_components
    for num_c in comp:
        if num_c == num_comp:
            morf.append(prediction_overall)
            rand.append(prediction_overall)
        else:
            audio_morf, _ = explanation.get_exp_components(0, positive_components=True,
                                                                  negative_components=True,
                                                                  num_components=num_c)
            morf.append(predict_fn(audio_morf))
            random_mask = np.zeros(decomposition.get_number_components(),).astype(bool)
            random_indices = random.sample(range(decomposition.get_number_components()), num_c)
            random_mask[random_indices] = True
            random_audio = decomposition.get_components_mask(random_mask)
            rand.append(predict_fn(random_audio))

    return morf, rand


def evaluate_data(components, data_path):
    """
    function to generate and the summary files of the conducted evaluation
    :param components: list of ints, contains the different numbers of components that were tested
    :param data_path: path to csv file containing the results
    """
    read_file = open(f'{data_path}/quant_eval.csv', 'r')
    csv_reader = csv.reader(read_file, delimiter=';')  # TODO: changed
    _ = next(csv_reader)
    number_files = 0
    true_morf = [0] * len(components)
    true_rand = [0] * len(components)

    for row in csv_reader:
        if row[2] == 'morf':
            prediction_whole = np.rint(float(row[1]))
            for i in range(len(components)):
                if np.rint(float(row[3 + i])) == prediction_whole:
                    true_morf[i] += 1
        else:
            prediction_whole = np.rint(float(row[1]))
            number_files += 1
            for i in range(len(components)):
                if np.rint(float(row[3 + i])) == prediction_whole:
                    true_rand[i] += 1
    read_file.close()

    for index, c in enumerate(components):
        # save these to txt file for further processing
        percentage_true_exp = float(true_morf[index]) / float(number_files)
        percentage_rand = float(true_rand[index]) / float(number_files)
        path_save_summary = f"./{data_path}/{c}_components.txt"
        with open(path_save_summary, 'w') as summary:
            summary.write(f"Number samples: {number_files}")
            summary.write("\n")
            summary.write(f"Number true explanations: {true_morf}")
            summary.write("\n")
            summary.write(f"Number true random predictions: {true_rand}")
            summary.write("\n")
            summary.write(f"Percentage of true explanations: {percentage_true_exp}")
            summary.write("\n")
            summary.write(f"Percentage of random true predictions: {percentage_rand}")


def main_quantitative_analysis(decomposition_type, results_path, data_directory, num_samples, components, total_comp,
                               threshold, predict_fn, sr, list_files):
    """
    main function performing the evaluation
    :param decomposition_type: decomposition type to use
    :param results_path: where to store the results
    :param data_directory: path to the audio files
    :param num_samples: number of samples to use for the neighborhood to train the linear classifier
    :param components: list of number of components to include during the evaluation
    :param total_comp: int, number of total components for decomposition, can be None for Loudness and ls
    :param threshold: int, power threshold for loudness and ls decompositions, can be None for remaining decompositions
    :param predict_fn: function, prediction of classifier on the entire audio sample
    :param sr: int, sample rate of audio
    :param list_files: path to a list containing file names to use for the evaluation, if not None:
                        evaluation will be conducted on all audio files in data_directory
    :return: nothing, writes results in summary files
    """
    output = open(f'{results_path}/quant_eval.csv', 'w')
    writer = csv.writer(output)
    header = ['filename', 'c(entire file)', 'results type', 'c(comp)']
    writer.writerow(header)
    output.close()

    with open(list_files) as f:
        files_to_process = [line.rstrip() for line in f]
    for file in files_to_process:
        filename = f'{file}.flac'
        output = open(f'{results_path}/quant_eval.csv', 'a')
        writer = csv.writer(output)
        print("Starting with... ", filename)
        path_file = f'{data_directory}/{filename}'
        audio, _ = librosa.load(path_file, sr=sr)
        prediction_overall = predict_fn(audio)
        if decomposition_type == 'loudness' or decomposition_type == 'ls':
            explanation, decomposition = get_explanation(audio, None, sr, predict_fn, num_samples,
                                                         threshold, decomposition_type)
        else:
            explanation, decomposition = get_explanation(audio, total_comp, sr, predict_fn,
                                                         num_samples, threshold, decomposition_type)

        morf, rand = evaluate_explanation(components, explanation, decomposition, predict_fn, decomposition_type,
                                          prediction_overall)
        writer.writerow([filename, prediction_overall, 'morf'] + morf)
        writer.writerow([filename, prediction_overall, 'rand'] + rand)
        output.close()

    output.close()
    evaluate_data(components, results_path)

