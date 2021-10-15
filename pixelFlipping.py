import os
import numpy as np
import librosa
import lime_cough
from temporal_decomposition import TemporalDecomposition
from spectral_decomposition import SpectralDecomposition
from loudness_decomposition import LoudnessDecomposition
from loudness_spectral_decomp import LoudnessSpectralDecomposition
from NMF_decomp import NMFDecomposition
import soundfile
import random
import csv
import sys
import warnings
from pathlib import Path


def get_explanation(audio, total_components, sr, predict_fn, num_samples=64, threshold=75,
                    decomposition_type='temporal'):
    """
    initializes decomposition and explanation objects and generates an explanation instance
    :param audio: np array((n,)) audio for which to generate the eplanation
    :param total_components: int, number of components to be generated (for temporal and spectral decompositions)
    :param sr: int, audio sample rate
    :param num_samples: int, number of neighborhood samples to train the linear classifier
    :param decomposition_type: "spectral", "temporal" or "loudness": decomposition type for the audio
    :return: explanation and decomposition objects
    """
    if decomposition_type == 'temporal':
        decomposition = TemporalDecomposition(audio, total_components)
    elif decomposition_type == 'spectral':
        decomposition = SpectralDecomposition(audio, sr, total_components)
    elif decomposition_type == 'loudness':
        decomposition = LoudnessDecomposition(audio, sr, threshold=threshold)
    elif decomposition_type == 'ls':
        decomposition = LoudnessSpectralDecomposition(audio, sr)
    elif decomposition_type == 'nmf':
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


def evaluate_explanation(comps, explanation, decomposition, predict_fn, results_path, sample_rate, decomposition_type):
    """
    evaluates the performance of an explanation by gradually removing most relevant and random components and predicting
    on the truncated audio file
    :param comps: list of ints, number of components  to remove and test the performance of the classifier
    :param explanation: explanation object to test
    :param decomposition: decomposition object of audio file
    :param results_path: where to store the results
    :param sample_rate: int, audio sample rate
    :param decomposition_type: type of decomposition
    :return: tuple (morf, rand) both are lists containing the predictions of the model on the modified audio files
    """
    # return two lists, one for morf, one for rand
    w = [[x[0], x[1]] for x in explanation.local_exp[0]]
    components, weights = np.array(w, dtype=int)[:, 0], np.array(w)[:, 1]
    morf = []
    rand = []
    # remove most important n components and random components
    # predict on generated audios
    if decomposition_type == 'loudness' or decomposition_type == 'ls':
        # get num components that need to be removed for percentage levels and append to comp
        percentages = np.array(comps)
        num_comp = decomposition.get_number_components()
        comp = np.rint(percentages * num_comp).astype(int)
        comp[np.where(comp == 0)] = 1
    else:
        comp = comps
        num_comp = decomposition.num_components
    for num_remove in comp:
        # morf: most recent first
        morf_indices = components[num_remove:]
        morf_mask = np.zeros(decomposition.get_number_components(),).astype(bool)
        morf_mask[morf_indices] = True
        morf_audio = decomposition.get_components_mask(morf_mask)

        random_mask = np.zeros(decomposition.get_number_components(),).astype(bool)
        random_indices = random.sample(range(decomposition.get_number_components()), (num_comp - num_remove))
        random_mask[random_indices] = True
        random_audio = decomposition.get_components_mask(random_mask)

        morf.append(predict_fn(morf_audio))
        rand.append(predict_fn(random_audio))

    return morf, rand


def evaluate_data(comps, data_path):
    """
    function that evaluates the overall performance of CoughLIME by comparing the percentages of most relevant first
    components removed leading to the same prediction as the entire file with random components leading to the same
    predictions
    :param comps: list of number of components that were removed
    :param data_path: path to csv file containing the results that need to be aggregated
    :return: nothing, saves summary files
    """
    read_file = open(f'{data_path}/pixel_flipping.csv', 'r')
    csv_reader = csv.reader(read_file, delimiter=';')  # TODO: changed
    _ = next(csv_reader)
    number_files = 0
    true_morf = [0] * len(comps)
    true_rand = [0] * len(comps)

    for row in csv_reader:
        if row[2] == 'morf':
            prediction_whole = np.rint(float(row[1]))
            for i in range(len(comps)):
                if np.rint(float(row[3 + i])) == prediction_whole:
                    true_morf[i] += 1
        else:
            prediction_whole = np.rint(float(row[1]))
            number_files += 1
            for i in range(len(comps)):
                if np.rint(float(row[3 + i])) == prediction_whole:
                    true_rand[i] += 1
    read_file.close()

    for index, removed in enumerate(comps):
        path_save_summary = f"{data_path}/{removed}_removed_components.txt"
        with open(path_save_summary, 'w') as summary:
            summary.write(f"Number samples: {number_files}")
            summary.write("\n")
            summary.write(f"Number true explanations: {true_morf}")
            summary.write("\n")
            summary.write(f"Number true random predictions: {true_rand}")
            summary.write("\n")
            summary.write(f"Percentage of true explanations: {float(true_morf[index]) / float(number_files)}")
            summary.write("\n")
            summary.write(f"Percentage of random true predictions: {float(true_rand[index]) / float(number_files)}")


def main_pixel_flipping(decomposition_type, results_path, data_directory, num_samples, components, threshold,
                        predict_fn, sr, list_files=None):
    """
    main function performing the pixel flipping
    :param decomposition_type: decomposition type to use
    :param results_path: where to store the results
    :param data_directory: path to the audio files
    :param num_samples: number of samples to use for the neighborhood to train the linear classifier
    :param components: list of number of components to remove during the evaluation
    :param list_files: if not None: path to a list containing file names to use for the evaluation, if not None:
                        evaluation will be conducted on all audio files in data_directory
    :return: nothing, writes results in summary files
    """
    # for file in data directory
    # generate explanation
    # save to csv file
    audio_directory = os.fsencode(data_directory)
    output = open(f'{results_path}/pixel_flipping.csv', 'w')
    writer = csv.writer(output)
    header = ['filename', 'c(entire file)', 'results type', 'c(removed_comp)']
    writer.writerow(header)
    output.close()
    output_comp = open(f'{results_path}/moc_{decomposition_type}.csv', 'w')
    writer_comp = csv.writer(output_comp)
    header_comp = ['filename', 'num_comp']
    if decomposition_type not in ['loudness', 'ls']:
        for i in components:
            header_comp.append(f'comp_{i}')
        num_to_return = components[-1] + 1
    else:
        for i in range(7):
            header_comp.append(f'comp_{i}')
        num_to_return = 7
    writer_comp.writerow(header_comp)
    output_comp.close()

    if list_files:
        with open(list_files) as f:
            files_to_process = [line.rstrip() for line in f]
        for file in files_to_process:
            filename = f'{file}.flac'
            output = open(f'{results_path}/pixel_flipping.csv', 'a')
            writer = csv.writer(output)
            output_comp = open(f'{results_path}/moc_{decomposition_type}.csv', 'a')
            writer_comp = csv.writer(output_comp)
            print("Starting with... ", filename)
            path_file = f'{data_directory}/{filename}'
            audio, _ = librosa.load(path_file, sr=sr)
            prediction_overall = predict_fn(audio)
            if decomposition_type == 'loudness':
                explanation, decomposition = get_explanation(audio, None, sr, predict_fn, num_samples,
                                                             threshold, decomposition_type)
            else:
                explanation, decomposition = get_explanation(audio, components[-1] + 1, sr, predict_fn,
                                                             num_samples, threshold, decomposition_type)

            morf, rand = evaluate_explanation(components, explanation, decomposition, predict_fn, results_path,
                                              sr, decomposition_type)
            writer.writerow([filename, prediction_overall, 'morf'] + morf)
            writer.writerow([filename, prediction_overall, 'rand'] + rand)
            output.close()
            _, indices = explanation.get_exp_components(0, True, True, num_to_return, return_indices=True)
            writer_comp.writerow([filename, decomposition.num_components] + list(indices))
            output_comp.close()
    else:
        for file in os.listdir(audio_directory):
            filename = os.fsdecode(file)
            print("Starting with... ", filename)
            path_file = f'{data_directory}/{filename}'
            audio, _ = librosa.load(path_file, sr=sr)
            prediction_overall = predict_fn(audio)
            explanation, decomposition = get_explanation(audio, components[-1] + 1, sr, predict_fn,
                                                         num_samples, decomposition_type)
            morf, rand = evaluate_explanation(components, explanation, decomposition, predict_fn, results_path,
                                              sr, decomposition_type)
            writer.writerow([filename, prediction_overall, 'morf'] + morf)
            writer.writerow([filename, prediction_overall, 'rand'] + rand)


    output.close()
    evaluate_data(components, results_path)


def significance(decomposition_type, results_path, data_directory, num_samples, components, threshold, predict_fn,
                 list_files=None, number_runs=5):
    """
    wrapper function that calls the main pixel flipping function various times to obtain statistically significant
    results
    :param decomposition_type: decomposition type to use
    :param results_path: where to store the results
    :param data_directory: path to the audio files
    :param num_samples: int, number of samples to use for the neighborhood to train the linear classifier
    :param components: list of number of components to remove during the evaluation
    :param list_files: if not None: path to a list containing file names to use for the evaluation, if not None:
                        evaluation will be conducted on all audio files in data_directory
    :param number_runs: int, number of runs to perform for significance analysis
    :return: nothing, stores results in text files
    """
    for run in range(number_runs):
        new_directory_name = f'{results_path}/output_run_{run}'
        Path(new_directory_name).mkdir(parents=True, exist_ok=True)
        results = f'{results_path}/output_run_{run}'
        main_pixel_flipping(decomposition_type, results, data_directory, num_samples, components, threshold,
                            predict_fn, list_files)


if __name__ == '__main__':
    sys.path.append('/Users/anne/Documents/Uni/Robotics/Masterarbeit/MA_Code/DICOVA/DiCOVA_baseline')
    warnings.filterwarnings("ignore", message="Trying to unpickle estimator LogisticRegression from version "
                                              "0.24.1 when using version 0.23.2. This might lead to breaking code "
                                              "or invalid results. Use at your own risk.")

    # main_pixel_flipping(7, 'temporal', './eval/', '/Users/anne/Documents/Uni/Robotics/
    # Masterarbeit/MA_Code/DICOVA/DiCOVA_Train_Val_Data_Release/AUDIO/', 128) # TODO: adapt path
    comp = [0, 0.1, 0.25, 0.5, 0.75, 0.9]
    evaluate_data(comp, './eval/')

