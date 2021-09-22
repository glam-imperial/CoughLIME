import os
import numpy as np
import librosa
import lime_cough
from temporal_segmentation import TemporalSegmentation
from spectral_segmentation import SpectralSegmentation
from loudness_decomposition import LoudnessDecomposition
import soundfile
import random
import predict_dicova
import csv
import sys
import warnings


def get_explanation(audio, total_components, sr, num_samples=64, factorization_type='temporal'):
    if factorization_type == 'temporal':
        factorization = TemporalSegmentation(audio, total_components)
    elif factorization_type == 'spectral':
        factorization = SpectralSegmentation(audio, sr, total_components)
    elif factorization_type == 'loudness':
        factorization = LoudnessDecomposition(audio, sr)
    else:
        print("Error: factorization type not recognized")
        sys.exit()
    explainer = lime_cough.LimeCoughExplainer()
    explanation = explainer.explain_instance(segmentation=factorization,
                                             classifier_fn=predict_dicova.predict,
                                             labels=[0],
                                             num_samples=num_samples,
                                             batch_size=16,
                                             )
    return explanation, factorization


def evaluate_explanation(number_components, explanation, factorization, results_path, sample_rate, factorization_type):
    # return two lists, one for morf, one for rand
    w = [[x[0], x[1]] for x in explanation.local_exp[0]]
    components, weights = np.array(w, dtype=int)[:, 0], np.array(w)[:, 1]
    morf = []
    rand = []
    # remove most important n components and random components
    # predict on generated audios
    if factorization_type == 'loudness':
        # get num components that need to be removed for percentage levels and append to comp
        percentages = np.array([0, 0.1, 0.25, 0.5, 0.75, 0.9])
        num_comp = factorization.get_number_segments()
        comp = np.rint(percentages * num_comp).astype(int)
        comp[np.where(comp == 0)] = 1
    else:
        comp = range(number_components)
        num_comp = number_components
    for num_remove in comp:
        # morf: most recent first
        morf_indices = components[num_remove:]
        morf_mask = np.zeros(factorization.get_number_segments(),).astype(bool)
        morf_mask[morf_indices] = True
        morf_audio = factorization.get_segments_mask(morf_mask)
        path_morf = f"{results_path}curr_morf.wav"
        soundfile.write(path_morf, morf_audio, sample_rate)

        random_mask = np.zeros(factorization.get_number_segments(),).astype(bool)
        random_indices = random.sample(range(factorization.get_number_segments()), (num_comp - num_remove))
        random_mask[random_indices] = True
        random_audio = factorization.get_segments_mask(random_mask)
        path_rand = f"{results_path}curr_random.wav"
        soundfile.write(path_rand, random_audio, sample_rate)

        morf.append(predict_dicova.predict_single_audio(path_morf))
        rand.append(predict_dicova.predict_single_audio(path_rand))

    return morf, rand


def evaluate_data(comps, data_path):
    read_file = open(f'{data_path}/pixel_flipping.csv', 'r')
    csv_reader = csv.reader(read_file)
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
        path_save_summary = f"./eval/{removed}_removed_components.txt"
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


def main_pixel_flipping(factorization_type, results_path, data_directory, num_samples, number_components=None, list_files=None):
    # for file in data directory
    # generate explanation
    # save to csv file
    audio_directory = os.fsencode(data_directory)
    output = open(f'{results_path}/pixel_flipping.csv', 'w')
    writer = csv.writer(output)
    header = ['filename', 'c(entire file)', 'results type', 'c(removed_comp)']
    writer.writerow(header)

    if list_files:
        with open(list_files) as f:
            files_to_process = [line.rstrip() for line in f]
        for file in files_to_process:
            filename = f'{file}.flac'
            print("Starting with... ", filename)
            path_file = f'{data_directory}/{filename}'
            prediction_overall = predict_dicova.predict_single_audio(path_file)
            audio, sr = librosa.load(path_file)
            explanation, factorization = get_explanation(audio, number_components, sr, num_samples, factorization_type)
            morf, rand = evaluate_explanation(number_components, explanation, factorization, results_path, sr, factorization_type)
            writer.writerow([filename, prediction_overall, 'morf'] + morf)
            writer.writerow([filename, prediction_overall, 'rand'] + rand)
    else:
        for file in os.listdir(audio_directory):
            filename = os.fsdecode(file)
            print("Starting with... ", filename)
            path_file = f'{data_directory}/{filename}'
            prediction_overall = predict_dicova.predict_single_audio(path_file)
            audio, sr = librosa.load(path_file)
            explanation, factorization = get_explanation(audio, number_components, sr, num_samples, factorization_type)
            morf, rand = evaluate_explanation(number_components, explanation, factorization, results_path, sr, factorization_type)
            writer.writerow([filename, prediction_overall, 'morf'] + morf)
            writer.writerow([filename, prediction_overall, 'rand'] + rand)

    output.close()
    comp = [0, 0.1, 0.25, 0.5, 0.75, 0.9]
    evaluate_data(comp, results_path)


if __name__ == '__main__':
    sys.path.append('/Users/anne/Documents/Uni/Robotics/Masterarbeit/MA_Code/DICOVA/DiCOVA_baseline')
    warnings.filterwarnings("ignore", message="Trying to unpickle estimator LogisticRegression from version 0.24.1 when using version 0.23.2. This might lead to breaking code or invalid results. Use at your own risk.")

    # main_pixel_flipping(7, 'temporal', './eval/', '/Users/anne/Documents/Uni/Robotics/Masterarbeit/MA_Code/DICOVA/DiCOVA_Train_Val_Data_Release/AUDIO/', 128) # TODO: adapt path
    comp = [0, 0.1, 0.25, 0.5, 0.75, 0.9]
    evaluate_data(comp, './eval/')

