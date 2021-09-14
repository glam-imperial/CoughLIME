import os
import numpy as np
import librosa
from pathlib import Path
import pandas as pd
import lime_cough
from temporal_segmentation import TemporalSegmentation
from spectral_segmentation import SpectralSegmentation
from loudness_decomposition import LoudnessDecomposition
import soundfile
import random
import predict_dicova
import sys


def get_explanation(audio, total_components=None, sr=None, decomp_type='temporal', num_samples=64):
    if decomp_type == 'temporal':
        factorization = TemporalSegmentation(audio, total_components)
    elif decomp_type == 'spectral':
        factorization = SpectralSegmentation(audio, sr, total_components)
    elif decomp_type == 'loudness':
        factorization = LoudnessDecomposition(audio, sr)
    else:
        print("Error: decomposition type not recognized")
        sys.exit()
    explainer = lime_cough.LimeCoughExplainer()
    explanation = explainer.explain_instance(segmentation=factorization,
                                             classifier_fn=predict_dicova.predict,
                                             labels=[0],
                                             num_samples=num_samples,
                                             batch_size=16,
                                             )
    return explanation, factorization


def save_mix(explanation, num_components, filename, factorization, sample_rate, gen_random=False):
    label = list(explanation.local_exp.keys())[0]
    audio, component_indeces = explanation.get_exp_components(label, positive_components=True,
                                                              negative_components=True,
                                                              num_components=num_components,
                                                              return_indeces=True)
    # num components: how many components should model take for explanations
    path_name_write = f"./quantitative_evaluation/{num_components}_components/explanations/{filename[:-5]}_e.wav"
    soundfile.write(path_name_write, audio, sample_rate)
    print("Indices", component_indeces)
    if gen_random:
        # random components must also be generated
        random_mask = np.zeros(factorization.get_number_segments(),).astype(bool)
        random_indices = random.sample(range(factorization.get_number_segments()), num_components)
        random_mask[random_indices] = True
        random_audio = factorization.get_segments_mask(random_mask)
        path_name_write = f"./quantitative_evaluation/{num_components}_components/random_components/{filename[:-5]}_r.wav"
        soundfile.write(path_name_write, random_audio, sample_rate)


def save_predictions_explanations(components, audio_directory_str, total_components=None, decomp_type='temporal', num_samples=64):
    # for each audio file
    # make directory for the number of components if not exists
    file_names = []
    predictions_entire_file = []
    comp_exp = []
    comp_random = []

    for _ in components:
        comp_exp.append([])
        comp_random.append([])

    # TODO: adapt path
    audio_directory = os.fsencode(audio_directory_str)

    for file in os.listdir(audio_directory):
        filename = os.fsdecode(file)
        print("Starting with... ", filename)
        path_file = f'{audio_directory_str}/{filename}'
        file_names.append(filename)

        # get prediction for whole audio file
        prediction_overall = predict_dicova.predict_single_audio(path_file)
        predictions_entire_file.append(prediction_overall)

        # get explanation
        audio, fs = librosa.load(path_file)
        explanation, factorization = get_explanation(audio, total_components, sr=fs, decomp_type=decomp_type, num_samples=num_samples)

        # get mixes for top_components and save them, adapt this for loudness decomposition
        if decomp_type != 'loudness':
            for index, num_components in enumerate(components):
                morf, rand = get_predictions(explanation, factorization, num_components, fs)
                comp_exp[index].append(morf)
                comp_random[index].append(rand)
        elif decomp_type == 'loudness':
            for index, level in enumerate(components):  # components = levels
                total = factorization.get_number_segments()
                to_retrieve = int(np.rint(float(level) * float(total)))
                if to_retrieve == 0:
                    to_retrieve = 1
                morf, rand = get_predictions(explanation, factorization, to_retrieve, fs)
                comp_exp[index].append(morf)
                comp_random[index].append(rand)
        else:
            print("Error: decomposition type not recognized, aborting")
            sys.exit()

    # create csv "summary" with pandas for every component
    for index, num_components in enumerate(components):
        summary_dict = {'File_name': file_names, 'Prediction_whole_file': predictions_entire_file, 'Prediction_top_comp': comp_exp[index], 'Prediction_random_comp': comp_random[index]}
        summary_df = pd.DataFrame.from_dict(summary_dict)
        path_csv = f"./quantitative_evaluation/{num_components}_summary.csv"
        summary_df.to_csv(path_csv)


def get_predictions(explanation, factorization, num_components, fs):
    label = list(explanation.local_exp.keys())[0]
    audio, component_indeces = explanation.get_exp_components(label, positive_components=True,
                                                              negative_components=True,
                                                              num_components=num_components,
                                                              return_indeces=True)
    # num components: how many components should model take for explanations
    path_name_write = f"./quantitative_evaluation/current_exp.wav"
    soundfile.write(path_name_write, audio, fs)
    morf = predict_dicova.predict_single_audio(path_name_write)
    # generate random

    random_mask = np.zeros(factorization.get_number_segments(),).astype(bool)
    random_indices = random.sample(range(factorization.get_number_segments()), num_components)
    random_mask[random_indices] = True
    random_audio = factorization.get_segments_mask(random_mask)
    path_name_write = f"./quantitative_evaluation/current_rand.wav"
    soundfile.write(path_name_write, random_audio, fs)
    rand = predict_dicova.predict_single_audio(path_name_write)
    return morf, rand


def evaluate_data(components, run=0):
    for num_components in components:
        path_df = f"./quantitative_evaluation/{num_components}_summary.csv"
        df = pd.read_csv(path_df)
        data_array = df.to_numpy()
        shape_data = np.shape(data_array)
        number_data = 0
        true_exp = 0
        true_rand = 0
        for i in range(shape_data[0]):
            number_data += 1
            prediction_whole = np.rint(data_array[i, 2])
            prediction_exp = np.rint(data_array[i, 3])
            prediction_rand = np.rint(data_array[i, 4])
            if prediction_whole == prediction_exp:
                true_exp += 1
            if prediction_whole == prediction_rand:
                true_rand += 1

        # save these to txt file for further processing
        print("Number of samples", number_data)
        print("Number of true explanations", true_exp)
        print("Number of true random component predictions", true_rand)
        percentage_true_exp = float(true_exp) / float(number_data)
        percentage_rand = float(true_rand) / float(number_data)
        print("Percentage of true explanations", percentage_true_exp)
        print("Percentage of random true predictions", percentage_rand)
        path_save_summary = f"./quantitative_evaluation/{num_components}_components.txt"
        with open(path_save_summary, 'w') as summary:
            summary.write(f"Number samples: {number_data}")
            summary.write("\n")
            summary.write(f"Number true explanations: {true_exp}")
            summary.write("\n")
            summary.write(f"Number true random predictions: {true_rand}")
            summary.write("\n")
            summary.write(f"Percentage of true explanations: {percentage_true_exp}")
            summary.write("\n")
            summary.write(f"Percentage of random true predictions: {percentage_rand}")


def perform_quantitative_analysis(audio_dir, components=None, total_components=None, decomp='temporal'):
    # TODO: make entire new functions for loudness
    new_directory_name = './quantitative_evaluation'
    Path(new_directory_name).mkdir(parents=True, exist_ok=True)
    new_directory_name = f'./quantitative_evaluation/output_run_0'
    Path(new_directory_name).mkdir(parents=True, exist_ok=True)
    if decomp != 'loudness':
        if components is None:
            print("Error: please specify components, aborting")
            sys.exit()
        save_predictions_explanations(components, audio_directory_str=audio_dir, total_components=total_components, decomp_type=decomp)
        evaluate_data(components)
    elif decomp == 'loudness':
        levels = [0.1, 0.25, 0.5, 0.75, 1]
        save_predictions_explanations(levels, audio_directory_str=audio_dir, total_components=None, decomp_type=decomp)
        evaluate_data(levels)
    else:
        print("Error: decomposition type not recognized")
        sys.exit()


def significance_tests(total_runs):
    for run in range(total_runs):
        components = [1, 3, 5, 7]
        total_components = 7
        new_directory_name = './quantitative_evaluation'
        Path(new_directory_name).mkdir(parents=True, exist_ok=True)
        save_predictions_explanations(components, total_components)
        new_directory_name = f'./quantitative_evaluation/output_run_{run}'
        Path(new_directory_name).mkdir(parents=True, exist_ok=True)
        evaluate_data(components, run)
