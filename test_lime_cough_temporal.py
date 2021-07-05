import os
import pickle
import numpy as np
import librosa
import configparser
from pathlib import Path
import pandas as pd
import sys
import lime_cough
from temporal_segmentation import TemporalSegmentation
import soundfile
import warnings
import random


def compute_mfcc(s, config):
    # computes MFCC of s as defined in the dicova baseline code
    F = librosa.feature.mfcc(s, sr=int(config['default']['sampling_rate']),
                             n_mfcc=int(config['mfcc']['n_mfcc']),
                             n_fft=int(config['default']['window_size']),
                             hop_length=int(config['default']['window_shift']),
                             n_mels=int(config['mfcc']['n_mels']),
                             fmax=int(config['mfcc']['fmax']))

    features = np.array(F)
    if config['mfcc']['add_deltas'] in ['True', 'true', 'TRUE', '1']:
        deltas = librosa.feature.delta(F) #width=5
        features = np.concatenate((features, deltas), axis=0)

    if config['mfcc']['add_delta_deltas'] in ['True', '∞true', 'TRUE', '1']:
        ddeltas = librosa.feature.delta(F, order=2) #width=5
        features = np.concatenate((features, ddeltas), axis=0)

    return features


def predict(input_audio):
    # predicts output label for batch_size audios at a time
    # based on dicova baseline code, slightly adapted for audioLIME
    # TODO: adapt paths
    this_config = '/Users/anne/Documents/Uni/Robotics/Masterarbeit/MA_Code/DICOVA/DiCOVA_baseline/conf/feature.conf'
    path_model = '/Users/anne/Documents/Uni/Robotics/Masterarbeit/MA_Code/DICOVA/DiCOVA_baseline/results_lr/fold_1/model.pkl'

    config = configparser.ConfigParser()
    config.read(this_config)
    batch_size = len(input_audio)
    labels = np.zeros((batch_size, 1))

    file_model = open(path_model, 'rb')
    rf_model = pickle.load(file_model)

    for i, audio in enumerate(input_audio):
        F = compute_mfcc(audio.flatten('F'), config)

        score = rf_model.validate([F.T])
        score = np.mean(score[0], axis=0)[1]
        labels[i, 0] = score
    return labels


def predict_single_audio(audio_path):
    # predicts output label
    # based on dicova baseline code, slightly adapted for audioLIME
    # TODO: update paths
    this_config = '/Users/anne/Documents/Uni/Robotics/Masterarbeit/MA_Code/DICOVA/DiCOVA_baseline/conf/feature.conf'
    path_model = '/Users/anne/Documents/Uni/Robotics/Masterarbeit/MA_Code/DICOVA/DiCOVA_baseline/results_lr/fold_1/model.pkl'

    config = configparser.ConfigParser()
    config.read(this_config)

    file_model = open(path_model, 'rb')
    rf_model = pickle.load(file_model)

    sample_rate = librosa.get_samplerate(audio_path)
    audio_array, _ = librosa.load(audio_path, sr=sample_rate)

    # this line is the problem why librosa outputs nan
    if np.max(np.abs(audio_array)) != 0:
        audio_array = audio_array/np.max(np.abs(audio_array))

    F = compute_mfcc(audio_array, config)

    score = rf_model.validate([F.T])
    label = np.mean(score[0], axis=0)[1]
    return label


def get_explanation(audio):
    factorization = TemporalSegmentation(audio, 7)
    explainer = lime_cough.LimeCoughExplainer()
    explanation = explainer.explain_instance(segmentation=factorization,
                                             classifier_fn=predict,
                                             labels=[0],
                                             num_samples=64,
                                             batch_size=16,
                                             )
    # num_samples: how many perturbations of the input audio components to use to train the Ridge classifier for the explanations (part of the lime base code)
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
        # TODO: adapt random generation, generate random mask
        random_mask = np.zeros(factorization.get_number_segments(),).astype(bool)
        random_indices = random.sample(range(factorization.get_number_segments()), num_components)
        random_mask[random_indices] = True
        random_audio = factorization.get_segments_mask(random_mask)
        path_name_write = f"./quantitative_evaluation/{num_components}_components/random_components/{filename[:-5]}_r.wav"
        soundfile.write(path_name_write, random_audio, sample_rate)


def save_predictions_explanations(components):
    # for each audio file
    # make directory for the number of components if not exists
    file_names = []
    predictions_entire_file = []
    comp_exp = []
    comp_random = []
    for num_components in components:
        new_directory = f"./quantitative_evaluation/{num_components}_components"
        Path(new_directory).mkdir(parents=True, exist_ok=True)
        new_directory = f"./quantitative_evaluation/{num_components}_components/explanations"
        Path(new_directory).mkdir(parents=True, exist_ok=True)
        new_directory = f"./quantitative_evaluation/{num_components}_components/random_components"
        Path(new_directory).mkdir(parents=True, exist_ok=True)

        comp_exp.append([])
        comp_random.append([])

    # TODO: adapt path
    audio_directory_str = '/Users/anne/Documents/Uni/Robotics/Masterarbeit/MA_Code/DICOVA/DiCOVA_Train_Val_Data_Release/AUDIO'
    audio_directory = os.fsencode(audio_directory_str)

    counter = 0

    for file in os.listdir(audio_directory):
        filename = os.fsdecode(file)
        if filename.endswith(".flac"):
            print("Starting with... ", filename)
            path_file = f'{audio_directory_str}/{filename}'
            file_names.append(filename)

            # get prediction for whole audio file
            prediction_overall = predict_single_audio(path_file)
            predictions_entire_file.append(prediction_overall)

            # get explanation
            fs = librosa.get_samplerate(path_file)
            audio, _ = librosa.load(path_file, sr=fs)
            explanation, factorization = get_explanation(audio)

            # get mixes for top_components and save them
            for index, num_components in enumerate(components):
                save_mix(explanation, num_components, filename, factorization, fs, gen_random=True)

                # get predictions
                path_name = f"./quantitative_evaluation/{num_components}_components/explanations/{filename[:-5]}_e.wav"
                prediction_exp = predict_single_audio(path_name)

                path_name = f"./quantitative_evaluation/{num_components}_components/random_components/{filename[:-5]}_r.wav"
                prediction_rand = predict_single_audio(path_name)

                comp_exp[index].append(prediction_exp)
                comp_random[index].append(prediction_rand)
        else:
            print("Information: different file extension detected: ", file)
        counter += 1
        print("progress report:", counter)

    # create csv "summary" with pandas for every component
    for index, num_components in enumerate(components):
        summary_dict = {'File_name': file_names, 'Prediction_whole_file': predictions_entire_file, 'Prediction_top_comp': comp_exp[index], 'Prediction_random_comp': comp_random[index]}
        summary_df = pd.DataFrame.from_dict(summary_dict)
        path_csv = f"./quantitative_evaluation/{num_components}_components/summary.csv"
        summary_df.to_csv(path_csv)


def evaluate_data(components):
    for num_components in components:
        # tested, nothing to adapt
        path_df = f'./quantitative_evaluation/{num_components}_components/summary.csv'
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


def test_single_file():
    audio_path = '/Users/anne/Documents/Uni/Robotics/Masterarbeit/MA_Code/DICOVA/DiCOVA_Train_Val_Data_Release/AUDIO/AtACyGlV_cough.flac'
    predicted_entire = predict_single_audio(audio_path)
    # TODO: adapt
    fs = librosa.get_samplerate(audio_path)
    audio, _ = librosa.load(audio_path, sr=fs)
    explanation, factorization = get_explanation(audio)
    filename = '1_test12344'
    save_mix(explanation, 3, filename, factorization, fs, gen_random=False)
    path_name = f"./test/{filename[:-5]}_e.wav"
    prediction_exp = predict_single_audio(path_name)
    print(predicted_entire)
    print(prediction_exp)
    figure = explanation.as_pyplot_figure()
    figure.show()
    figure.savefig('./explanation.png')


def perform_quantitative_analysis():
    components = [1, 3, 5, 7]
    new_directory_name = './quantitative_evaluation'
    Path(new_directory_name).mkdir(parents=True, exist_ok=True)
    save_predictions_explanations(components)
    # check from here
    evaluate_data(components)


if __name__ == '__main__':
    #test on single file
    # TODO: adapt path
    sys.path.append('/Users/anne/Documents/Uni/Robotics/Masterarbeit/MA_Code/DICOVA/DiCOVA_baseline')
    warnings.filterwarnings("ignore", message="Trying to unpickle estimator LogisticRegression from version 0.24.1 when using version 0.24.2. This might lead to breaking code or invalid results. Use at your own risk.")
    test_single_file()

    # make folder for results of quantitative analysis
    #perform_quantitative_analysis()