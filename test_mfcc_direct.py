import os
import numpy as np
import librosa
from pathlib import Path
import pandas as pd
import sys
import lime_cough
from segmentation_mfcc_direct import MFCCDirectSegmentation
import soundfile
import warnings
import random
import predict_from_mfcc
import time
import configparser


def get_explanation(audio, sample_rate, total_components):
    this_config = '/Users/anne/Documents/Uni/Robotics/Masterarbeit/MA_Code/DICOVA/DiCOVA_baseline/conf/feature.conf'
    config = configparser.ConfigParser()
    config.read(this_config)
    factorization = MFCCDirectSegmentation(audio, sample_rate, total_components, config=config)
    explainer = lime_cough.LimeCoughExplainer()
    explanation = explainer.explain_instance(segmentation=factorization,
                                             classifier_fn=predict_from_mfcc.predict,
                                             labels=[0],
                                             num_samples=64,
                                             batch_size=16,
                                             )
    return explanation, factorization


def save_mix(explanation, num_components, filename, factorization, sample_rate, gen_random=False):
    label = list(explanation.local_exp.keys())[0]
    _, component_indeces = explanation.get_exp_components(label, positive_components=True,
                                                              negative_components=True,
                                                              num_components=num_components,
                                                              return_indeces=True)
    # num components: how many components should model take for explanations
    path_name_write = f"./quantitative_evaluation/{num_components}_components/explanations/{filename[:-5]}_e.wav"
    audio = factorization.return_audio_from_indices(component_indeces)
    soundfile.write(path_name_write, audio, sample_rate)
    print("Indices", component_indeces)
    if gen_random:
        # random components must also be generated
        random_indices = random.sample(range(factorization.get_number_segments()), num_components)
        random_audio = factorization.return_audio_from_indices(random_indices)
        path_name_write = f"./quantitative_evaluation/{num_components}_components/random_components/{filename[:-5]}_r.wav"
        soundfile.write(path_name_write, random_audio, sample_rate)


def save_predictions_explanations(components, total_components):
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
            prediction_overall = predict_from_mfcc.predict_single_audio(path_file)
            predictions_entire_file.append(prediction_overall)

            # get explanation
            fs = librosa.get_samplerate(path_file)
            audio, _ = librosa.load(path_file, sr=fs)
            explanation, factorization = get_explanation(audio, fs, total_components)

            # get mixes for top_components and save them
            for index, num_components in enumerate(components):
                save_mix(explanation, num_components, filename, factorization, fs, gen_random=True)

                # get predictions
                path_name = f"./quantitative_evaluation/{num_components}_components/explanations/{filename[:-5]}_e.wav"
                prediction_exp = predict_from_mfcc.predict_single_audio(path_name)

                path_name = f"./quantitative_evaluation/{num_components}_components/random_components/{filename[:-5]}_r.wav"
                prediction_rand = predict_from_mfcc.predict_single_audio(path_name)

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
    start = time.time()
    audio_path = '/Users/anne/Documents/Uni/Robotics/Masterarbeit/MA_Code/DICOVA/DiCOVA_Train_Val_Data_Release/AUDIO/AtACyGlV_cough.flac'
    predicted_entire = predict_from_mfcc.predict_single_audio(audio_path)
    audio, fs = librosa.load(audio_path)
    total_components = 7
    explanation, factorization = get_explanation(audio, fs, total_components)
    filename = 'mfcc_test1234'
    save_mix(explanation, 3, filename, factorization, fs, gen_random=False)
    path_name = f"./quantitative_evaluation/3_components/explanations/{filename[:-5]}_e.wav"
    prediction_exp = predict_from_mfcc.predict_single_audio(path_name)
    print(predicted_entire)
    print(prediction_exp)
    figure = explanation.as_pyplot_figure()
    figure.show()
    figure.savefig('./explanation_mfcc.png')
    explanation.show_image_mask_spectrogram(0, positive_only=True, negative_only=False, hide_rest=False, num_features=5, min_weight=0., save_path='./tests_components/test1_mfcc.png')
    explanation.show_image_mask_spectrogram(0, positive_only=False, negative_only=False, hide_rest=False, num_features=5, min_weight=0., save_path='./tests_components/test2_mfcc.png')
    end = time.time()
    print("time taken for mfcc", end - start)


def perform_quantitative_analysis():
    components = [1, 3, 5, 7]
    total_components = 7
    new_directory_name = './quantitative_evaluation'
    Path(new_directory_name).mkdir(parents=True, exist_ok=True)
    save_predictions_explanations(components, total_components)
    evaluate_data(components)


if __name__ == '__main__':
    #test on single file
    # TODO: adapt path
    sys.path.append('/Users/anne/Documents/Uni/Robotics/Masterarbeit/MA_Code/DICOVA/DiCOVA_baseline')
    warnings.filterwarnings("ignore", message="Trying to unpickle estimator LogisticRegression from version 0.24.1 when using version 0.24.2. This might lead to breaking code or invalid results. Use at your own risk.")
    test_single_file()

    # make folder for results of quantitative analysis
    #perform_quantitative_analysis()