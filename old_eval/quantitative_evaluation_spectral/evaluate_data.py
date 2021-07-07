import pandas as pd
import numpy as np
import sys
import os
import warnings
import predict_dicova


def make_summary(components):
    original_audios = '/Users/anne/Documents/Uni/Robotics/Masterarbeit/MA_Code/DICOVA/DiCOVA_Train_Val_Data_Release/AUDIO'
    processed_files = './1_components/explanations/'
    audio_directory = os.fsencode(processed_files)

    file_names = []
    predictions_entire_file = []
    comp_exp = []
    comp_random = []
    for _ in components:
        comp_exp.append([])
        comp_random.append([])
    counter = 0

    for file in os.listdir(audio_directory):
        filename = os.fsdecode(file)
        if filename.endswith(".wav"):
            print("Starting with... ", filename)
            path_file = f'{original_audios}/{filename[:-6]}.flac'
            file_names.append(filename)

            # get prediction for whole audio file
            prediction_overall = predict_dicova.predict_single_audio(path_file)
            predictions_entire_file.append(prediction_overall)

            # get mixes for top_components and save them
            for index, num_components in enumerate(components):
                # get predictions
                path_name = f"./{num_components}_components/explanations/{filename[:-6]}_e.wav"
                prediction_exp = predict_dicova.predict_single_audio(path_name)

                path_name = f"./{num_components}_components/random_components/{filename[:-6]}_r.wav"
                prediction_rand = predict_dicova.predict_single_audio(path_name)

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
        path_csv = f"./{num_components}_components/summary.csv"
        summary_df.to_csv(path_csv)


def evaluate_data(components):
    for num_components in components:
        path_df = f'./{num_components}_components/summary.csv'
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
        path_save_summary = f"./{num_components}_components.txt"
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


if __name__ == '__main__':
    #test on single file
    # TODO: adapt path
    sys.path.append('/Users/anne/Documents/Uni/Robotics/Masterarbeit/MA_Code/DICOVA/DiCOVA_baseline')
    sys.path.append('/Users/anne/PycharmProjects/LIME_cough')
    warnings.filterwarnings("ignore", message="Trying to unpickle estimator LogisticRegression from version 0.24.1 when using version 0.24.2. This might lead to breaking code or invalid results. Use at your own risk.")
    components = [1, 3, 5, 7]
    make_summary(components)
    evaluate_data(components)
