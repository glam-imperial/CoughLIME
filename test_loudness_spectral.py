import librosa
import sys
import warnings
import predict_dicova
import pixelFlipping
import quantitativeEvaluation
import os
import csv
import soundfile


def test_single_file():
    """
    function to test a single file with the loudness decomposition
    :return:
    """
    filename = 'BRdoMJMm_cough.flac'
    type_sample = 'neg'
    audio_path = f'/Users/anne/Documents/Uni/Robotics/Masterarbeit/MA_Code/DICOVA/DiCOVA_Train_Val_Data_Release/AUDIO/{filename}'
    # TODO: adapt
    print(predict_dicova.predict_single_audio(audio_path))
    audio, sr = librosa.load(audio_path)
    explanation, decomposition = quantitativeEvaluation.get_explanation(audio, sr=sr, decomp_type='ls', num_samples=10)


if __name__ == '__main__':
    #test on single file
    # TODO: adapt path
    sys.path.append('/Users/anne/Documents/Uni/Robotics/Masterarbeit/MA_Code/DICOVA/DiCOVA_baseline')
    warnings.filterwarnings("ignore", message="Trying to unpickle estimator LogisticRegression from version 0.24.1 when using version 0.24.2. This might lead to breaking code or invalid results. Use at your own risk.")

    """test on single file"""
    test_single_file()

    """quantitative evaluation as in audiolime"""
    # make folder for results of quantitative analysis
    # audio_directory_str = '/Users/anne/Documents/Uni/Robotics/Masterarbeit/MA_Code/DICOVA/DiCOVA_Train_Val_Data_Release/AUDIO'
    # quantitativeEvaluation.perform_quantitative_analysis(audio_directory_str, decomp='loudness')
    # significance analysis
    # total_runs = 5
    # quantitativeEvaluation.significance_tests(total_runs)

    data_path = '/Users/anne/Documents/Uni/Robotics/Masterarbeit/MA_Code/DICOVA/DiCOVA_Train_Val_Data_Release/AUDIO/'
    lists = '/Users/anne/Documents/Uni/Robotics/Masterarbeit/MA_Code/DICOVA/DiCOVA_Train_Val_Data_Release/LISTS/val_fold_1.txt'
    # test_figures()
    """pixel flipping"""
    # pixelFlipping.main_pixel_flipping('loudness', './eval/', data_path, 128, list_files=lists)
    # pixelFlipping.significance("loudness", './eval', data_path, 200, number_runs=5)
