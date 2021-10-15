import os
import librosa
from pathlib import Path
import sys
import soundfile
import warnings
import predict_dicova
import time
import quantitativeEvaluation
import pixelFlipping
from spectral_decomposition import SpectralDecomposition


def test_single_file():
    filename = 'osCanEgJ_cough.flac'
    audio_path = f'/Users/anne/Documents/Uni/Robotics/Masterarbeit/MA_Code/DICOVA/DiCOVA_Train_Val_Data_Release/AUDIO/{filename}'
    # TODO: adapt
    audio, sr = librosa.load(audio_path)
    explanation, decomposition = quantitativeEvaluation.get_explanation(audio, sr=sr, total_components=7, decomp_type='nmf', num_samples=64)
    print("Segments", decomposition.get_number_components())


if __name__ == '__main__':
    sys.path.append('/Users/anne/Documents/Uni/Robotics/Masterarbeit/MA_Code/DICOVA/DiCOVA_baseline')
    warnings.filterwarnings("ignore", message="Trying to unpickle estimator LogisticRegression from version 0.24.1 when using version 0.24.2. This might lead to breaking code or invalid results. Use at your own risk.")

    """test on single file"""
    # test_single_file()

    comps = list(range(7))
    audio = '/Users/anne/Documents/Uni/Robotics/Masterarbeit/MA_Code/DICOVA/DiCOVA_Train_Val_Data_Release/AUDIO/'
    list_file = '/Users/anne/Documents/Uni/Robotics/Masterarbeit/MA_Code/DICOVA/DiCOVA_Train_Val_Data_Release/LISTS/val_fold_1.txt'
    """quantitative evaluation as in audiolime"""
    # make folder for results of quantitative analysis
    # quantitativeEvaluation.perform_quantitative_analysis(audio, components=[1, 3, 5, 7], total_components=7, decomp='spectral')
    # significance analysis
    # total_runs = 5
    # quantitativeEvaluation.significance_tests(total_runs)

    """pixel flipping"""
    pixelFlipping.main_pixel_flipping('nmf', './eval/', audio, 64, comps, list_files=list_file)

