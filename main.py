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

    """ preprocessing """
    filename = 'NBKvlVNm_cough.flac'
    type_sample = 'neg'
    audio_path = f'/Users/anne/Documents/Uni/Robotics/Masterarbeit/MA_Code/DICOVA/DiCOVA_Train_Val_Data_Release/AUDIO/{filename}'
    # TODO: adapt
    audio, sr = librosa.load(audio_path)

    """ explanation generation """
    explanation, decomposition = quantitativeEvaluation.get_explanation(audio, sr=sr, decomp_type='loudness', num_samples=200)
    # decomposition.visualize_decomp(save_path='./figures/loudness_test.png')

    """ listenable examples """
    print("Segments", decomposition.get_number_components())

    """ visualizations """


if __name__ == "__main__":
    """ test on single file """