import librosa
from pathlib import Path
import sys
import soundfile
import warnings
import predict_dicova
import time
import quantitativeEvaluation
import pixelFlipping


def test_single_file():
    start = time.time()
    filename = 'cNMOJqng_cough.flac'
    audio_path = f'/Users/anne/Documents/Uni/Robotics/Masterarbeit/MA_Code/DICOVA/DiCOVA_Train_Val_Data_Release/AUDIO/{filename}'
    predicted_entire = predict_dicova.predict_single_audio(audio_path)
    # TODO: adapt
    audio, fs = librosa.load(audio_path)
    total_components = 7
    explanation, factorization = quantitativeEvaluation.get_explanation(audio, fs, total_components)
    new_filename = f'test_{filename}'
    quantitativeEvaluation.save_mix(explanation, 1, new_filename, factorization, fs, gen_random=False)
    quantitativeEvaluation.save_mix(explanation, 3, new_filename, factorization, fs, gen_random=False)
    path_name = f"./quantitative_evaluation/3_components/explanations/{new_filename[:-5]}_e.wav"
    prediction_exp = predict_dicova.predict_single_audio(path_name)
    print(predicted_entire)
    print(prediction_exp)
    weighted_audio = explanation.weighted_audio(0, positive_components=True, negative_components=True, num_components=3)
    path_name = f"./spectral_tests/{new_filename[:-5]}_3_weighted.wav"
    soundfile.write(path_name, weighted_audio, fs)
    # explanation.show_image_mask_spectrogram(0, positive_only=True, negative_only=False, hide_rest=True, num_features=3, min_weight=0., save_path='./tests_components/mel_pos_hide.png')
    # explanation.show_image_mask_spectrogram(0, positive_only=False, negative_only=False, hide_rest=True, num_features=3, min_weight=0., save_path='./tests_components/mel_both_hide.png')
    explanation.show_image_mask_spectrogram(0, positive_only=True, negative_only=False, hide_rest=False, num_features=3, min_weight=0., save_path='./tests_components/mel_pos_show.png', show_colors=True)
    explanation.show_image_mask_spectrogram(0, positive_only=False, negative_only=False, hide_rest=False, num_features=3, min_weight=0., save_path='./tests_components/mel_both_show.png', show_colors=True)
    end = time.time()
    print("time elapsed for mel:", end - start)


if __name__ == '__main__':
    sys.path.append('/Users/anne/Documents/Uni/Robotics/Masterarbeit/MA_Code/DICOVA/DiCOVA_baseline')
    warnings.filterwarnings("ignore", message="Trying to unpickle estimator LogisticRegression from version 0.24.1 when using version 0.24.2. This might lead to breaking code or invalid results. Use at your own risk.")

    """test on single file"""
    # test_single_file()

    audio = '/Users/anne/Documents/Uni/Robotics/Masterarbeit/MA_Code/DICOVA/DiCOVA_Train_Val_Data_Release/AUDIO/'
    list = '/Users/anne/Documents/Uni/Robotics/Masterarbeit/MA_Code/DICOVA/DiCOVA_Train_Val_Data_Release/LISTS/val_fold_1.txt'
    """quantitative evaluation as in audiolime"""
    # make folder for results of quantitative analysis
    # quantitativeEvaluation.perform_quantitative_analysis(audio, components=[1, 3, 5, 7], total_components=7, decomp='spectral')
    # significance analysis
    # total_runs = 5
    # quantitativeEvaluation.significance_tests(total_runs)

    """pixel flipping"""
    pixelFlipping.main_pixel_flipping(7, 'spectral', './eval/', audio, 64, list_files=list)



