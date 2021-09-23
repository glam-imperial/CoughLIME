import librosa
import sys
import warnings
import predict_dicova
import pixelFlipping
import quantitativeEvaluation


def test_single_file():
    filename = 'rAKaNjve_cough.flac'
    audio_path = f'/Users/anne/Documents/Uni/Robotics/Masterarbeit/MA_Code/DICOVA/DiCOVA_Train_Val_Data_Release/AUDIO/{filename}'
    # TODO: adapt
    audio, sr = librosa.load(audio_path)
    explanation, factorization = quantitativeEvaluation.get_explanation(audio, sr=sr, decomp_type='loudness')
    # factorization.visualize_decomp(save_path='./figures/loudness_test.png')
    print("Segments", factorization.get_number_segments())
    explanation.show_image_mask_spectrogram(0, positive_only=False, negative_only=False, hide_rest=False, num_features=3, min_weight=0., save_path=None, show_colors=True)
    # path_name = f"./sonification/{filename[:-5]}_e_3_comp.wav"
    # quantitativeEvaluation.save_mix(explanation, 3, path_name, factorization, sr, gen_random=False)
    # path_name = f"./sonification/{filename[:-5]}_e_1_comp.wav"
    # quantitativeEvaluation.save_mix(explanation, 1, path_name, factorization, sr, gen_random=False)


if __name__ == '__main__':
    #test on single file
    # TODO: adapt path
    sys.path.append('/Users/anne/Documents/Uni/Robotics/Masterarbeit/MA_Code/DICOVA/DiCOVA_baseline')
    warnings.filterwarnings("ignore", message="Trying to unpickle estimator LogisticRegression from version 0.24.1 when using version 0.24.2. This might lead to breaking code or invalid results. Use at your own risk.")

    """test on single file"""
    # test_single_file()

    """quantitative evaluation as in audiolime"""
    # make folder for results of quantitative analysis
    # audio_directory_str = '/Users/anne/Documents/Uni/Robotics/Masterarbeit/MA_Code/DICOVA/DiCOVA_Train_Val_Data_Release/AUDIO'
    # quantitativeEvaluation.perform_quantitative_analysis(audio_directory_str, decomp='loudness')
    # significance analysis
    # total_runs = 5
    # quantitativeEvaluation.significance_tests(total_runs)

    data_path = '/Users/anne/Documents/Uni/Robotics/Masterarbeit/MA_Code/DICOVA/DiCOVA_Train_Val_Data_Release/AUDIO/'
    lists = '/Users/anne/Documents/Uni/Robotics/Masterarbeit/MA_Code/DICOVA/DiCOVA_Train_Val_Data_Release/LISTS/val_fold_1.txt'

    """pixel flipping"""
    # pixelFlipping.main_pixel_flipping('loudness', './eval/', data_path, 128, list_files=lists)
    pixelFlipping.significance("loudness", './eval', data_path, 200, number_runs=5)
