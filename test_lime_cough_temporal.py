import librosa
import sys
import warnings
import predict_dicova
import pixelFlipping
import quantitativeEvaluation
import soundfile


def test_single_file():
    filename = 'tejPPvGf_cough.flac'
    type_sample = 'neg'
    audio_path = f'/Users/anne/Documents/Uni/Robotics/Masterarbeit/MA_Code/DICOVA/DiCOVA_Train_Val_Data_Release/AUDIO/{filename}'
    # TODO: adapt
    audio, sr = librosa.load(audio_path)
    print(predict_dicova.predict_single_audio(audio_path))
    explanation, decomposition = quantitativeEvaluation.get_explanation(audio, total_components=7, sr=sr, decomp_type='temporal', num_samples=200)
    # decomposition.visualize_decomp(save_path='./figures/loudness_test.png')
    print("Components", decomposition.get_number_components())
    for c in [1, 3, 5]:
        audio, component_indices = explanation.get_exp_components(0, positive_components=True,
                                                                  negative_components=True,
                                                                  num_components=c,
                                                                  return_indices=True)
        # num components: how many components should model take for explanations
        path_name_write = f"./sonification/paper/{filename}_{type_sample}_top{c}_temporal.wav"
        soundfile.write(path_name_write, audio, sr)


if __name__ == '__main__':
    #test on single file
    # TODO: adapt path
    sys.path.append('/Users/anne/Documents/Uni/Robotics/Masterarbeit/MA_Code/DICOVA/DiCOVA_baseline')
    warnings.filterwarnings("ignore", message="Trying to unpickle estimator LogisticRegression from version 0.24.1 when using version 0.24.2. This might lead to breaking code or invalid results. Use at your own risk.")

    """test on single file"""
    test_single_file()

    """quantitative evaluation as in audiolime"""
    # make folder for results of quantitative analysis
    # quantitativeEvaluation.perform_quantitative_analysis()
    # significance analysis
    # total_runs = 5
    # quantitativeEvaluation.significance_tests(total_runs)

    """pixel flipping"""
    # pixelFlipping.main_pixel_flipping(7, 'temporal', './eval/', '/Users/anne/Documents/Uni/Robotics/Masterarbeit/MA_Code/DICOVA/DiCOVA_Train_Val_Data_Release/AUDIO/', 200)
