import librosa
import sys
import warnings
import predict_dicova
import pixelFlipping
import quantitativeEvaluation


def test_single_file():
    audio_path = '/Users/anne/Documents/Uni/Robotics/Masterarbeit/MA_Code/DICOVA/DiCOVA_Train_Val_Data_Release/AUDIO/iyWdhFuN_cough.flac'
    predicted_entire = predict_dicova.predict_single_audio(audio_path)
    # TODO: adapt
    fs = librosa.get_samplerate(audio_path)
    audio, _ = librosa.load(audio_path, sr=fs)
    total_components = 7
    explanation, factorization = quantitativeEvaluation.get_explanation(audio, total_components)
    factorization.visualize_decomp(save_path='./figures/temporal_decomp.png')
    """filename = '1_test12344'
    quantitativeEvaluation.save_mix(explanation, 3, filename, factorization, fs, gen_random=False)
    path_name = f"./test/{filename[:-5]}_e.wav"
    prediction_exp = predict_dicova.predict_single_audio(path_name)
    print(predicted_entire)
    print(prediction_exp)
    figure = explanation.as_pyplot_figure()
    figure.show()
    figure.savefig('./explanation.png')"""


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
