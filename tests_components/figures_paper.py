import sys
import warnings
import predict_dicova
import test_lime_cough_spectral
import librosa


# create two plots for two covid positive and two covid negative samples
def create_plots_single_file(filename):
    audio_path = f'/vol/bitbucket/aw821/DiCOVA_Train_Val_Data_Release/AUDIO/{filename}'
    predicted_entire = predict_dicova.predict_single_audio(audio_path)
    # TODO: adapt
    audio, fs = librosa.load(audio_path)
    total_components = 7
    explanation, factorization = test_lime_cough_spectral.get_explanation(audio, fs, total_components)
    explanation.show_image_mask_spectrogram(0, positive_only=True,
                                            negative_only=False, hide_rest=False,
                                            num_features=3, min_weight=0.,
                                            save_path=f'/Users/anne/PycharmProjects/LIME_cough/figures/{filename}_pos.png',
                                            show_colors=False)
    explanation.show_image_mask_spectrogram(0, positive_only=False, negative_only=False,
                                            hide_rest=False, num_features=3, min_weight=0.,
                                            save_path=f'/Users/anne/PycharmProjects/LIME_cough/figures/{filename}_both.png',
                                            show_colors=True)


if __name__ == '__main__':
    sys.path.append('/vol/bitbucket/aw821/DiCOVA_baseline')
    warnings.filterwarnings("ignore", message="Trying to unpickle estimator LogisticRegression from version 0.24.1 when using version 0.24.2. This might lead to breaking code or invalid results. Use at your own risk.")
    # positive files
    create_plots_single_file('DDdWaoiR_cough.flac')
    create_plots_single_file('DdvWRFdS_cough.flac')
    create_plots_single_file('HBqmFqKH_cough.flac')

    # negative files
    create_plots_single_file('HkTHkyvR_cough.flac')
    create_plots_single_file('KXRNyqiT_cough.flac')
    create_plots_single_file('MfBttSrb_cough.flac')