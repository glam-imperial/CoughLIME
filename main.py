import librosa
import pixelFlipping
import quantitativeEvaluation
import soundfile


def test_single_file():

    """ preprocessing """
    filename = 'NBKvlVNm_cough.flac'
    audio_path = f'/vol/bitbucket/aw821/CoughLIME/DiCOVA_Train_Val_Data_Release/AUDIO/{filename}'
    # TODO: adapt
    audio, sr = librosa.load(audio_path)

    """ explanation generation """
    dec = 'loudness'  # adapt for decomposition type
    explanation, decomposition = quantitativeEvaluation.get_explanation(audio, sr=sr, decomp_type=dec, num_samples=200)
    decomposition.visualize_decomp(save_path='./figures/loudness_test.png')

    """ listenable examples """
    print("Segments", decomposition.get_number_components())
    audio, component_indices = explanation.get_exp_components(0, positive_components=True,
                                                              negative_components=True, num_components=3,
                                                              return_indices=True)
    path_write = f"./sonification/paper/{filename}_top3_loudness.wav"
    soundfile.write(path_write, audio, sr)

    """ visualizations """
    saving = f'./figures/{filename[:-5]}_neg.png'
    # figure highlighting the three most important components
    explanation.show_image_mask_spectrogram(0, False, False, False, 3, 0.0, saving, True, False)


if __name__ == "__main__":
    """ test on single file """
    test_single_file()

    """ pixel flipping evaluation """
    data_path = '/vol/bitbucket/aw821/DiCOVA_Train_Val_Data_Release/AUDIO/'
    lists = '/vol/bitbucket/aw821/DiCOVA_Train_Val_Data_Release/LISTS/val_fold_1.txt'

    comps = [0, 0.1, 0.25, 0.5, 0.75, 0.9]  # components to evaluate, percentages for loudness, integers for other types

    pixelFlipping.main_pixel_flipping('loudness', './eval/', data_path, 128, comps, list_files=lists)