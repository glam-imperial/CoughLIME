import librosa
import pixel_flipping
import quantitative_evaluation
import soundfile
import predict_dicova
import sys
sys.path.append("./cider/")
import predict_cider
import make_graphs_evaluation


def test_single_file():
    """ preprocessing """
    filename = 'BRdoMJMm_cough.flac'
    audio_path = f'/vol/bitbucket/aw821/CoughLIME/DiCOVA_Train_Val_Data_Release/AUDIO/{filename}'  # path to audio files
    sr = 24000
    audio, _ = librosa.load(audio_path, sr=sr)

    """ explanation generation """
    dec = 'loudness'  # adapt for decomposition type
    explanation, decomposition = pixel_flipping.get_explanation(audio, None, sr, predict_cider.predict, num_samples=64,
                                                                threshold=75, decomposition_type=dec)

    decomposition.visualize_decomp(save_path='./figures/loudness_test.png')  # visualizes the decomposition

    """ listenable examples """
    audio, component_indices = explanation.get_exp_components(0, positive_components=True,
                                                              negative_components=True, num_components=3,
                                                              return_indices=True)
    path_write = f"./listenable_examples/{filename}_top3.wav"
    soundfile.write(path_write, audio, sr)

    path_weighted = f"./listenable_examples/{filename}_top3_weighted.wav"
    weighted = explanation.weighted_audio(0, True, True, 3)
    soundfile.write(path_weighted, weighted, sr)

    """ visualizations """
    save_path = f'./figures/{filename[:-5]}top3.png'
    # figure highlighting the three most important components
    explanation.show_image_mask_spectrogram(0, positive_only=False, negative_only=False, hide_rest=False,
                                            num_features=3, min_weight=0., save_path=save_path, show_colors=True)


if __name__ == "__main__":
    """ test on single file """
    test_single_file()

    """ pixel flipping evaluation """
    data_path = '/vol/bitbucket/aw821/DiCOVA_Train_Val_Data_Release/AUDIO/'
    list_files = '/vol/bitbucket/aw821/DiCOVA_Train_Val_Data_Release/LISTS/val_fold_1.txt'

    # components to evaluate, percentages for loudness and ls, integers for other types
    comps = [0, 0.1, 0.25, 0.5, 0.75, 0.9]
    dec = 'loudness'
    sr = 24000

    pixel_flipping.main_pixel_flipping(dec, './eval/', data_path, 64, comps, 75, predict_cider.predict, sr, list_files)
    make_graphs_evaluation.make_single_graph(comps, './eval', dauc=True)
