import librosa
import sys
import warnings
import predict_dicova
import pixelFlipping
import quantitativeEvaluation
sys.path.append("./cider/")
import predict_cider
import os
import csv
import soundfile
from pathlib import Path


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
    explanation, decomposition = quantitativeEvaluation.get_explanation(audio, sr=sr, decomp_type='loudness', num_samples=200)
    # decomposition.visualize_decomp(save_path='./figures/loudness_test.png')
    saving = f'./figures/{filename[:-5]}_neg.png'
    explanation.show_image_mask_spectrogram(0, False, False, False, 3, 0.0, saving, True, False)

    # for c in [1, 3, 5]:
    #    audio, component_indeces = explanation.get_exp_components(0, positive_components=True,
                                                              #negative_components=True,
                                                              #num_components=c,
                                                              #return_indeces=True)
        # num components: how many components should model take for explanations
    #    path_name_write = f"./sonification/paper/{filename}_{type_sample}_top{c}_loudness.wav"
    #    soundfile.write(path_name_write, audio, sr)
    path = f'./figures/paper /decomp_{filename}.png'
    decomposition.visualize_decomp(save_path=path)
    # explanation.show_image_mask_spectrogram(0, positive_only=False, negative_only=False, hide_rest=False, num_features=3, min_weight=0., save_path=path, show_colors=True)
    # path_name = f"./sonification/{filename[:-5]}_e_3_comp.wav"
    # quantitativeEvaluation.save_mix(explanation, 3, path_name, decomposition, sr, gen_random=False)
    # path_name = f"./sonification/{filename[:-5]}_e_1_comp.wav"
    # quantitativeEvaluation.save_mix(explanation, 1, path_name, decomposition, sr, gen_random=False)


def test_figures():
    # file_list = get_loudness_list()
    data_directory = '/Users/anne/Documents/Uni/Robotics/Masterarbeit/MA_Code/DICOVA/DiCOVA_Train_Val_Data_Release/AUDIO'
    # audio_directory = os.fsencode(data_directory)
    with open('./figures/pos_neg/negative_options') as f:
        files_to_process = [line.rstrip() for line in f]
    for file in files_to_process:  # os.listdir(audio_directory):
        # filename = os.fsdecode(file)
        print("Starting with... ", file)
        path_file = f'{data_directory}/{file}.flac'
        audio, sr = librosa.load(path_file)
        explanation, decomposition = quantitativeEvaluation.get_explanation(audio, sr=sr, decomp_type='loudness', num_samples=200)
        decomposition.visualize_decomp()
        print("Segments", decomposition.get_number_components())
        saving = f'./figures/pos_neg/{file}_neg.png'
        explanation.show_image_mask_spectrogram(0, positive_only=False, negative_only=False, hide_rest=False, num_features=3, min_weight=0., save_path=saving, show_colors=True, show_loudness=False)
        print("done")


def get_loudness_list():
    res = []
    data = '/Users/anne/Documents/Uni/Robotics/Masterarbeit/MA_Code/DICOVA/DiCOVA_Train_Val_Data_Release/metadata.csv'
    read_file = open(data, 'r')
    csv_reader = csv.reader(read_file)
    _ = next(csv_reader)
    for row in csv_reader:
        if row[1] == 'p':
            res.append(row[0])
    return res


if __name__ == '__main__':
    #test on single file
    # TODO: adapt path
    sys.path.append('/Users/anne/Documents/Uni/Robotics/Masterarbeit/MA_Code/DICOVA/DiCOVA_baseline')
    warnings.filterwarnings("ignore", message="Trying to unpickle estimator LogisticRegression from version 0.24.1 "
                                              "when using version 0.24.2. This might lead to breaking code or invalid "
                                              "results. Use at your own risk.")

    """test on single file"""
    # test_single_file()

    """quantitative evaluation as in audiolime"""
    # make folder for results of quantitative analysis
    # audio_directory_str = '/Users/anne/Documents/Uni/Robotics/Masterarbeit/MA_Code/DICOVA/DiCOVA_Train_Val_Data_Release/AUDIO'
    # quantitativeEvaluation.perform_quantitative_analysis(audio_directory_str, decomp='loudness')
    # significance analysis
    # total_runs = 5
    # quantitativeEvaluation.significance_tests(total_runs)
    sr = 24000
    data_path = '/Users/anne/Documents/Uni/Robotics/Masterarbeit/MA_Code/DICOVA/DiCOVA_Train_Val_Data_Release/AUDIO/'
    lists = '/Users/anne/Documents/Uni/Robotics/Masterarbeit/MA_Code/DICOVA/DiCOVA_Train_Val_Data_Release' \
            '/LISTS/val_fold_1.txt'
    ts = [45, 55, 65, 75, 85, 95]
    comp = [0, 0.1, 0.25, 0.5, 0.75, 0.9]
    for t in ts:
        output = f'./eval2/threshold_{t}/'
        Path(output).mkdir(parents=True, exist_ok=True)
        """pixel flipping"""
        pixelFlipping.main_pixel_flipping('loudness', output, data_path, 10, comp, threshold=t,
                                          predict_fn=predict_cider.predict, sr=sr, list_files=lists)
