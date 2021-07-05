import librosa
import numpy as np
import configparser
import pickle


def compute_mfcc(s, config):
    # computes MFCC of s as defined in the dicova baseline code
    mfcc = librosa.feature.mfcc(s, sr=int(config['default']['sampling_rate']),
                             n_mfcc=int(config['mfcc']['n_mfcc']),
                             n_fft=int(config['default']['window_size']),
                             hop_length=int(config['default']['window_shift']),
                             n_mels=int(config['mfcc']['n_mels']),
                             fmax=int(config['mfcc']['fmax']))

    features = np.array(mfcc)
    if config['mfcc']['add_deltas'] in ['True', 'true', 'TRUE', '1']:
        deltas = librosa.feature.delta(F)
        features = np.concatenate((features, deltas), axis=0)

    if config['mfcc']['add_delta_deltas'] in ['True', '∞true', 'TRUE', '1']:
        ddeltas = librosa.feature.delta(F, order=2)
        features = np.concatenate((features, ddeltas), axis=0)

    return features


def predict(input_audio):
    # predicts output label for batch_size audios at a time
    # based on dicova baseline code, slightly adapted for audioLIME
    # TODO: adapt paths
    this_config = '/Users/anne/Documents/Uni/Robotics/Masterarbeit/MA_Code/DICOVA/DiCOVA_baseline/conf/feature.conf'
    path_model = '/Users/anne/Documents/Uni/Robotics/Masterarbeit/MA_Code/DICOVA/DiCOVA_baseline/results_lr/fold_1/model.pkl'

    config = configparser.ConfigParser()
    config.read(this_config)
    batch_size = len(input_audio)
    labels = np.zeros((batch_size, 1))

    file_model = open(path_model, 'rb')
    rf_model = pickle.load(file_model)

    for i, audio in enumerate(input_audio):
        F = compute_mfcc(audio.flatten('F'), config)

        score = rf_model.validate([F.T])
        score = np.mean(score[0], axis=0)[1]
        labels[i, 0] = score
    return labels


def predict_single_audio(audio_path):
    # predicts output label
    # based on dicova baseline code, slightly adapted for audioLIME
    # TODO: update paths
    this_config = '/Users/anne/Documents/Uni/Robotics/Masterarbeit/MA_Code/DICOVA/DiCOVA_baseline/conf/feature.conf'
    path_model = '/Users/anne/Documents/Uni/Robotics/Masterarbeit/MA_Code/DICOVA/DiCOVA_baseline/results_lr/fold_1/model.pkl'

    config = configparser.ConfigParser()
    config.read(this_config)

    file_model = open(path_model, 'rb')
    rf_model = pickle.load(file_model)

    sample_rate = librosa.get_samplerate(audio_path)
    audio_array, _ = librosa.load(audio_path, sr=sample_rate)

    # this line is the problem why librosa outputs nan
    if np.max(np.abs(audio_array)) != 0:
        audio_array = audio_array/np.max(np.abs(audio_array))

    mfcc = compute_mfcc(audio_array, config)

    score = rf_model.validate([mfcc.T])
    label = np.mean(score[0], axis=0)[1]
    return label
