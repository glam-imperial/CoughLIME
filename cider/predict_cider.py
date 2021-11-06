import os
from conv_model_dicova import Conv_Model
import torch
import numpy as np
from torchvision import transforms
if torch.cuda.is_available():
    from data_preprocessing_dicova import COVID_dataset
else:
    from data_preprocessing_dicova_local import COVID_dataset
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, roc_curve, auc, recall_score
from tqdm import tqdm
import torch.nn as nn
import librosa


def custom_transform(signal, nfft=512):
    stft = librosa.stft(signal, n_fft=nfft, hop_length=512)
    spectrogram = np.abs(stft)
    features = librosa.amplitude_to_db(spectrogram)
    transform_tensor = transforms.Compose([
        transforms.ToTensor(),
    ])
    features = transform_tensor(features)
    return features


def pad(signal, window_size):
    sample_signal = np.zeros((window_size,))
    sample_signal[:signal.shape[0], ] = signal
    return sample_signal


def process_chunk(chunk, window_size):
    if chunk.shape[0] <= window_size:
        sample_signal = pad(chunk, window_size)
    else:
        sample_signal = chunk
    chunk = custom_transform(sample_signal)
    return chunk


def predict_single(signal, depth_scale, input_shape, window_size):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Conv_Model(
        dropout=False,
        depth_scale=depth_scale,
        input_shape=input_shape,       # Dynamically adjusts for different input sizes
        device=device,
        modality="cough"
    )
    if torch.cuda.is_available():
        model.load_state_dict(torch.load('./cider/model.pt'))
    else:
        model.load_state_dict(torch.load('./cider/model.pt', map_location='cpu'))

    chunks = np.array_split(signal, int(np.ceil(signal.shape[0] / window_size)))
    chunks = [process_chunk(chunk, window_size, ) for chunk in chunks]
    with torch.no_grad():
        chunks = DataLoader(chunks)
        clip_predicts = []
        for audio in chunks:
            predicts_soft = model(audio)
            predicts_soft = torch.sigmoid(predicts_soft).cpu().numpy()
            predicts = np.where(predicts_soft > 0.5, 1, 0)
            clip_predicts.append((predicts, predicts_soft))

        positive = np.count_nonzero([c[0] for c in clip_predicts])
        votes = {'1': positive, '0': len(clip_predicts)-positive}
        # If its a tie, use logits
        if votes['1'] == votes['0']:
            logits = (
                sum([c[1] for c in clip_predicts if c[0].item() == 0]), # Negative
                sum([c[1] for c in clip_predicts if c[0].item() == 1]), # Positive
            )
            predicts = np.argmax(logits).reshape(1,1)
        else:
            predicts = np.array(int(max(votes.items(), key=lambda x: x[1])[0])).reshape(1,1)
        average_logits = [c[1][0][0] for c in clip_predicts]
        logit = np.mean(average_logits)
    return logit, predicts


def predict(input_audio):
    """
    predicts the output score for a batch of audio files
    :param input_audio: list of audio files to predict
    :return: np.array(n,1) of scores for input audio
    """
    wsz = 6
    sr = 24000
    window_size = wsz * sr
    nfft = 512
    n_mfcc = 40
    depth_scale = 2
    sr = 24000
    eval_type = "maj_vote"
    max_logit = 1
    feature_type = 'stft'

    if feature_type == 'stft':
        input_shape = (int(1024 * nfft / 2048) + 1, int(94 * wsz * sr / 48000))
    if feature_type == 'mfcc':
        input_shape = (n_mfcc, int(94 * wsz * sr / 48000))

    if isinstance(input_audio, list) or len(np.shape(input_audio)) > 1:
        # various files, need loop
        batch_size = len(input_audio)
        scores = np.zeros((batch_size, 1))
        for i, audio in enumerate(input_audio):
            score, label = predict_single(audio, depth_scale, input_shape, window_size)
            scores[i, 0] = score
    else:
        # just predict on single file
        scores, label = predict_single(input_audio, depth_scale, input_shape, window_size)
    return scores


if __name__ == '__main__':
    sr = 24000
    audios = []
    filename = 'BRdoMJMm_cough.flac'
    audio_path = f'/Users/anne/Documents/Uni/Robotics/Masterarbeit/MA_Code/DICOVA/DiCOVA_Train_Val_Data_Release/AUDIO/{filename}'
    audio1, sr = librosa.load(audio_path, sr=sr)
    filename = 'tejPPvGf_cough.flac'
    audio_path = f'/Users/anne/Documents/Uni/Robotics/Masterarbeit/MA_Code/DICOVA/DiCOVA_Train_Val_Data_Release/AUDIO/{filename}'
    audio2, sr = librosa.load(audio_path, sr=sr)
    audios.append(audio1)
    audios.append(audio2)
    print(predict(audio1))
    print(predict(audio2))
    print(predict(audios))

