#from lime import lime_image
#import cv2
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile
import time


def predict():
    print("Just for test")
    return None


def test_lime_image():
    im = cv2.imread("husky.jpg")
    explainer = lime_image.LimeImageExplainer()
    explainer.explain_instance(im, predict)


def test_audio():
    filename = 'AtACyGlV_cough.flac'
    audio_path = f'/Users/anne/Documents/Uni/Robotics/Masterarbeit/MA_Code/DICOVA/DiCOVA_Train_Val_Data_Release/AUDIO/{filename}'
    fs = librosa.get_samplerate(audio_path)
    s, _ = librosa.load(audio_path, sr=fs)


def test_spectral_decomposition():
    start = time.time()
    print("Starting clock")
    filename = 'AtACyGlV_cough.flac'
    audio_path = f'/Users/anne/Documents/Uni/Robotics/Masterarbeit/MA_Code/DICOVA/DiCOVA_Train_Val_Data_Release/AUDIO/{filename}'
    audio, sample_rate = librosa.load(audio_path)
    n_mels = 128
    spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=n_mels)
    librosa.display.specshow(spec, sr=sample_rate, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.show()
    print(np.shape(spec))
    for i in range(8):
        spectrogram = np.zeros(np.shape(spec))
        spectrogram[i*16:(i+1)*16, :] = spec[i*16:(i+1)*16, :]
        reconstructed_audio = librosa.feature.inverse.mel_to_audio(spectrogram, sr=sample_rate)
        #path_write = f"./spectral_tests/{filename[:-5]}_{n_mels}_{i}.wav"
        #soundfile.write(path_write, reconstructed_audio, sample_rate)
        spec_db = librosa.power_to_db(spectrogram, ref=np.max)
        librosa.display.specshow(spec_db, sr=sample_rate, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.show()
    print(spec)
    end = time.time()
    print("elapsed time:", end - start)


def compare_inverses():
    filename = 'AtACyGlV_cough.flac'
    audio_path = f'/Users/anne/Documents/Uni/Robotics/Masterarbeit/MA_Code/DICOVA/DiCOVA_Train_Val_Data_Release/AUDIO/{filename}'
    audio, sample_rate = librosa.load(audio_path)
    # spectrogram
    start = time.time()
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
    end = time.time()
    print("time spec extraction", end - start)
    start = time.time()
    # reconstructed_audio_mel = librosa.feature.inverse.mel_to_audio(spectrogram, n_iter=8 , sr=sample_rate)
    end = time.time()
    print("Elapsed time spectrogram reconstruction: ", end - start)
    start = time.time()
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate)
    deltas = librosa.feature.delta(mfcc)
    print(np.shape(mfcc))
    features = np.concatenate((mfcc, deltas), axis=0)

    ddeltas = librosa.feature.delta(mfcc, order=2)
    features = np.concatenate((features, ddeltas), axis=0)

    print(np.shape(features))
    librosa.display.specshow(mfcc)
    plt.show()
    librosa.display.specshow(features)
    plt.show()
    end = time.time()
    print("time mfcc extraction", end - start)
    start = time.time()
    reconstructed_audio_mfcc = librosa.feature.inverse.mfcc_to_audio(mfcc=mfcc, sr=sample_rate)
    end = time.time()
    print("Elapsed time mfcc reconstruction: ", end - start)
    print('done :) ')


def test_mfcc():
    pass


if __name__ == "__main__":
    # test_lime_image()
    # test_audio()
    # test_spectral_decomposition()
    test_mfcc()
    compare_inverses()
