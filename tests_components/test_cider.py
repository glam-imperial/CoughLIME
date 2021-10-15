import librosa
import sys
sys.path.append('../cider')
import cider_dicova_test as cider
import os


def test_single_audio():
    filename = 'AtACyGlV_cough.flac'
    audio_path = f'/Users/anne/Documents/Uni/Robotics/Masterarbeit/MA_Code/DICOVA/DiCOVA_Train_Val_Data_Release/AUDIO/{filename}'
    audio, sample_rate = librosa.load(audio_path)
    return cider.predict_audio(audio)


if __name__ == "__main__":
    print(os.getcwd())
    print(test_single_audio())
