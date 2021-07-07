import opensmile
import librosa
import soundfile
from pathlib import Path


def test_feature_extraction():
    # convert to wav
    filename = "JgEUgPGO_cough.flac"
    file_path = f"/Users/anne/Documents/Uni/Robotics/Masterarbeit/MA_Code/DICOVA/DiCOVA_Train_Val_Data_Release/AUDIO/{filename}"
    audio, sr = librosa.load(file_path)
    new_directory = "./converted_files"
    Path(new_directory).mkdir(parents=True, exist_ok=True)
    path_converted_file = f"{new_directory}/{filename[:-5]}.wav"
    soundfile.write(path_converted_file, audio, sr)

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    y = smile.process_file(path_converted_file)
    print(y)

if __name__ == "__main__":
    test_feature_extraction()
