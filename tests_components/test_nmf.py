import librosa
import librosa.display
import numpy as np
import scipy
import matplotlib.pyplot as plt
import sklearn
import soundfile


def test_nmf():
    filename = 'AtACyGlV_cough.flac'
    audio_path = f'/Users/anne/Documents/Uni/Robotics/Masterarbeit/MA_Code/DICOVA/DiCOVA_Train_Val_Data_Release/AUDIO/{filename}'
    x, sample_rate = librosa.load(audio_path)
    num_components = 6
    S = librosa.stft(x)
    X, X_phase = librosa.magphase(S)
    W, H = librosa.decompose.decompose(X, n_components=num_components, sort=True)
    print(W.shape)  # spectral
    print(H.shape)  # temporal
    for n in range(num_components):
        # Re-create the STFT of a single NMF component.
        Y = np.outer(W[:,n], H[n])*X_phase

        # Transform the STFT into the time domain.
        y = librosa.istft(Y)

        path_write = f'./nmf/{filename[:-11]}_comp_{n}.wav'
        soundfile.write(path_write, y, sample_rate)

    # test, needed for decomp, watch out to check for special case that only one component is set to true!
    mask = np.array([False, True, False, True, False, True])
    print(mask)
    Y_test = np.dot(W[:, mask], H[mask, :]) * X_phase
    y_test = librosa.istft(Y_test)
    path_write = f'./nmf/{filename[:-11]}_3_comp_test.wav'
    soundfile.write(path_write, y_test, sample_rate)

    # Re-create the STFT from all NMF components.
    Y = np.dot(W, H)*X_phase

    # Transform the STFT into the time domain.
    reconstructed_signal = librosa.istft(Y, length=len(x))
    soundfile.write(f'./nmf/{filename[:-11]}_reconstrcuted.wav', reconstructed_signal, sample_rate)

    residual = x - reconstructed_signal
    residual[0] = 1 # hack to prevent automatic gain scaling
    soundfile.write(f'./nmf/{filename[:-11]}_residual.wav', residual, sample_rate)


if __name__ == "__main__":
    test_nmf()
