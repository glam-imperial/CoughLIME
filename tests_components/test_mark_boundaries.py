import matplotlib.pyplot as plt
import spectral_segmentation
import numpy as np
import librosa
from skimage.segmentation import mark_boundaries
from PIL import Image
import pylab


if __name__ == "__main__":
    # load an audio array and get its segmentation
    # make an exemplary mask and display the outcome
    filename = 'iyWdhFuN_cough.flac'
    audio_path = f'/Users/anne/Documents/Uni/Robotics/Masterarbeit/MA_Code/DICOVA/DiCOVA_Train_Val_Data_Release/AUDIO/{filename}'
    audio, sample_rate = librosa.load(audio_path)
    segmentation = spectral_segmentation.SpectralSegmentation(audio, sample_rate, 7)
    indices = [0, 1, 2, 3, 4, 5, 6]
    spectrogram = segmentation.return_spectrogram_indices(indices)
    spec_db = librosa.power_to_db(spectrogram, ref=np.max)
    mask = segmentation.return_mask_boundaries([0, 1, 2, 3, 4, 5, 6], [])
    marked = mark_boundaries(spec_db, mask)
    plt.imshow(marked[:, :, 2], origin="lower", cmap=plt.get_cmap("magma"))
    plt.colorbar(format='%+2.0f dB')
    image_array = np.ones(np.shape(mask) + (4,))
    mask_negative = np.zeros(np.shape(mask))
    mask_negative[np.where(mask == 0)] = 1
    mask_negative_green = np.ones(np.shape(mask))
    mask_negative_green[np.where(mask == -1)] = 0
    mask_negative_red = np.ones(np.shape(mask))
    mask_negative_red[np.where(mask == 1)] = 0
    image_array[:, :, 0] = mask_negative_red #0 for green, 1 for red
    image_array[:, :, 1] = mask_negative_green
    image_array[:, :, 2] = mask_negative
    image_array[:, :, 3] = np.abs(mask)
    # plt.imshow(image_array, origin="lower", interpolation="nearest", alpha=0.5)  # set opacity with alpha value
    plt.savefig("./spec.png")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    ax = plt.gca()
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    plt.title("Spectral Decomposition into 7 Components")
    plt.savefig("./test_spectral.png")
    plt.show()
    print("done")
