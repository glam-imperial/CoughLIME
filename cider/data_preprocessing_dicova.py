import os
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import pandas as pd
import librosa, librosa.display
import numpy as np
import matplotlib.pyplot as plt
from create_csv import create_csv
import random
import re
from itertools import cycle


class COVID_dataset(Dataset):
    '''
    Custom COVID dataset.
    '''
    def __init__(self, dset, fold_id,
                 eval_type='random',
                 transform=None,
                 window_size=1,
                 sample_rate=48000,
                 hop_length=512,
                 n_fft=2048,
                 masking=False,
                 pitch_shift=False, 
                 modality='breathcough',
                 kdd=False,
                 feature_type='stft',
                 n_mfcc=20,
                 onset_sample_method=False,
                 repetitive_padding = False,
                 dataset = "dicova"):

        if dataset == 'dicova':
            if modality == 'cough' and dset != 'test':
                path = '/vol/bitbucket/hgc19/DiCOVA_Train_Val_Data_Release'
                indexes = "File_name"
            elif modality == 'cough' and dset == 'test':
                path = '/vol/bitbucket/hgc19/DiCOVA_Track_1_Evaluation_Release'
                indexes = "File_name"
            elif modality == 'speechbreath':
                path = '/vol/bitbucket/hgc19/DiCOVA_Track_2_Release'
                indexes = "ID"
            else:
                raise 'This is not setup for cough breath for now'
            if dset != 'test' and not fold_id == 'all':
                file_path = os.path.join(path, 'LISTS', dset + '_fold_' + str(fold_id) + '.txt')
                df = pd.read_csv(file_path, header=None)
                train_fold = df[0].to_list()
                metadata = pd.read_csv(path + "/metadata.csv")
            elif dset == 'test' and modality == 'speechbreath':
                file_path = os.path.join(path, 'LISTS', 'eval_list.txt')
                df = pd.read_csv(file_path,header=None)
                train_fold = df[0].to_list()
                metadata = pd.read_csv(path + "/metadata.csv")
            elif fold_id == 'all' and dset != 'test' and not modality == 'speechbreath':
                metadata = pd.read_csv(path + "/metadata.csv")
                train_fold = metadata[indexes]
            elif fold_id == 'all' and dset != 'test' and modality == 'speechbreath':
                metadata = pd.read_csv(os.path.join(path, "metadata.csv"))
                test_fold = pd.read_csv('/vol/bitbucket/hgc19/DiCOVA_Track_2_Release/LISTS/eval_list.txt', header=None)
                test_fold = test_fold[0].to_list()
                train_fold = metadata[indexes]
                # remove test instances
                train_fold = [i for i in train_fold if i not in test_fold]

                self.overlap(train_fold, test_fold)


                print(train_fold)
            else:
                file_path = os.path.join(path, 'test_metadata.csv')
                metadata = pd.read_csv(file_path)
                train_fold = metadata[indexes].to_list()
        elif dataset == 'compare':
            path = '/vol/bitbucket/hgc19/COMPARE_data/' + modality
            indexes = "filename"
            metadata = pd.read_csv(path + "/lab/" + dset + ".csv")
            train_fold = metadata[indexes]
            if fold_id == 'all' and dset != 'test':
                metadata_val = pd.read_csv(path + "/lab/" + "devel" + ".csv")
                train_fold_val = metadata_val[indexes]
                metadata = pd.concat([metadata, metadata_val], ignore_index=True)
                train_fold = pd.concat([train_fold, train_fold_val], ignore_index=True)

        elif dataset == 'tos':
            path = '/vol/bitbucket/aa9120/Tos COVID-19/'
            indexes = "filename"
            metadata = pd.read_csv(path + dset + ".csv")
            train_fold = metadata[indexes]
        metadata = metadata.set_index(indexes)

        self.train_fold = train_fold
        if dset != 'test' or modality == 'speechbreath':
            self.metadata = metadata.loc[train_fold]
        else:
            self.metadata = metadata
        self.dset = dset  #train or val
        if dataset in ['dicova', 'tos']:
            self.root_dir = path + 'AUDIO/'
        else:
            self.root_dir = path + '/wav/'
        self.window_size = window_size * sample_rate
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.transform = transform
        self.eval_type = eval_type
        self.masking = masking
        self.pitch_shift = pitch_shift
        self.modality = modality
        self.feature_type = feature_type
        self.n_mfcc = n_mfcc
        self.onset_sample_method = onset_sample_method
        self.repetitive_padding = repetitive_padding
        self.path = path
        self.dataset = dataset
        # add extra training data from cambridge
        if kdd and modality == 'cough':
            self.kdd_data = pd.read_csv('/vol/bitbucket/hgc19/Dicova_Imperial/kddcoughdata.csv')
            self.train_fold.extend(self.kdd_data["path"].to_list())
            self.kdd_data = self.kdd_data.set_index("path")
    def __len__(self):
        return len(self.train_fold)

    def custom_transform(self, signal):
        """
        create log spectrograph of signal
        """
        if self.feature_type == 'stft':
            stft = librosa.stft(signal, n_fft=self.n_fft, hop_length=self.hop_length)
            spectrogram = np.abs(stft)
            features = librosa.amplitude_to_db(spectrogram)
        if self.feature_type == 'mfcc':
            features = librosa.feature.mfcc(signal, n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length)
        if self.masking:
            features = self.spec_augment(features)
        if self.transform:
            features = self.transform(features)
        return features

    def pad(self, signal):
        sample_signal = np.zeros((self.window_size,))
        sample_signal[:signal.shape[0],] = signal
        return sample_signal

    def pad_repetitive(self, signal):
        rpt_cnt = np.ceil(self.window_size / len(signal))
        signal = np.tile(signal, int(rpt_cnt))
        sample_signal = signal[:self.window_size, ]
        return sample_signal

    def __getitem__(self, index):

        # get path of chosen index
        audio_name = self.train_fold[index]
        #sorry v hacky but I am tired - here we load the data if it is a kdd instance
        #alican: HAHAHHAHAHAHHAHHAHAAH
        if 'vol/bitbucket' in audio_name:
            if self.kdd_data.loc[audio_name, 'Covid_status'] == 'p':
                label = 1
            else:
                label = 0
            chunks = self.load_process(audio_name)
            return chunks, label
        #print(audio_name)
        if self.dataset in ['dicova', 'tos']:
            covid_col_name = 'Covid_status'
            covid_pos_rep = 'p'
            covid_neg_rep = 'n'
        else:
            covid_col_name = 'label'
            covid_pos_rep = 'positive'
            covid_neg_rep = 'negative'


        if self.metadata.loc[audio_name, covid_col_name] == covid_pos_rep:
            label = 1
        elif self.metadata.loc[audio_name, covid_col_name] == covid_neg_rep:
            label = 0
        elif self.metadata.loc[audio_name, covid_col_name] == '?':
            label = 1000 # as has to be able to transformed into a torch tensor so None doesn't work
        else:
            raise f"Error, {self.metadata.loc[audio_name, covid_col_name]} is not a valid category"

        if self.dataset == 'dicova':
            if self.modality == 'cough' and self.dset != 'test':
                audio_path = '/vol/bitbucket/hgc19/DiCOVA_Train_Val_Data_Release/AUDIO/' + audio_name + '.flac'
                chunks = self.load_process(audio_path)
            elif self.modality == 'cough' and self.dset == 'test':
                chunks = self.load_process(os.path.join(self.path, 'AUDIO', str(audio_name) + '.flac'))
            elif self.modality == 'speechbreath':
                path_breath = '/vol/bitbucket/hgc19/DiCOVA_Track_2_Release/AUDIO/breathing-deep/' + audio_name + '_breathing-deep.flac'
                path_count = '/vol/bitbucket/hgc19/DiCOVA_Track_2_Release/AUDIO/counting-normal/' + audio_name + '_counting-normal.flac'
                path_vowel = '/vol/bitbucket/hgc19/DiCOVA_Track_2_Release/AUDIO/vowel-e/' + audio_name + '_vowel-e.flac'
                chunks1 = self.load_process(path_breath)
                chunks2 = self.load_process(path_count)
                chunks3 = self.load_process(path_vowel)

                if self.dset == 'train' or self.eval_type != 'maj_vote':
                    return torch.cat([chunks1, chunks2, chunks3], dim=0), label
                else:
                    # repeat the chunks so that all the info is considered of the longest recording
                    if len(chunks1) >= len(chunks2) and len(chunks1) >= len(chunks3):
                        zip_list = zip(chunks1, cycle(chunks2), cycle(chunks3))
                    elif len(chunks2) >= len(chunks1) and len(chunks2) >= len(chunks3):
                        zip_list = zip(cycle(chunks1), chunks2, cycle(chunks3))
                    elif len(chunks3) >= len(chunks1) and len(chunks3) >= len(chunks2):
                        zip_list = zip(cycle(chunks1), cycle(chunks2), chunks3)
                    else:
                        raise 'This should not be possible'
                    return [torch.cat([i, j, k], dim=0) for i, j, k in zip_list], label

            else:
                raise 'coughbreath is currently not set up'

        else:
            audio_path = self.root_dir + audio_name
            chunks = self.load_process(audio_path)

        # get path of a cough or breath sample which was provided by the same user
        # if a cough sample is provided need to get a breath sample and visa
        # versa
        return chunks, label



    def load_process(self, audio_path):
        # load the data
        #audio_path = os.fspath(audio_path)
        signal, sample_rate = librosa.load(audio_path, sr=self.sample_rate)
        # perform pitch shift:
        if self.pitch_shift:
            step = np.random.uniform(-6,6)
            signal = librosa.effects.pitch_shift(
                signal, sample_rate, step)

        # For train, sample random window size from audiofile
        if self.dset == 'train' or self.eval_type != 'maj_vote':
            # Apply padding if necessary. Else sampsle random window.
            if self.onset_sample_method:
                onsets = librosa.onset.onset_detect(signal, units='time')
                onsets = onsets * self.sample_rate
                onsets = [int(i) for i in onsets]
                if len(onsets)==0:
                    sample_signal = signal
                else:
                    rand_onset = random.choice(onsets)
                    left_ind = int(rand_onset - (self.window_size/2))
                    right_ind = int(rand_onset + (self.window_size/2))
                    if rand_onset - (self.window_size/2) < 0:
                        left_ind=0
                    if rand_onset + (self.window_size/2) >= signal.shape[0]:
                        right_ind=signal.shape[0]-1

                    sample_signal = signal[left_ind:right_ind]
                if sample_signal.shape[0] <= self.window_size:
                    if self.repetitive_padding:
                        sample_signal = self.pad_repetitive(sample_signal)
                    else:
                        sample_signal = self.pad(sample_signal)

            else:

                if signal.shape[0] <= self.window_size:
                    if self.repetitive_padding:
                        sample_signal = self.pad_repetitive(signal)
                    else:
                        sample_signal = self.pad(signal)
                else:
                    if self.eval_type == 'random':
                        rand_indx = np.random.randint(0, signal.shape[0] - self.window_size)
                    else:
                        rand_indx = 0
                    sample_signal = signal[rand_indx:rand_indx + self.window_size]

            # perform transformations
            sample_signal = self.custom_transform(sample_signal)

            return sample_signal
        # For eval/test, chunk audiofile into chunks of size wsz and
        # process and return all
        else:
            if self.onset_sample_method:
                chunks = []
                onsets = librosa.onset.onset_detect(signal, units='time')
                onsets = onsets * self.sample_rate
                onsets = [int(i) for i in onsets]
                if len(onsets)==0:
                    sample_signal = signal
                    if sample_signal.shape[0] <= self.window_size:
                        if self.repetitive_padding:
                            sample_signal = self.pad_repetitive(sample_signal)
                        else:
                            sample_signal = self.pad(sample_signal)
                    sample_signal = self.custom_transform(sample_signal)
                    chunks.append(sample_signal)
                else:
                    for onset in onsets:
                        left_ind = int(onset - (self.window_size / 2))
                        right_ind = int(onset + (self.window_size / 2))
                        if onset - (self.window_size / 2) < 0:
                            left_ind = 0
                        if onset + (self.window_size / 2) >= signal.shape[0]:
                            right_ind = signal.shape[0] - 1

                        sample_signal = signal[left_ind:right_ind]
                        if sample_signal.shape[0] <= self.window_size:
                            if self.repetitive_padding:
                                sample_signal = self.pad_repetitive(sample_signal)
                            else:
                                sample_signal = self.pad(signal)
                        sample_signal = self.custom_transform(sample_signal)
                        chunks.append(sample_signal)

            else:
                chunks = np.array_split(signal, int(np.ceil(signal.shape[0] / self.window_size)))
                def process_chunk(chunk):
                    if chunk.shape[0] <= self.window_size:
                        if self.repetitive_padding:
                            sample_signal = self.pad_repetitive(chunk)
                        else:
                            sample_signal = self.pad(chunk)
                    chunk =  self.custom_transform(sample_signal)
                    return chunk
                chunks = [process_chunk(chunk) for chunk in chunks]

            return chunks
    

    def spec_augment(self,
                     spec: np.ndarray,
                     num_mask=2,
                     freq_masking_max_percentage=0.15,
                     time_masking_max_percentage=0.3):

        spec = spec.copy()
        for i in range(num_mask):
            all_frames_num, all_freqs_num = spec.shape
            freq_percentage = random.uniform(0.0, freq_masking_max_percentage)

            num_freqs_to_mask = int(freq_percentage * all_freqs_num)
            f0 = np.random.uniform(low=0.0,
                                   high=all_freqs_num - num_freqs_to_mask)
            f0 = int(f0)
            spec[:, f0:f0 + num_freqs_to_mask] = 0

            time_percentage = random.uniform(0.0, time_masking_max_percentage)

            num_frames_to_mask = int(time_percentage * all_frames_num)
            t0 = np.random.uniform(low=0.0,
                                   high=all_frames_num - num_frames_to_mask)
            t0 = int(t0)
            spec[t0:t0 + num_frames_to_mask, :] = 0

        return spec



    def nth_repl(self, s, sub, repl, n):
        find = s.find(sub)
        # If find is not -1 we have found at least one match for the substring
        i = find != -1
        # loop util we find the nth or we find no match
        while find != -1 and i != n:
            # find + 1 means we start searching from after the last match
            find = s.find(sub, find + 1)
            i += 1
        # If i is equal to n we found nth match so replace
        if i == n:
            return s[:find] + repl + s[find + len(sub):]
        return s

    def overlap(self, list_1, list_2):
        '''
        sanity check that there is no leakage from test set into train/val
        '''
        overlap = [i for i in list_1 if i in list_2]

        if len(overlap) > 0:
            raise 'You have cross over between test and train!!!! Investigate'


if __name__ == "__main__":
    test_dataset = COVID_dataset('val', None)
    for i in tqdm(range(len(test_dataset))):
        sample, label = test_dataset[i]

        plt.figure()
        librosa.display.specshow(sample,
                                sr=24000,
                                hop_length=512)
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.colorbar(format="%+2.0f dB")
        plt.title("Spectrogram (dB)")
        path_to_save = 'figs/log_spectrogram'+str(i)+'.png'
        #plt.savefig(path_to_save)
        plt.show()
        plt.close()
        print(sample.shape)
