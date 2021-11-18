import os
import time

import librosa
import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data


RANDOM_SEED = 2019
DATASET = ['PMEmo2019', '1000Song']
DATASET_DIR = 'E:\Music Emotion Recognition\dataset'


class PMEmoDataset(Data.Dataset):
    def __init__(self, X_path, y_arousal, y_valence, music_id):
        self.X_path = X_path
        self.y_arousal = y_arousal
        self.y_valence = y_valence
        self.music_id = music_id

    def __getitem__(self, item):  # item 就是label表中的序号
        return self._audio2Mel(self.X_path[item]), self.y_arousal[item], self.y_valence[item], self.music_id[item]

    def __len__(self):
        return len(self.music_id)

    def _audio2Mel(self, audioFilePath, sampling_rate=22050, spec_duration=20):
        '''
        load melspec to for audio.
        '''
        # 生成结尾前10s的随机数，保证长度为spec_duration
        y, sr = librosa.load(audioFilePath, sr=sampling_rate)
        if len(y) < spec_duration * sr:
            y = np.pad(y, pad_width=((0, spec_duration * sr - len(y))))
            spec_offset = 0
        else:
            spec_offset = np.random.randint(0, len(y) - spec_duration * sr + 1)
        melSpectro = librosa.feature.melspectrogram(y[spec_offset: spec_offset + spec_duration * sr], sr=sampling_rate)
        logMelSpectro = librosa.amplitude_to_db(melSpectro, amin=1e-07)
        return logMelSpectro


def load_data(dataset_name='PMEmo2019', mode='static', normalize_label=True):
    tick = time.time()
    # load from file
    if not dataset_name in ['PMEmo2019', '1000Song']:
        raise KeyError("Unknown dataset {} (must be one of {})".format(dataset_name, DATASET))
    assert mode in ['static', 'dynamic'], "Unknown mode! (static or dynamic)"
    
    dataset = pd.read_csv(os.path.join(DATASET_DIR, '{}/annotations/{}_annotations.csv'.format(dataset_name, mode)))

    # train-test split
    songs = dataset['musicId'].unique()

    np.random.seed(RANDOM_SEED)
    np.random.shuffle(songs)
    train_songs = songs[:700]
    test_songs = songs[700:]

    iftestset = dataset['musicId'].apply(lambda x: x in test_songs)
    testset = dataset[iftestset]
    trainset = dataset[~iftestset]
    
    # obtain train and test data
    featureNames = dataset.columns[2:-2]

    y_train_arousal = np.array(trainset['Arousal(mean)'])
    y_train_valence = np.array(trainset['Valence(mean)'])

    y_test_arousal = np.array(testset['Arousal(mean)'])
    y_test_valence = np.array(testset['Valence(mean)'])

    music_id_train = np.array(trainset['musicId'])
    music_id_test = np.array(testset['musicId'])

    X_train_path = [os.path.join(DATASET_DIR, dataset_name, 'chorus', '{}.mp3'.format(music_id)) for music_id in music_id_train]
    X_test_path = [os.path.join(DATASET_DIR, dataset_name, 'chorus', '{}.mp3'.format(music_id)) for music_id in music_id_test]
    
    if normalize_label:
        y_train_arousal = (y_train_arousal - 0.5) * 2
        y_test_arousal = (y_test_arousal - 0.5) * 2

        y_train_valence = (y_train_valence - 0.5) * 2
        y_test_valence = (y_test_valence - 0.5) * 2

    tock = time.time()
    print("Data ({}) loaded in {:.2f}s: {:d} songs for train and {:d} songs for test".format(
        mode, tock - tick, len(set(music_id_train)), len(set(music_id_test))))
    return X_train_path, y_train_arousal, y_train_valence, X_test_path, y_test_arousal, y_test_valence, music_id_train, music_id_test


def build_online_dataloader(batch_size=16):
    X_train_path, y_train_arousal, y_train_valence, X_valid_path, y_valid_arousal, y_valid_valence, music_id_train, music_id_valid = load_data()

    train_dataset = PMEmoDataset(X_train_path, y_train_arousal, y_train_valence, music_id_train)
    valid_dataset = PMEmoDataset(X_valid_path, y_valid_arousal, y_valid_valence, music_id_valid)

    train_dataloader = Data.DataLoader(train_dataset, batch_size=batch_size)
    valid_dataloader = Data.DataLoader(valid_dataset, batch_size=batch_size)

    return train_dataloader, valid_dataloader


if __name__ == '__main__':
    train_dataloader, valid_dataloader = build_online_dataloader()
    for batch_idx, (X, y_arousal, y_valence, music_id) in enumerate(train_dataloader):
        print(X.shape, y_arousal.shape)
        print(music_id)
        