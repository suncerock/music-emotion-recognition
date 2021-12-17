"""
Dataset for training
Support dataset: 
    - PMEmo: spectrogram, static annotations
"""

import os
import time
import pickle

import librosa
import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data


class PMEMODataset(Data.Dataset):
    def __init__(self, X, y_arousal, y_valence, music_id):
        self.X = X
        self.y_arousal = y_arousal
        self.y_valence = y_valence
        self.music_id = music_id

    def __getitem__(self, index):
        return dict(X=self.X[index], y_arousal=self.y_arousal[index],  y_valence=self.y_valence[index])

    def __len__(self):
        return len(self.music_id)
        

def _audio2Mel(audioFilePath, sampling_rate=22050, min_duration=20):
    if os.path.exists(audioFilePath.replace('mp3', 'pkl').replace('chorus', 'spectrogram')):
        with open(audioFilePath.replace('mp3', 'pkl').replace('chorus', 'spectrogram'), 'rb') as f:
            logMelSpectro = pickle.load(f)
        return logMelSpectro

    y, sr = librosa.load(audioFilePath, sr=sampling_rate)

    if len(y) < min_duration * sr:
        y = np.pad(y, pad_width=((0, min_duration * sr - len(y))))

    melSpectro = librosa.feature.melspectrogram(y, sr=sampling_rate)
    logMelSpectro = librosa.amplitude_to_db(melSpectro, amin=1e-07)

    with open(audioFilePath.replace('mp3', 'pkl').replace('chorus', 'spectrogram'), 'wb') as f:
        pickle.dump(logMelSpectro, f)

    return logMelSpectro


def load_PMEMO_data(data_dir, normalize_label=True, random_seed=2019):
    dataset = pd.read_csv(os.path.join(data_dir, 'annotations/static_annotations.csv'))

    # train-test split
    songs = dataset['musicId'].unique()

    np.random.seed(random_seed)
    np.random.shuffle(songs)
    train_songs = songs[:700]
    test_songs = songs[700:]

    iftestset = dataset['musicId'].apply(lambda x: x in test_songs)
    testset = dataset[iftestset]
    trainset = dataset[~iftestset]

    # obtain train and test data
    y_train_arousal = np.array(trainset['Arousal(mean)'], dtype=np.float32)
    y_train_valence = np.array(trainset['Valence(mean)'], dtype=np.float32)

    y_test_arousal = np.array(testset['Arousal(mean)'], dtype=np.float32)
    y_test_valence = np.array(testset['Valence(mean)'], dtype=np.float32)

    music_id_train = np.array(trainset['musicId'])
    music_id_test = np.array(testset['musicId'])

    X_train = [_audio2Mel(os.path.join(data_dir, 'chorus/{}.mp3'.format(music_id))) for music_id in music_id_train]
    X_test = [_audio2Mel(os.path.join(data_dir, 'chorus/{}.mp3'.format(music_id))) for music_id in music_id_test]

    if normalize_label:
        y_train_arousal = (y_train_arousal - 0.5) * 2
        y_test_arousal = (y_test_arousal - 0.5) * 2

        y_train_valence = (y_train_valence - 0.5) * 2
        y_test_valence = (y_test_valence - 0.5) * 2

    return X_train, y_train_arousal, y_train_valence, X_test, y_test_arousal, y_test_valence, music_id_train, music_id_test

def build_PMEMO_dataloader(data_dir,
                           length=800,
                           normalize_label=True,
                           random_seed=2019,
                           batch_size=64,
                           shuffle=False):
    X_train, y_train_arousal, y_train_valence, X_valid, y_valid_arousal, y_valid_valence, id_train, id_valid = load_PMEMO_data(
        data_dir, normalize_label=normalize_label, random_seed=random_seed)
    
    def collate_fn(data, segment=True):
        batch_X, batch_y_arousal, batch_y_valence = [], [], []

        for sample in data:
            if segment:
                offset = np.random.randint(0, sample['X'].shape[1] - length + 1)
                batch_X.append(sample['X'][:, offset: offset + length])
            else:
                batch_X.append(sample['X'])
            batch_y_arousal.append(sample['y_arousal'])
            batch_y_valence.append(sample['y_valence'])

        batch_X = np.array(batch_X, dtype=np.float32)
        batch_y_arousal = np.array(batch_y_arousal, dtype=np.float32)
        batch_y_valence = np.array(batch_y_valence, dtype=np.float32)

        return torch.from_numpy(batch_X), torch.from_numpy(batch_y_arousal), torch.from_numpy(batch_y_valence)
    
    train_collate_fn = lambda x:collate_fn(x, segment=True)
    valid_collate_fn = lambda x:collate_fn(x, segment=False)

    train_dataset = PMEMODataset(X_train, y_train_arousal, y_train_valence, id_train)
    valid_dataset = PMEMODataset(X_valid, y_valid_arousal, y_valid_valence, id_valid)

    train_dataloader = Data.DataLoader(train_dataset, batch_size=batch_size, collate_fn=train_collate_fn, shuffle=shuffle)
    valid_dataloader = Data.DataLoader(valid_dataset, batch_size=1, collate_fn=valid_collate_fn)

    return train_dataloader, valid_dataloader


if __name__ == '__main__':
    data_dir = "E:/Music Emotion Recognition/dataset/PMEmo2019/"

    train_dataloader, valid_dataloader = build_PMEMO_dataloader(data_dir=data_dir,
                                                                length=800,
                                                                normalize_label=True,
                                                                random_seed=2019,
                                                                batch_size=64,
                                                                shuffle=False)

    for X, y_arousal, y_valence in train_dataloader:
        print(X.shape, y_arousal.shape, y_valence.shape)
    for X, y_arousal, y_valence in valid_dataloader:
        print(X.shape, y_arousal.shape, y_valence.shape)