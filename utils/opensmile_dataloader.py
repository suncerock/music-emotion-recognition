import os
import time

import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data


RANDOM_SEED = 2019
DATASET = ['PMEmo2019', '1000Song']
DATASET_DIR = 'E:\Music Emotion Recognition\dataset'


def load_data(dataset='PMEmo2019', mode='static', normalize_feature=True, normalize_label=True):
    tick = time.time()
    # load from file
    if not dataset in ['PMEmo2019', '1000Song']:
        raise KeyError("Unknown dataset {} (must be one of {})".format(dataset, DATASET))
    assert mode in ['static', 'dynamic'], "Unknown mode! (static or dynamic)"
    
    features = pd.read_csv(os.path.join(DATASET_DIR, '{}/features/{}_features.csv'.format(dataset, mode)))
    annotations = pd.read_csv(os.path.join(DATASET_DIR, '{}/annotations/{}_annotations.csv'.format(dataset, mode)))
    dataset = pd.merge(features, annotations, on=['musicId']) if mode == 'static' else pd.merge(features, annotations, on=['musicId', 'frameTime'])

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

    X_train = np.array(trainset[featureNames])
    y_train_arousal = np.array(trainset['Arousal(mean)'])
    y_train_valence = np.array(trainset['Valence(mean)'])

    X_test = np.array(testset[featureNames])
    y_test_arousal = np.array(testset['Arousal(mean)'])
    y_test_valence = np.array(testset['Valence(mean)'])

    music_id_train = np.array(trainset['musicId'])
    music_id_test = np.array(testset['musicId'])

    if normalize_feature:
        feature_mean = np.array(dataset[featureNames].mean())
        feature_std = np.array(dataset[featureNames].std())
        X_train = (X_train - feature_mean) / feature_std
        X_test = (X_test - feature_mean) / feature_std
    
    if normalize_label:
        y_train_arousal = (y_train_arousal - 0.5) * 2
        y_test_arousal = (y_test_arousal - 0.5) * 2

        y_train_valence = (y_train_valence - 0.5) * 2
        y_test_valence = (y_test_valence - 0.5) * 2

    tock = time.time()
    print("Data ({}) loaded in {:.2f}s: {:d} songs for train and {:d} songs for test".format(
        mode, tock - tick, len(set(music_id_train)), len(set(music_id_test))))
    return X_train, y_train_arousal, y_train_valence, X_test, y_test_arousal, y_test_valence, music_id_train, music_id_test


class OpenSmileDataset(Data.Dataset):
    def __init__(self, X, y_arousal, y_valence, music_id):
        self.X = X
        self.y_arousal = y_arousal
        self.y_valence = y_valence
        self.music_id = music_id

    def __getitem__(self, index):
        return dict(X=self.X[index],
                    y_arousal=self.y_arousal[index],
                    y_valence=self.y_valence[index],
                    music_id=self.music_id[index])

    def __len__(self):
        return len(self.music_id)


def build_opensmile_dataloader(batch_size=64, chunk_size=40, shuffle=False):
    X_train_song, X_valid_song = [], []
    X_train, _, _, X_valid, _, _, music_id_train_seg, music_id_valid_seg = load_data(mode='dynamic')
    _, y_train_arousal, y_train_valence, _, y_valid_arousal, y_valid_valence, music_id_train, music_id_valid = load_data(mode='static')
    for music in music_id_train:
        X_train_song.append(X_train[music_id_train_seg == music])
    for music in music_id_valid:
        X_valid_song.append(X_valid[music_id_valid_seg == music])

    def collate_fn(data):
        batch_X, batch_y_arousal, batch_y_valence = [], [], []
        for sample in data:
            X, y_arousal, y_valence, music_id = sample['X'], sample['y_arousal'], sample['y_valence'], sample['music_id']
            if len(X) < chunk_size:
                pad = chunk_size - len(X)
                X = np.pad(X, pad_width=((0, pad), (0, 0)))
            else:
                start = np.random.randint(low=0, high=len(X) - chunk_size + 1)
                X = X[start: start + chunk_size]
            batch_X.append(X)
            batch_y_arousal.append(y_arousal)
            batch_y_valence.append(y_valence)

        batch_X = np.array(batch_X, dtype=np.float32)
        batch_y_arousal = np.array(batch_y_arousal, dtype=np.float32)
        batch_y_valence = np.array(batch_y_valence, dtype=np.float32)

        return torch.from_numpy(batch_X), torch.from_numpy(batch_y_arousal), torch.from_numpy(batch_y_valence)

    train_dataset = OpenSmileDataset(X_train_song, y_train_arousal, y_train_valence, music_id_train)
    valid_dataset = OpenSmileDataset(X_train_song, y_valid_arousal, y_valid_valence, music_id_valid)

    train_dataloader = Data.DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)
    valid_dataloader = Data.DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate_fn)

    return train_dataloader, valid_dataloader


if __name__ == '__main__':
    train_dataloader, valid_dataloader = build_opensmile_dataloader()
    for batch_idx, (X, y_arousal, y_valence) in enumerate(train_dataloader):
        print(X.shape)
        print(y_arousal.shape)
        print(y_valence.shape)
    