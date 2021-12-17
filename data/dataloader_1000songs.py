"""
Dataset for training
Support dataset: 
    - 1000 Songs for Emotional Analysis of Music: spectrogram, static annotations
"""

import os
import time
import pickle

import numpy as np
import torch
import torch.utils.data as Data


class Songs1000Dataset(Data.Dataset):
    def __init__(self, X, y_arousal, y_valence, music_id):
        self.X = X
        self.y_arousal = y_arousal
        self.y_valence = y_valence
        self.music_id = music_id

    def __getitem__(self, index):
        music_id = self.music_id[index]
        return dict(X=self.X[music_id], y_arousal=self.y_arousal[music_id],  y_valence=self.y_valence[music_id])

    def __len__(self):
        return len(self.music_id)
        

def load_1000songs_data(data_path, train_list, valid_list, annotations=None, normalize_label=True):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    with open(train_list) as f:
        id_train = [int(x.strip()) for x in f.readlines()]
    with open(valid_list) as f:
        id_valid = [int(x.strip()) for x in f.readlines()]

    X_train = {x:data[x] for x in id_train}
    X_valid = {x:data[x] for x in id_valid}

    y_arousal = {}
    y_valence = {}


    with open(annotations) as f:
        f.readline()
        for line in f.readlines():
            song_id, arousal, _, valence, _ = line.split(',')
            y_arousal[int(song_id)] = float(arousal) if not normalize_label else (float(arousal) - 5) / 4
            y_valence[int(song_id)] = float(valence) if not normalize_label else (float(valence) - 5) / 4

    y_train_arousal = {x:y_arousal[x] for x in id_train}
    y_train_valence = {x:y_valence[x] for x in id_train}
    y_valid_arousal = {x:y_arousal[x] for x in id_valid}
    y_valid_valence = {x:y_valence[x] for x in id_valid}

    return X_train, y_train_arousal, y_train_valence, X_valid, y_valid_arousal, y_valid_valence, id_train, id_valid


def build_1000songs_dataloader(data_path,
                               train_list,
                               valid_list,
                               annotations=None,
                               normalize_label=True,
                               batch_size=64,
                               shuffle=False):
    X_train, y_train_arousal, y_train_valence, X_valid, y_valid_arousal, y_valid_valence, id_train, id_valid = load_1000songs_data(
        data_path, train_list, valid_list, annotations=annotations, normalize_label=normalize_label)
    
    def collate_fn(data):
        batch_X, batch_y_arousal, batch_y_valence = [], [], []
        for sample in data:
            batch_X.append(sample['X'])
            batch_y_arousal.append(sample['y_arousal'])
            batch_y_valence.append(sample['y_valence'])

        batch_X = np.array(batch_X, dtype=np.float32)
        batch_y_arousal = np.array(batch_y_arousal, dtype=np.float32)
        batch_y_valence = np.array(batch_y_valence, dtype=np.float32)

        return torch.from_numpy(batch_X), torch.from_numpy(batch_y_arousal), torch.from_numpy(batch_y_valence)

    train_dataset = Songs1000Dataset(X_train, y_train_arousal, y_train_valence, id_train)
    valid_dataset = Songs1000Dataset(X_valid, y_valid_arousal, y_valid_valence, id_valid)

    train_dataloader = Data.DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)
    valid_dataloader = Data.DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate_fn)

    return train_dataloader, valid_dataloader


if __name__ == '__main__':
    data_path = "E:/Music Emotion Recognition/dataset/1000Songs/spectrogram.pkl"
    train_list = "E:/Music Emotion Recognition/dataset/1000Songs/1000songs_train.txt"
    valid_list = "E:/Music Emotion Recognition/dataset/1000Songs/1000songs_valid.txt"
    annotations = "E:/Music Emotion Recognition/dataset/1000Songs/annotations/static_annotations.csv"

    train_dataloader, valid_dataloader = build_1000songs_dataloader(data_path=data_path,
                                                                    train_list=train_list,
                                                                    valid_list=valid_list,
                                                                    annotations=annotations,
                                                                    normalize_label=True,
                                                                    batch_size=64,
                                                                    shuffle=False)

    for X, y_arousal, y_valence in train_dataloader:
        print(X.shape, y_arousal, y_valence)