import os

import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
np.random.seed(2019)

DATASET = ['PMEmo2019', '1000Song']
DATASET_DIR = 'E:\Music Emotion Recognition\dataset'

def load_dynamic_data(dataset='PMEmo2019', normalize_feature=True, normalize_label=True):
    # load from file
    if not dataset in ['PMEmo2019', '1000Song']:
        raise KeyError("Unknown dataset {} (must be one of {})".format(dataset, DATASET))

    features = pd.read_csv(os.path.join(DATASET_DIR, '{}/features/dynamic_features.csv'.format(dataset)))
    annotations = pd.read_csv(os.path.join(DATASET_DIR, '{}/annotations/dynamic_annotations.csv'.format(dataset)))
    dataset = pd.merge(features, annotations, on=['musicId', 'frameTime'])

    # train-test split
    songs = dataset['musicId'].unique()
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

    return X_train, y_train_arousal, y_train_valence, X_test, y_test_arousal, y_test_valence, music_id_train, music_id_test


class OpenSmileDynamicDataset(Data.Dataset):
    def __init__(self, X, y_arousal, y_valence, music_id):
        self.X = X
        self.y_arousal = y_arousal
        self.y_valence = y_valence
        self.music_id = music_id

    def __getitem__(self, index):
        return dict(X=self.X[index], y_arousal=self.y_arousal[index], y_valence=self.y_valence[index], music_id=self.music_id[index])

    def __len__(self):
        return len(self.music_id)


def build_opensmile_dynamic_dataloaer(batch_size=64):
    X_train, y_train_arousal, y_train_valence, X_valid, y_valid_arousal, y_valid_valence, music_id_train, music_id_valid = load_dynamic_data()
    
    def collate_fn(data):
        X = np.array([sample['X'] for sample in data], dtype=np.float32)
        y_arousal = np.array([sample['y_arousal'] for sample in data], dtype=np.float32)
        y_valence = np.array([sample['y_valence'] for sample in data], dtype=np.float32)
        music_id = np.array([sample['music_id'] for sample in data], dtype=np.int32)
        return torch.from_numpy(X), torch.from_numpy(y_arousal), torch.from_numpy(y_valence), music_id

    train_dataset = OpenSmileDynamicDataset(X_train, y_train_arousal, y_train_valence, music_id_train)
    valid_dataset = OpenSmileDynamicDataset(X_valid, y_valid_arousal, y_valid_valence, music_id_valid)

    train_dataloader = Data.DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)
    valid_dataloader = Data.DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate_fn)

    return train_dataloader, valid_dataloader


if __name__ == '__main__':
    train_dataloader, valid_dataloader = build_opensmile_dynamic_dataloaer(batch_size=8)
    print(len(train_dataloader), len(valid_dataloader))

    for batch_idx, (X, y_arousal, y_valence, music_id) in enumerate(train_dataloader):
        print(X.shape)
        print(y_arousal.shape)
        print(y_valence.shape)
        print(music_id)
        break

    for batch_idx, (X, y_arousal, y_valence, music_id) in enumerate(valid_dataloader):
        print(X.shape)
        print(y_arousal.shape)
        print(y_valence.shape)
        print(music_id)
        break
    