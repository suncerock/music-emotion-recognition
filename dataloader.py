import os

import numpy as np
import pandas as pd
np.random.seed(2019)

DATASET = ['PMEmo2019', '1000Song']

def load_dynamic_data(dataset='PMEmo2019', normalize_feature=True, normalize_label=True):
    # load from file
    if not dataset in ['PMEmo2019', '1000Song']:
        raise KeyError("Unknown dataset {} (must be one of {})".format(dataset, DATASET))

    features = pd.read_csv('./dataset/{}/features/dynamic_features.csv'.format(dataset))
    annotations = pd.read_csv('./dataset/{}/annotations/dynamic_annotations.csv'.format(dataset))
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
        arousal_mean = dataset['Arousal(mean)'].mean()
        arousal_std = dataset['Arousal(mean)'].std()
        y_train_arousal = (y_train_arousal - arousal_mean) / arousal_std
        y_test_arousal = (y_test_arousal - arousal_mean) / arousal_std

        valence_mean = dataset['Valence(mean)'].mean()
        valence_std = dataset['Valence(mean)'].std()
        y_train_valence = (y_train_valence - valence_mean) / valence_std
        y_test_valence = (y_test_valence - valence_mean) / valence_std

    return X_train, y_train_arousal, y_train_valence, X_test, y_test_arousal, y_test_valence, music_id_train, music_id_test


if __name__ == '__main__':
    X_train, y_train_arousal, y_train_valence, X_test, y_test_arousal, y_test_valence, music_id_train, music_id_test = load_dynamic_data()
    print("X_train: ", X_train.shape)
    print("y_train_arousal: ", y_train_arousal.shape)
    print("y_train_valence: ", y_train_valence.shape)
    print("X_test: ", X_test.shape)
    print("y_test_arousal: ", y_test_arousal.shape)
    print("y_test_valence: ", y_test_valence.shape)
    print(music_id_train)
    print(music_id_test)
    