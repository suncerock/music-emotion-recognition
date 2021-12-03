#! usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@suncerock
Generate spectrogram for 1000 songs dataset
The spectrogram are saved as a pickle file in a dict {music_id: int, spectrogram: np.ndarray}
'''

import os
import pickle
from tqdm import tqdm

import librosa
import numpy as np


def generate_spec(wav_dir, dest_file):
    data = dict()

    if os.path.exists(dest_file):
        while True:
            res = input("{} already exists!\nDo you want to overwrite it? [y/n] ".format(dest_file))
            if res == 'y':
                break
            elif res == 'n':
                return

    for wav_file in tqdm(os.listdir(wav_dir)):
        song_id = int(wav_file.split('.')[0])
        y, _ = librosa.load(os.path.join(wav_dir, wav_file), sr=44100)  # sampling rate fixed for 1000 songs dataset
        
        # force to have same length
        if len(y) < 45 * 44100:
            y = np.pad(y, pad_width=((0, 45 * 44100 - len(y))))
        else:
            y = y[:45 * 44100]

        melSpectro = librosa.feature.melspectrogram(y, sr=44100)
        logMelSpectro = librosa.amplitude_to_db(melSpectro, amin=1e-07)
        data[song_id] = logMelSpectro

    with open(dest_file, 'wb') as f:
        pickle.dump(data, f)

    return
    

def train_test_split(song_info_path, dest_file):
    f = open(song_info_path, 'r', encoding='utf-8')
    
    f_train = open(dest_file.split('.')[0] + '_train.txt', 'w')
    f_valid = open(dest_file.split('.')[0] + '_valid.txt', 'w')

    f.readline()
    for line in f.readlines():
        song_id = line.split(',')[0]
        print(line.strip().split(',')[-1])
        if line.strip().split(',')[-1] == '"development"':
            f_train.write(song_id + '\n')
        else:
            f_valid.write(song_id + '\n')
    
    f.close()
    f_train.close()
    f_valid.close()

    return


if __name__ == "__main__":
    wav_dir = "E:/Music Emotion Recognition/dataset/1000Songs/wav"
    dest_file = "E:/Music Emotion Recognition/dataset/1000Songs/spectrogram.pkl"
    song_info_path = "E:/Music Emotion Recognition/dataset/1000Songs/songs_info.csv"
    split_dest = "E:/Music Emotion Recognition/dataset/1000Songs/1000songs.txt"

    generate_spec(wav_dir, dest_file)
    train_test_split(song_info_path, split_dest)
