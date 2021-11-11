#! usr/bin/env python3
# -*- coding: utf-8 -*-

#########################################
# FIXME: NOT AVALIABLE YET! DON'T USE!! #
#########################################

'''
This features.py is used to extract audio features based on openSIMLE.
Require: openSMILE-2.2rc1
OpenSMILE only support audios in WAV format, 
so before using this script you could
transform MP3s into WAVs by transformat.sh.

@suncerock 2021/11/5
Modified by Yiwei Ding
openSMILE Python can be used instead of the original openSMILE
'''

__author__ = 'huizhang', 'Yiwei Ding'

import csv
import os
import shutil
from math import floor
from tqdm import tqdm

import numpy as np
import opensmile

def extract_all_wav_feature(wavdir, distfile):
    '''Extract 6373-dimension static features into one dist file.

    Args:
        wavdir: Path to audios in WAV format.
        distfile: Path of distfile.

    Returns:
        Distfile containing 6373-dimension static features of all the WAVs.
    '''
    
    SMILExtract = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.Functionals
    )

    if os.path.exists(distfile):
        os.remove(distfile)

    wav = [f for f in os.listdir(wavdir) if f[-4:] == ".wav"]
    f = open(distfile, 'w')
    for w in tqdm(wav):
        wavpath = os.path.join(wavdir,w)
        y = SMILExtract.process_file(wavpath)
        y = np.array(y).tolist()
        f.write(w + ',' + ','.join(map(str, y[0])) + '\n')
    f.close()

def extract_frame_feature(wavdir, distdir):
    '''Extract lld features in frame size: 60ms, step size: 10ms.

    Args:
        wavdir: Path to audios in WAV format.
        distdir: Path of distdir.
        opensmiledir: Path to opensimle project root.

    Returns:
        Distfiles containing lld features for each WAV.
    '''

    SMILExtract = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors
    )

    if os.path.exists(distdir):
        shutil.rmtree(distdir)
    os.mkdir(distdir)

    wav = [f for f in os.listdir(wavdir) if f[-4:] == ".wav"]
    for w in tqdm(wav):
        wavpath = os.path.join(wavdir,w)
        distfile = os.path.join(distdir,w[:-4]+".csv")
        
        f = open(distfile, 'w')
        y = SMILExtract.process_file(wavpath)
        y = np.array(y).tolist()
        y = [','.join(map(str, line))+'\n' for line in y]
        f.writelines(y)
        f.close()


def process_dynamic_feature(llddir, distdir, all_songs_distfile, delimiter=","):
    '''Obtain dynamic features in window size: 1s, shift size: 0.5s.

    Args:
        llddir: Path to lld feature files.
        distdir: Path of distdir.
        all_songs_distfile: Path of distfile.
        delimiter: csv delimiter in lld feature files, default=';'.

    Returns:
        Distfiles containing 260-dimension dynamic features all WAVs.
    '''

    if os.path.exists(distdir):
        shutil.rmtree(distdir)
    os.mkdir(distdir)

    window = 1
    overlap = 0.5

    llds = [f for f in os.listdir(llddir) if f[-4:] == ".csv"]
    all_dynamic_features = []
    all_musicId = []

    for lld in llds:
        musicId = []
        lldpath = os.path.join(llddir,lld)
        single_song_distfile = os.path.join(distdir,lld)

        dynamic_features = _compute_feature_with_window_and_overlap(lldpath, window, overlap, delimiter)
        for i in range(len(dynamic_features)):
            musicId.append(lld[:-4])
        _write_features_to_csv(musicId, dynamic_features, single_song_distfile)

        all_musicId += musicId
        all_dynamic_features += dynamic_features

    _write_features_to_csv(all_musicId, all_dynamic_features, all_songs_distfile)

def _compute_feature_with_window_and_overlap(lldpath, window, overlap, delimiter):
    '''Compute the mean and std for frame-wise features in window size: 1s, shift size: 0.5s.'''

    fs = 0.01
    num_in_new_frame = floor(overlap/fs)
    num_in_window = floor(window/fs)

    # load the features from disk
    all_frame = []
    with open(lldpath) as f:
        reader = csv.reader(f,delimiter=delimiter)
        for row in reader:
            frame_feature = []
            for i in range(len(row)-1): #旧的frametime不用记录
                frame_feature.append(float(row[i+1]))
            all_frame.append(frame_feature)

    # compute new number of frames
    new_num_of_frame = floor(len(all_frame)/num_in_new_frame)
    all_new_frame = []

    # compute mean and std in each window as the feature corresponding to the frame. 
    for i in range(new_num_of_frame):
        start_index = num_in_new_frame * i
        new_frame_array = np.array(all_frame[start_index:start_index+num_in_window])

        mean_llds = np.mean(new_frame_array,axis=0)
        std_llds = np.std(new_frame_array,axis=0)
        new_frametime = i * overlap

        new_frame = [new_frametime] + mean_llds.tolist() + std_llds.tolist()
        all_new_frame.append(new_frame)

    return all_new_frame

def _write_features_to_csv(musicIds, contents, distfile):
    '''Write all the features into one file, and add the last column as the annotation value'''
    
    with open(distfile,"w") as newfile:
        writer = csv.writer(newfile)
        for i in range(len(contents)):
            writer.writerow([musicIds[i]] + contents[i])


if __name__ == "__main__":
    wavdir ="E:\Music Emotion Recognition\dataset\PMemo2019 test\chorus\wav"

    static_distfile = "E:\Music Emotion Recognition\dataset\PMemo2019 test\static_features.csv"
    lld_distdir = "E:\Music Emotion Recognition\dataset\PMemo2019 test\IS13features_lld"
    dynamic_distdir = "E:\Music Emotion Recognition\dataset\PMemo2019 test\dynamic_features"
    all_dynamic_distfile = "E:\Music Emotion Recognition\dataset\PMemo2019 test\dynamic_features.csv"

    delimiter = ","

    # extract_all_wav_feature(wavdir,static_distfile)
    # extract_frame_feature(wavdir,lld_distdir)
    process_dynamic_feature(lld_distdir,dynamic_distdir,all_dynamic_distfile,delimiter)