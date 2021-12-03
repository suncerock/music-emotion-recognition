# Dataset Description

## 1000 Songs for Emotional Analysis of Music

Dataset homepage: [Emotion in Music Database (1000 songs)](https://cvml.unige.ch/databases/emoMusic/)

- Fixed length and sample rate: 45s, 44100Hz
- Static valence-arousal annotations for each song, 1~9
- Dynamic valence-arousal annotations for each song, 2Hz, 15~45s, -1~+1
- 744 songs are already split into training set (619 songs) and validation set (125 songs)

File structure
```
|-- 1000Songs
|  |-- annotations
|    |-- valence_cont_std.csv
|    |-- arousal_cont_std.csv
|    |-- valence_cont_average.csv
|    |-- arousal_cont_average.csv
|    |-- static_annotations.csv
|  |-- chorus
|    |-- wav
|      |-- 2.wav
|      |-- 3.wav
|      |-- ...
|    |-- 2.mp3
|    |-- 3.mp3
|    |-- ...
|  |-- songs_info.csv
```