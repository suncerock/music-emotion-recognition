# music-emotion-recognition

## Datasets

- [AMS (AllMusic Mood Subset)](https://github.com/fdlm/listening-moods): multi-label classification dataset, 53585, 6695, 6713 tracks for training, validation and test, 188 mood tags, from [3]
- [1000 Songs for Emotional Analysis of Music](https://dl.acm.org/doi/10.1145/2506364.2506365): both dynamic and static V-A annotations, from [5]. [1] uses different feature sets including spectral features, chroma ir ComPare feature set. Please refer to [1] for more details on features.
- [PMEmo](https://dl.acm.org/doi/10.1145/3206025.3206037): both dynamic and static V-A annotations and some features, from [6], used in [2].

## Metrics

- Categorial: Multi-label classification, used by [3]
- Valance-Arousal: Two-dimensional regression, used by [1, 2], evaluated with Kendall's \tau per song, RMSE (root-mean-squared error) or R^2 score.

For more information, please refer to section 2.3 of [3]. Note that the music emotion can be dynamic, i.e. time varing, such as in [1].

## References
[1] [Explaining Perceived Emotion Predictions in Music: an Attentive Approach](https://program.ismir2020.net/poster_1-18.html) - ISMIR 2020

[2] [The Multiple Voices of Musical Emotions: Source Separation for Improving Music Emotion Recognition Models and Their Interpretability](https://program.ismir2020.net/poster_2-19.html) - ISMIR 2020

[3] [Mood Classification Using Listening Data](https://program.ismir2020.net/poster_4-10.html) - ISMIR 2020

[4] [Music Emotion Recognition: A State of the Art Review](https://ismir2010.ismir.net/proceedings/ismir2010-45.pdf) - ISMIR 2010

[5] [1000 Songs for Emotional Analysis of Music](https://dl.acm.org/doi/10.1145/2506364.2506365) - ACM CrowdMM 2013

[6] [The PMEmo Dataset for Music Emotion Recognition](https://dl.acm.org/doi/10.1145/3206025.3206037) - ICMR 2018
