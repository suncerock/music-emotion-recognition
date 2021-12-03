class Config(object):
    def __init__(self):
        self.data_cfg = dict(
            data_path="E:/Music Emotion Recognition/dataset/1000Songs/spectrogram.pkl",
            train_list="E:/Music Emotion Recognition/dataset/1000Songs/1000songs_train.txt",
            valid_list="E:/Music Emotion Recognition/dataset/1000Songs/1000songs_valid.txt",
            mode='static',
            static_anno="E:/Music Emotion Recognition/dataset/1000Songs/annotations/static_annotations.csv",
            dynamic_arousal_anno="E:/Music Emotion Recognition/dataset/1000Songs/annotations/arousal_cont_average.csv",
            dynamic_valence_anno="E:/Music Emotion Recognition/dataset/1000Songs/annotations/valence_cont_average.csv",
            batch_size=64,
            shuffle=False
        )

        self.model_cfg = dict(

        )