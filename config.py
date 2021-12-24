class Songs1000DatasetConfig(object):
    def __init__(self):
        self.type = "1000songs"
        self.data_path = "E:/Music Emotion Recognition/dataset/1000Songs/spectrogram.pkl"
        self.train_list = "E:/Music Emotion Recognition/dataset/1000Songs/1000songs_train.txt"
        self.valid_list = "E:/Music Emotion Recognition/dataset/1000Songs/1000songs_valid.txt"
        self.annotations = "E:/Music Emotion Recognition/dataset/1000Songs/annotations/static_annotations.csv"
        self.normalize_label = True
        self.batch_size = 64
        self.shuffle = False


class PMEmoDatasetConfig(object):
    def __init__(self):
        self.type = "PMEmo"
        self.data_dir = "E:/Music Emotion Recognition/dataset/PMEmo2019/"
        self.length = 800
        self.normalize_label = True
        self.random_seed = 2019
        self.batch_size = 64
        self.shuffle = False


class StepLRSchedulerConfig(object):
    def __init__(self):
        self.type = 'step'
        self.decay_t = 1
        self.decay_rate = 0.9

        self.warmup_steps = 0
        self.warmup_lr_init = 0.
        self.warmup_prefix = True

        self.t_in_epochs = True


class CosineLRSchedulerConfig(object):
    def __init__(self):
        self.type = 'cosine'
        self.t_initial = 100
        self.lr_min = 0.

        self.cycle_mul = 1.
        self.cycle_decay = 1.
        self.cycle_limit = 1

        self.warmup_steps = 0
        self.warmup_lr_init = 0.
        self.warmup_prefix = True

        self.t_in_epochs = False


class Config(object):
    def __init__(self):
        self.data_cfg = PMEMO_cfg
        self.scheduler_cfg = StepLRSchedulerConfig()
        self.model_cfg = dict(

        )