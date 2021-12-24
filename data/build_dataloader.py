from .dataloader_1000songs import build_1000songs_dataloader
from .dataloader_PMEmo import build_PMEMO_dataloader

def build_dataloader(config):
    if config.type == '1000songs':
        train_dataloader, valid_dataloader = build_1000songs_dataloader(data_path=getattr(config, "data_path"),
                                                                        train_list=getattr(config, "train_list"),
                                                                        valid_list=getattr(config, "valid_list"),
                                                                        annotations=getattr(config, "annotations"),
                                                                        normalize_label=getattr(config, "normalize_label", True),
                                                                        batch_size=getattr(config, "batch_size", 64),
                                                                        shuffle=getattr(config, "shuffle", False))
    elif config.type == 'PMEmo':
        train_dataloader, valid_dataloader = build_PMEMO_dataloader(data_dir=getattr(config, "data_dir"),
                                                                    length=getattr(config, "800"),
                                                                    normalize_label=getattr(config, "normalize_label", True),
                                                                    random_seed=getattr(config, "random_seed", 2019),
                                                                    batch_size=getattr(config, "batch_size", 64),
                                                                    shuffle=getattr(config, "shuffle", False))
    else:
        raise Exception("Unknown dataset type! (Must be one of ['1000songs', 'PMEmo']")
    return train_dataloader, valid_dataloader