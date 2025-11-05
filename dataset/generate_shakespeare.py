import json
import numpy as np
import random
import yaml
from pathlib import Path

from dataset.utils.dataset_utils import check
from utils.language_utils import letter_to_index, word_to_indices

random.seed(1)
np.random.seed(1)
data_path_train = "../../shakespeare/data/train/all_data_niid_2_keep_0_train_9.json"
data_path_test = "../../shakespeare/data/test/all_data_niid_2_keep_0_test_9.json"


# https://github.com/TalwalkarLab/leaf/blob/master/models/shakespeare/stacked_lstm.py#L40
def process_x(raw_x_batch):
    x_batch = [word_to_indices(word) for word in raw_x_batch]
    x_batch = np.array(x_batch)
    return x_batch


def process_y(raw_y_batch):
    y_batch = [letter_to_index(c) for c in raw_y_batch]
    y_batch = np.array(y_batch)
    return y_batch

def generate_dataset(cfg):
    dir_path = Path(cfg['dir_path'] + '-' + f'{cfg["client_num"]}')
    dir_path.mkdir(parents=True, exist_ok=True)

    if check(cfg): return

    train_path = dir_path / "train"
    test_path = dir_path / "test"
    train_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)

    with Path(data_path_train).open() as f:
        raw_train = json.load(f)['user_data']
    with Path(data_path_test).open() as f:
        raw_test  = json.load(f)['user_data']

    train_ = [{'x': process_x(v['x']), 'y': process_y(v['y'])} for v in raw_train.values()]
    test_ = [{'x': process_x(v['x']), 'y': process_y(v['y'])} for v in raw_test.values()]

    idx = sorted(range(len(train_)), key=lambda i: len(train_[i]['x']))
    train, test = [train_[i] for i in idx], [test_[i] for i in idx]

    for idx, data in enumerate(train):
        with (train_path / f"{idx}.npz").open('wb') as f:
            np.savez_compressed(f, data=data)

    for idx, data in enumerate(test):
        with (test_path / f"{idx}.npz").open('wb') as f:
            np.savez_compressed(f, data=data)


if __name__ == "__main__":
    with Path('config.yaml').open('r') as f:
        config = yaml.load(f.read(), Loader=yaml.Loader)
    generate_dataset(config)