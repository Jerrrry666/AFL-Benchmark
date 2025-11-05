import random
from pathlib import Path

import numpy as np
import torchvision
import torchvision.transforms as transforms
import yaml
from tqdm import tqdm

from utils.dataset_utils import check, save_file, separate_data, split_data

random.seed(1)
np.random.seed(1)

# Helper to convert torchvision dataset (with transform) to numpy arrays
def dataset_to_numpy(ds):
    X_list = []
    y_list = []
    for i in tqdm(range(len(ds)), desc="Converting dataset to numpy"):
        img_t, label = ds[i]  # transform is already applied here
        X_list.append(img_t.numpy())  # CHW, float32 after ToTensor/Normalize
        y_list.append(label)
    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)
    return X, y


def generate_dataset(cfg):
    dir_path = Path(cfg['dir_path'] + '-' + f'{cfg["client_num"]}')
    dir_path.mkdir(parents=True, exist_ok=True)

    if check(cfg): return

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    trainset = torchvision.datasets.CIFAR10(root="~/Dataset/",
                                            train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root="~/Dataset/",
                                           train=False, download=True, transform=transform)

    X_train, y_train = dataset_to_numpy(trainset)
    X_test, y_test = dataset_to_numpy(testset)
    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)

    cfg['class_num'] = len(set(y))
    X, y, statistic = separate_data((X, y), cfg)
    train_data, test_data = split_data(X, y, cfg)
    save_file(train_data, test_data, cfg)


if __name__ == "__main__":
    with Path('config.yaml').open('r') as f:
        config = yaml.load(f.read(), Loader=yaml.Loader)
    assert config['dir_path'].lower() == 'cifar10', 'Dataset name does not match saving dir_path (dataset/config.yaml) !'
    generate_dataset(config)
