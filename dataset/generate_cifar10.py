import random
from pathlib import Path

import numpy as np
import torchvision
import torchvision.transforms as transforms
import yaml

from utils.dataset_utils import check, save_file, separate_data, split_data

random.seed(1)
np.random.seed(1)


def generate_dataset(cfg):
    dir_path = Path(cfg['dir_path'] + '_' + f'{cfg["client_num"]}')
    dir_path.mkdir(parents=True, exist_ok=True)

    if check(cfg): return

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    trainset = torchvision.datasets.CIFAR10(root="~/Dataset/CIFAR10",
                                            train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root="~/Dataset/CIFAR10",
                                           train=False, download=True, transform=transform)

    trainset.data = np.array(trainset.data)
    trainset.targets = np.array(trainset.targets)
    testset.data = np.array(testset.data)
    testset.targets = np.array(testset.targets)

    X = np.concatenate([trainset.data, testset.data])
    y = np.concatenate([trainset.targets, testset.targets])

    cfg['class_num'] = len(set(y))
    X, y, statistic = separate_data((X, y), cfg)
    train_data, test_data = split_data(X, y, cfg)
    save_file(train_data, test_data, cfg)


if __name__ == "__main__":
    with Path('config.yaml').open('r') as f:
        config = yaml.load(f.read(), Loader=yaml.Loader)
    generate_dataset(config)
