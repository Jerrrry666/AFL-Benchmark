import random
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import yaml

from utils.dataset_utils import check, save_file, separate_data, split_data

random.seed(1)
np.random.seed(1)

def generate_dataset(cfg):
    dir_path = Path(cfg['dir_path'] + '_' + f'{cfg["num_clients"]}')
    dir_path.mkdir(parents=True, exist_ok=True)

    if check(cfg): return

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    trainset = torchvision.datasets.EMNIST(root=str(dir_path / "rawdata"),
                                           split='balanced', train=True, download=True, transform=transform)
    testset = torchvision.datasets.EMNIST(root=str(dir_path / "rawdata"),
                                          split='balanced', train=False, download=True, transform=transform)
    trainset.data, trainset.targets = next(
        iter(torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False)))
    testset.data, testset.targets = next(
        iter(torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False)))
    X = np.concatenate([trainset.data.numpy(), testset.data.numpy()])
    y = np.concatenate([trainset.targets.numpy(), testset.targets.numpy()])

    cfg['class_num'] = len(set(y))
    X, y, statistic = separate_data((X, y), cfg)
    train_data, test_data = split_data(X, y, cfg)
    save_file(train_data, test_data, cfg)


if __name__ == "__main__":
    with Path('config.yaml').open('r') as f:
        config = yaml.load(f.read(), Loader=yaml.Loader)
    generate_dataset(config)