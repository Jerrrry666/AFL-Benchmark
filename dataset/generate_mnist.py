import numpy as np
import random
import torchvision
import torchvision.transforms as transforms
import yaml
from pathlib import Path

from utils.dataset_utils import check, save_file, separate_data, split_data

random.seed(1)
np.random.seed(1)


def generate_dataset(cfg):
    dir_path = Path(cfg['dir_path'] + '-' + f'{cfg["client_num"]}')
    dir_path.mkdir(parents=True, exist_ok=True)

    if check(cfg): return

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    trainset = torchvision.datasets.MNIST(root="~/Dataset/",
                                          train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root="~/Dataset/",
                                         train=False, download=True, transform=transform)

    X = np.concatenate([trainset.data, testset.data])
    y = np.concatenate([np.array(trainset.targets), np.array(testset.targets)])

    cfg['class_num'] = len(set(y))
    X, y, statistic = separate_data((X, y), cfg)
    train_data, test_data = split_data(X, y, cfg)
    save_file(train_data, test_data, cfg)


if __name__ == "__main__":
    with Path('config.yaml').open('r') as f:
        config = yaml.load(f.read(), Loader=yaml.Loader)
    assert config['dir_path'].lower() == 'mnist', 'Dataset name does not match saving dir_path (dataset/config.yaml) !'
    generate_dataset(config)
