"""
DomainNet数据集生成器
支持双层noniid分布：标签层面和风格层面的独立分布控制
"""

import json
import random
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

from utils.dataset_utils import check, save_file, split_data
from utils.dual_distribution_utils import (
    save_detailed_statistics,
    split_dual_distribution,
)

random.seed(1)
np.random.seed(1)

# DomainNet特定配置
ROOT_DIR = "/data/mayiming/Dataset/DomainNet/"
DOMAIN_NAMES = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
NUM_CLASSES_PER_DOMAIN = 345


class DomainNetLoader:
    """DomainNet数据集加载器"""

    def __init__(self, root_dir=ROOT_DIR):
        self.root_dir = Path(root_dir)
        self.domain_names = DOMAIN_NAMES
        self.num_domains = len(self.domain_names)
        self.num_classes_per_domain = NUM_CLASSES_PER_DOMAIN

        # 简化的缓存文件路径
        self.cache_data_path = self.root_dir / "cached_data.npz"
        self.cache_metadata_path = self.root_dir / "cache_metadata.json"

    def _is_cache_valid(self):
        """简单检查缓存文件是否存在"""
        return self.cache_data_path.exists() and self.cache_metadata_path.exists()

    def _save_cache(self, data, labels, domains):
        """保存数据到缓存"""
        print("💾 Saving data to cache...")

        # 保存数据
        np.savez_compressed(
            str(self.cache_data_path), data=data, labels=labels, domains=domains
        )

        # 保存简化的元数据
        cache_metadata = {
            "cache_time": str(np.datetime64("now")),
            "total_samples": len(data),
            "num_classes": len(np.unique(labels)),
            "num_domains": len(np.unique(domains)),
        }

        with open(self.cache_metadata_path, "w") as f:
            json.dump(cache_metadata, f, indent=2)

        print(f"✅ Cache saved to {self.cache_data_path}")

    def _load_from_cache(self):
        """从缓存加载数据"""
        print("📂 Loading data from cache...")

        try:
            cache_data = np.load(str(self.cache_data_path), allow_pickle=True)
            data = cache_data["data"]
            labels = cache_data["labels"]
            domains = cache_data["domains"]

            print(
                f"✅ Cache loaded: {len(data)} samples, {len(np.unique(labels))} classes, {len(np.unique(domains))} domains"
            )
            return data, labels, domains

        except Exception as e:
            print(f"❌ Failed to load cache: {e}")
            return None, None, None

    def load_dataset(self):
        """加载DomainNet数据集"""
        print("🔍 Checking for cached data...")

        # 尝试从缓存加载
        if self._is_cache_valid():
            data, labels, domains = self._load_from_cache()
            if data is not None:
                return data, labels, domains

        print("📁 No valid cache found, scanning filesystem...")
        print("Loading DomainNet dataset...")

        all_data = []
        all_labels = []
        all_domains = []

        for domain_idx, domain_name in enumerate(self.domain_names):
            domain_path = self.root_dir / domain_name

            if not domain_path.exists():
                print(f"⚠️ Warning: Domain path {domain_path} does not exist!")
                continue

            print(f"📂 Loading domain: {domain_name}")

            # 获取该domain下的所有类别文件夹
            class_dirs = [d for d in domain_path.iterdir() if d.is_dir()]
            class_dirs = sorted(class_dirs)  # 确保顺序一致

            for class_idx, class_dir in enumerate(
                tqdm(class_dirs, desc=f"Classes in {domain_name}")
            ):
                class_name = class_dir.name

                # 获取该类别下的所有图片文件
                img_files = []
                for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                    img_files.extend(class_dir.glob(ext))

                for img_file in img_files:
                    try:
                        # 尝试加载图片以验证文件完整性
                        with Image.open(img_file) as img:
                            img.verify()  # 验证图片完整性

                        all_data.append(str(img_file))
                        all_labels.append(class_idx)
                        all_domains.append(domain_idx)

                    except Exception as e:
                        print(f"⚠️  Skipping corrupted image: {img_file}")
                        continue

        print(
            f"✅ Dataset loaded: {len(all_data)} samples, {len(set(all_labels))} classes, {len(set(all_domains))} domains"
        )

        # 保存到缓存
        if all_data:  # 只有在有数据时才保存缓存
            data = np.array(all_data)
            labels = np.array(all_labels)
            domains = np.array(all_domains)
            self._save_cache(data, labels, domains)
            return data, labels, domains
        else:
            print("❌ No data found!")
            return np.array([]), np.array([]), np.array([])

    def get_class_mapping(self):
        """获取类别映射信息"""
        class_mapping = {}
        for domain_idx, domain_name in enumerate(self.domain_names):
            domain_path = self.root_dir / domain_name
            if domain_path.exists():
                class_dirs = sorted([d for d in domain_path.iterdir() if d.is_dir()])
                class_mapping[domain_name] = [d.name for d in class_dirs]
        return class_mapping


def generate_dataset(cfg):
    dir_path = Path(
        f'{cfg['dir_path']}-{cfg["client_num"]}-L{cfg["label_partition"]}-F{cfg["feature_partition"]}'
    )
    dir_path.mkdir(parents=True, exist_ok=True)

    if check(cfg):
        return

    # 初始化DomainNet数据集加载器
    dataset = DomainNetLoader()

    # 加载原始数据
    X, y, domains = dataset.load_dataset()

    if len(X) == 0:
        raise RuntimeError(
            "No data loaded! Please check if DomainNet dataset exists at the specified path."
        )

    print(f"Dataset summary:")
    print(f"  Total samples: \t{len(X)}")
    print(f"  Number of classes: \t{len(np.unique(y))}")
    print(f"  Number of domains: \t{len(np.unique(domains))}")
    print(f"  Domain distribution: \t{np.bincount(domains)}")

    # 更新配置
    cfg["class_num"] = len(np.unique(y))
    cfg["domain_num"] = len(np.unique(domains))

    # 使用双层分布控制进行数据切分
    print("\nApplying dual distribution split...")
    X_clients, y_clients, statistic = split_dual_distribution(X, y, domains, cfg)

    # 切分训练集和测试集
    print("\nSplitting train/test data...")
    train_data, test_data = split_data(X_clients, y_clients, cfg)

    # 保存文件
    save_file(train_data, test_data, cfg)

    # 保存详细的统计信息到文件
    statistics_path = dir_path / "client_distribution_statistics.yaml"
    save_detailed_statistics(statistic, str(statistics_path), DOMAIN_NAMES)

    # 保存额外的DomainNet特定信息
    _save_domainnet_metadata(dir_path, cfg, dataset)


def _save_domainnet_metadata(dir_path, cfg, dataset):
    """保存DomainNet特定的元数据"""
    metadata = {
        "domain_names": DOMAIN_NAMES,
        "num_domains": len(DOMAIN_NAMES),
        "num_classes_per_domain": NUM_CLASSES_PER_DOMAIN,
        "class_mapping": dataset.get_class_mapping(),
        "dual_distribution_config": {
            "label_partition": cfg.get("label_partition", "uni"),
            "label_alpha": cfg.get("label_alpha", 10000),
            "feature_partition": cfg.get("feature_partition", "uni"),
            "feature_alpha": cfg.get("feature_alpha", 10000),
        },
    }

    metadata_path = dir_path / "domainnet_metadata.yaml"
    with metadata_path.open("w") as f:
        yaml.dump(metadata, f, default_flow_style=False)

    print(f"DomainNet metadata saved to: {metadata_path}")


if __name__ == "__main__":
    with Path("config.yaml").open("r") as f:
        config = yaml.load(f.read(), Loader=yaml.Loader)

    # 验证配置
    assert (
        config["dir_path"].lower() == "domainnet"
    ), "Dataset name does not match saving dir_path (dataset/config.yaml) !"

    assert (
        config["partition"] == "dual"
    ), "DomainNet requires partition: dual for dual distribution!"

    # 生成数据集
    generate_dataset(config)
