import argparse
import importlib
from pathlib import Path

import yaml


def _load_yaml_dict(path: Path) -> dict:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _normalize_cfg(cfg: dict, ignore_keys=None) -> dict:
    if ignore_keys is None:
        ignore_keys = {"resume_round", "rnd", "round", "suffix", "test_gap", "seed"}
    return {k: v for k, v in (cfg or {}).items() if k not in ignore_keys}


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', type=str, help='Algorithm')

    # ===== Basic Setting ======
    parser.add_argument('--suffix', type=str, help="Suffix for file")
    parser.add_argument('--resume_round', type=int, default=-1, help="Round to resume from, -1 for max")
    parser.add_argument('--device', type=int, help="Device to use")
    parser.add_argument('--dataset', type=str, help="Dataset")
    parser.add_argument('--model', type=str, help="Model")
    parser.add_argument('--model_in_cpu', action='store_true')

    # ===== Federated Setting =====
    parser.add_argument('--total_num', type=int, help="Total clients num")
    parser.add_argument('--sr', type=float, help="Clients sample rate")
    parser.add_argument('--rnd', type=int, help="Communication rounds")
    parser.add_argument('--test_gap', type=int, help='Rounds between test phases')

    # ===== Local Training Setting =====
    parser.add_argument('--bs', type=int, help="Batch size")
    parser.add_argument('--epoch', type=int, help="Epoch num")
    parser.add_argument('--lr', type=float, help="Learning rate")
    parser.add_argument('--gamma', type=float, help="Exponential decay of learning rate")

    # ===== Async Setting =====
    parser.add_argument('--decay', type=float, default=0.3, help="Basic weight decay in asynchronous aggregation")

    # === read args from yaml ===
    with open('config.yaml', 'r') as f:
        yaml_config = yaml.load(f.read(), Loader=yaml.Loader)
    for k, v in yaml_config.items():
        parser.set_defaults(**{k: v})

    # === read args from command ===
    args, _ = parser.parse_known_args()

    # === read specific args from each method
    alg_module = importlib.import_module(f'alg.{args.alg}')
    spec_args = alg_module.add_args(parser) if hasattr(alg_module, 'add_args') else args
    return spec_args


def compare_configs(config_path_a: str | Path,
                    config_path_b: str | Path,
                    compare_keys: list[str] | None = None) -> bool:
    """
    Compare two YAML config files based only on specified keys.
    If compare_keys is None, defaults to a core set of important hyperparameters.
    """
    path_a = Path(config_path_a)
    path_b = Path(config_path_b)
    if not path_a.exists() or not path_b.exists():
        print(f"One or both config files not found: {path_a}, {path_b}")
        return False

    cfg_a = _load_yaml_dict(path_a)
    cfg_b = _load_yaml_dict(path_b)

    # Define default keys to compare if none provided
    if compare_keys is None:
        compare_keys = [
            # Basic config
            "alg", "dataset", "model",
            # Federated config
            "total_num", "sr",
            # Local training config
            "bs", "epoch", "lr", "gamma"
        ]

    sub_a = {k: cfg_a.get(k) for k in compare_keys if k in cfg_a}
    sub_b = {k: cfg_b.get(k) for k in compare_keys if k in cfg_b}

    return sub_a == sub_b
