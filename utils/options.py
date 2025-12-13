import argparse
import importlib
from pathlib import Path
from typing import List, Optional

import yaml


def _load_yaml_dict(path: Path) -> dict:
    try:
        with path.open('r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def args_parser():
    # ===== minimal parser to capture --cfg from CLI =====
    prelim_parser = argparse.ArgumentParser(add_help=False)
    prelim_parser.add_argument('--cfg', type=str, default='config1.yaml', help='Model')
    prelim_args, _ = prelim_parser.parse_known_args()

    # Load YAML defaults using the config path from CLI
    config_path = Path(prelim_args.cfg)
    if not config_path.exists():
        raise FileExistsError(f"Warning: config file does not exist: {config_path}")
    config_path = Path(prelim_args.cfg)
    yaml_config = _load_yaml_dict(config_path)

    # ===== full parser with all common arguments =====
    parser = argparse.ArgumentParser(parents=[prelim_parser])

    parser.add_argument('--alg', type=str, help='Algorithm')

    # ----- Basic Setting -----
    parser.add_argument('--suffix', type=str, help="Suffix for file")
    parser.add_argument('--resume_round', type=int, default=-1, help="Round to resume from, -1 for max")
    parser.add_argument('--device', type=int, help="Device to use")
    parser.add_argument('--max_per_device', type=int, default=1, help="Maximum concurrent clients per device")
    parser.add_argument('--dataset', type=str, help="Dataset")
    parser.add_argument('--model', type=str, help="Model")

    # ----- Federated Setting -----
    parser.add_argument('--total_num', type=int, help="Total clients num")
    parser.add_argument('--sr', type=float, help="Clients sample rate")
    parser.add_argument('--rnd', type=int, help="Communication rounds")
    parser.add_argument('--test_gap', type=int, help='Rounds between test phases')

    # ----- Local Training Setting -----
    parser.add_argument('--bs', type=int, help="Batch size")
    parser.add_argument('--epoch', type=int, help="Epoch num")
    parser.add_argument('--lr', type=float, help="Learning rate")
    parser.add_argument('--gamma', type=float, help="Exponential decay of learning rate")

    # ----- Async Setting -----
    parser.add_argument('--decay', type=float, default=0.3, help="Basic weight decay in asynchronous aggregation")

    # Apply YAML defaults
    parser.set_defaults(**yaml_config)

    # Pre-parse known args to detect --alg without failing on yet-to-be-added options
    args, _ = parser.parse_known_args()
    if args.alg:
        alg_module = importlib.import_module(f'alg.{args.alg}')
        if hasattr(alg_module, 'add_args'):
            alg_module.add_args(parser)
    # Final parse after possibly extending the parser
    args = parser.parse_args()

    # Final parse: CLI > alg defaults > YAML defaults > code defaults
    return args


def compare_configs(config_path_a: str | Path,
                    config_path_b: str | Path,
                    compare_keys: Optional[List[str]] = None) -> bool:
    """Compare two YAML config files on a subset of keys.
    Returns True if the selected key-values match exactly.
"""
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

    path_a = Path(config_path_a)
    path_b = Path(config_path_b)
    if not path_a.exists() or not path_b.exists():
        print(f"One or both config files not found: {path_a}, {path_b}")
        return False

    cfg_a = _load_yaml_dict(path_a)
    cfg_b = _load_yaml_dict(path_b)

    sub_a = {k: cfg_a.get(k) for k in compare_keys}
    sub_b = {k: cfg_b.get(k) for k in compare_keys}

    return sub_a == sub_b
