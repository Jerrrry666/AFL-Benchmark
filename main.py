import csv
import importlib
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from alg.asyncbase import AsyncBaseClient, AsyncBaseServer, Status
from utils.options import args_parser

sys.path.append(str(Path(__file__).resolve().parent.parent))


class FedSim:
    def __init__(self, args):
        self.args = args
        self.suffix = Path("logs") / args.suffix
        self.begin_round = 0
        self.resume_round = args.resume_round if args.resume_round != -1 else 'max'

        # === load trainer ===
        alg_module = importlib.import_module(f"alg.{args.alg}")

        # === init clients & server ===
        self.clients = [alg_module.Client(idx, args) for idx in tqdm(range(args.total_num), desc='Building clients')]
        self.server = alg_module.Server(args, self.clients)

        # === check if suffix dir exists and config matches ===
        can_resume = False
        log_cache = None
        if self.suffix.exists():
            existing_configs = list(self.suffix.glob("*.yaml"))
            if existing_configs:
                from utils.options import compare_configs
                can_resume = compare_configs(args.cfg, existing_configs[0])
                log_cache = f"Resuming training under {self.suffix}..." if can_resume else f"\nConfig mismatch, creating new directory."
            else:
                log_cache = f"No config found in {self.suffix}, starting new training run."
            print(log_cache)
        self.suffix.mkdir(parents=True, exist_ok=True)
        # backup current config
        import shutil
        config_target = self.suffix / f"{args.alg}_{args.dataset}_{args.model}_{args.total_num}c_{args.epoch}E_lr{args.lr}.yaml"
        shutil.copy("config.yaml", config_target)

        # === logger path ===
        logger_path = self.suffix / f"{args.alg}_{args.dataset}_{args.model}_{args.total_num}c_{args.epoch}E_lr{args.lr}.log"
        self.logger = setup_logger(str(logger_path))
        self.logger.info(log_cache) if log_cache is not None else None

        # === resume if possible ===
        if can_resume:
            try:
                self.begin_round = self.load_checkpoint(begin_round_idx=self.resume_round)
            except Exception as e:
                self.logger.warning(f"Failed to resume training from {self.suffix}: {e}. Starting a new training run.")

    def simulate(self):
        acc_list = []
        time_list = []
        rnd_list = []
        TEST_GAP = self.args.test_gap
        CKPT_GAP = self.args.ckpt_gap

        # check if it is an async methods
        if isinstance(self.server, AsyncBaseServer):
            TEST_GAP *= int(self.args.total_num * self.args.sr)
        try:
            for rnd in tqdm(range(self.begin_round, self.server.total_round), desc="Communication Round",
                            initial=self.begin_round, total=self.server.total_round):
                # ===================== train =====================
                self.server.round = rnd

                # 异步多卡并行的新逻辑
                if isinstance(self.server, AsyncBaseServer):
                    # 持续运行直到有足够客户端完成训练可以聚合
                    while True:
                        self.server.run()

                        # 检查是否有足够的客户端完成训练
                        if len(self.server.pending_aggregation_queue) >= self.server.aggregation_batch_size:
                            break

                        # 检查是否所有客户端都处于活跃状态（避免死锁）
                        active_clients = len([c for c in self.server.clients if c.status == Status.ACTIVE])
                        if active_clients >= self.server.MAX_CONCURRENCY:
                            # 如果所有客户端都在训练，等待一小段时间再继续
                            import time
                            time.sleep(0.1)
                            continue
                        else:
                            break  # 可以继续采样新的客户端
                else:
                    # 同步服务器的原有逻辑
                    self.server.run()

                # ===================== save ckpt =====================
                if (CKPT_GAP > 0) and ((self.server.total_round - rnd <= 10) or (rnd % CKPT_GAP == (CKPT_GAP - 1))):
                    _ = self.save_checkpoint(rnd)

                # ===================== test =====================
                if (self.server.total_round - rnd <= 10) or (rnd % TEST_GAP == (TEST_GAP - 1)):
                    if CKPT_GAP == 0 or CKPT_GAP < 0:
                        _ = self.save_checkpoint(rnd)

                    # test
                    # todo: independent thread for test? then need load from checkpoint file
                    ret_dict = self.server.test_all()
                    acc = ret_dict["acc"]
                    acc_list.append(acc)
                    time = self.server.wall_clock_time
                    time_list.append(time)
                    rnd_list.append(rnd)

                    # Write to CSV after each test
                    csv_path = self.suffix / "acc&time_history.csv"
                    with open(csv_path, "a") as f:
                        writer = csv.writer(f)
                        if f.tell() == 0:  # Write header if file is empty
                            writer.writerow(["rnd", "acc", "time"])
                        writer.writerow([rnd, acc, time])

                    self.logger.info(f"[Round {rnd}]\tAcc: {acc:.2f}\t| Time: {time:.2f}s")


        except KeyboardInterrupt:
            ...
        finally:
            avg_count = 10
            acc_avg = np.mean(acc_list[-avg_count:]).item() if acc_list else 0.0
            acc_max = np.max(acc_list).item() if acc_list else 0.0

            self.logger.info("==========Summary==========")
            self.logger.info(f"[Total] Acc: {acc_avg:.2f}\t| Max Acc: {acc_max:.2f}")

            # Check consistency in finally block
            try:
                csv_path = self.suffix / "acc&time_history.csv"
                recorded_rnd, recorded_acc, recorded_time = [], [], []
                if csv_path.exists():
                    with open(csv_path, "r") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            recorded_rnd.append(int(row["rnd"]))
                            recorded_acc.append(float(row["acc"]))
                            recorded_time.append(float(row["time"]))

                # Take union of lists to ensure consistency
                combined = set(zip(rnd_list, acc_list, time_list)) | set(zip(recorded_rnd, recorded_acc, recorded_time))
                combined = sorted(combined, key=lambda x: x[0])  # Sort by rnd

                with open(csv_path, "w") as f:
                    writer = csv.writer(f)
                    writer.writerow(["rnd", "acc", "time"])
                    writer.writerows(combined)
            except Exception as e:
                self.logger.error(f"Error during finalization: {e}")

    def save_checkpoint(self, round_idx: int) -> Path:
        """
        Save the server model and all clients' tensors into one file named by round index.
        Path: logs/<suffix>/checkpoints/round_<r>.pt
        """
        ckpt_dir = self.suffix / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"round_{round_idx}.pt"

        try:
            # build checkpoint dictionary
            ckpt_data = {
                "round"          : round_idx,
                "wall_clock_time": self.server.wall_clock_time,
                "server_model"   : self.server.model.state_dict(),
                "clients"        : {
                    client.id: {
                        "client_personalized_tensor":
                            client.cached_personalized_params if isinstance(client, AsyncBaseClient)
                            else client.model2personalized_tensor(),
                    }
                    for client in self.clients
                },
            }
            torch.save(ckpt_data, ckpt_path)
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint at round {round_idx}: {e}")
            raise
        return ckpt_path

    def load_checkpoint(self, suffix_dir_path: str | Path | None = None,
                        begin_round_idx: int | str = 'max') -> int | None:
        """
        Load server and client states from checkpoint file.
        If suffix_dir_path is provided, it replaces self.suffix.
        """
        # 若给定新的 suffix 目录，则替换默认的 self.suffix
        base_dir = Path(suffix_dir_path) if suffix_dir_path is not None else self.suffix
        ckpt_dir = base_dir / "checkpoints"
        if not ckpt_dir.exists():
            self.logger.error(f"Checkpoint not found: {ckpt_dir}")
            raise ValueError(f"Checkpoint not found: {ckpt_dir}")

        if begin_round_idx == "max":
            ckpt_files = sorted(ckpt_dir.glob("round_*.pt"))
            if not ckpt_files:
                self.logger.error(f"No checkpoint files found in {ckpt_dir}")
                return
            ckpt_path = ckpt_files[-1]
            begin_round_idx = int(ckpt_path.stem.split("_")[1])
            if begin_round_idx >= self.server.total_round:
                self.logger.error(f"Checkpoint round {begin_round_idx} exceeds total rounds {self.server.total_round}.")
                raise ValueError(f"Checkpoint round {begin_round_idx} exceeds total rounds {self.server.total_round}.")
        else:
            ckpt_path = ckpt_dir / f"round_{begin_round_idx}.pt"

        try:
            ckpt_data = torch.load(ckpt_path, map_location="cpu")

            # === restore server ===
            self.server.model.load_state_dict(ckpt_data["server_model"])
            self.server.round = ckpt_data.get("round", begin_round_idx + 1)
            self.server.wall_clock_time = ckpt_data.get("wall_clock_time", 0.0)

            # === restore clients' personalized params ===
            client_states = ckpt_data.get("clients", {})
            for client in self.clients:
                if client.id not in client_states:
                    self.logger.warning(f"Client {client.id} missing in checkpoint.")
                    continue
                client.personalized_tensor2model(client_states[client.id]["client_personalized_tensor"])

            self.logger.info(f"Resumed from checkpoint from {ckpt_path}")
            return begin_round_idx + 1

        except Exception as e:
            self.logger.error(f"Failed to load checkpoint {ckpt_path}: {e}")
            raise


def setup_logger(log_path: str | Path, name: str = "fed", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers: return logger
    logger.setLevel(level)
    logger.propagate = False
    fmt = logging.Formatter("%(message)s")
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(path, encoding="utf-8")):
        h.setLevel(level)
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


if __name__ == "__main__":
    args = args_parser()
    fed = FedSim(args=args)
    fed.simulate()
