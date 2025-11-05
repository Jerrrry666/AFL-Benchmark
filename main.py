import importlib
import logging
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from alg.asyncbase import AsyncBaseServer
from utils.options import args_parser

sys.path.append(str(Path(__file__).resolve().parent.parent))

def setup_logger(log_path: str, name: str = "fed") -> logging.Logger:
    """
    log to file and terminal at same time。
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter("%(message)s")

    # output to file
    if log_path:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    # output to terminal
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


class FedSim:
    def __init__(self, args):
        self.args = args
        args.suffix = Path("logs") / args.suffix
        args.suffix.mkdir(parents=True, exist_ok=True)

        # === 组织日志路径 ===
        output_path = args.suffix / f"{args.alg}_{args.dataset}_{args.model}_{args.total_num}c_{args.epoch}E_lr{args.lr}"
        log_file = output_path.with_suffix(".log")
        self.logger = setup_logger(str(log_file))

        # === load trainer ===
        alg_module = importlib.import_module(f"alg.{args.alg}")

        # === init clients & server ===
        self.clients = [alg_module.Client(idx, args) for idx in tqdm(range(args.total_num))]
        self.server = alg_module.Server(args, self.clients)

    def simulate(self):
        acc_list = []
        TEST_GAP = self.args.test_gap

        # check if it is an async methods
        if isinstance(self.server, AsyncBaseServer):
            TEST_GAP *= int(self.args.total_num * self.args.sr)
            self.server.total_round *= int(self.args.total_num * self.args.sr)
        try:
            for rnd in tqdm(range(0, self.server.total_round), desc="Communication Round", leave=False):
                # ===================== train =====================
                self.server.round = rnd
                self.server.run()

                # ===================== test =====================
                if (self.server.total_round - rnd <= 10) or (rnd % TEST_GAP == (TEST_GAP - 1)):
                    ret_dict = self.server.test_all()
                    acc = ret_dict["acc"]
                    acc_list.append(acc)

                    self.logger.info(f"[Round {rnd}] Acc: {acc:.2f} | Time: {self.server.wall_clock_time:.2f}s")

        except KeyboardInterrupt:
            ...
        finally:
            avg_count = 10
            acc_avg = np.mean(acc_list[-avg_count:]).item() if acc_list else 0.0
            acc_max = np.max(acc_list).item() if acc_list else 0.0

            self.logger.info("==========Summary==========")
            self.logger.info(f"[Total] Acc: {acc_avg:.2f} | Max Acc: {acc_max:.2f}")


if __name__ == "__main__":
    args = args_parser()
    fed = FedSim(args=args)
    fed.simulate()