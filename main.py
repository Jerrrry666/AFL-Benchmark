import importlib
import logging
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from alg.asyncbase import AsyncBaseServer
from utils.options import args_parser


# sys.path.append(str(Path(__file__).resolve().parent.parent))


class FedSim:
    def __init__(self, args):
        self.args = args
        self.suffix = Path("logs") / args.suffix
        self.suffix.mkdir(parents=True, exist_ok=True)

        # backup config.yaml
        import shutil
        shutil.copy("config.yaml",
                    self.suffix / f"{args.alg}_{args.dataset}_{args.model}_{args.total_num}c_{args.epoch}E_lr{args.lr}.yaml")

        # === logger path ===
        logger_path = self.suffix / f"{args.alg}_{args.dataset}_{args.model}_{args.total_num}c_{args.epoch}E_lr{args.lr}.log"
        self.logger = setup_logger(str(logger_path))

        # === load trainer ===
        alg_module = importlib.import_module(f"alg.{args.alg}")

        # === init clients & server ===
        self.clients = [alg_module.Client(idx, args) for idx in tqdm(range(args.total_num))]
        self.server = alg_module.Server(args, self.clients)

    def simulate(self):
        acc_list = []
        time_list = []
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
                    time = self.server.wall_clock_time
                    time_list.append(time)

                    self.logger.info(f"[Round {rnd}]\tAcc: {acc:.2f}\t| Time: {time:.2f}s")

        except KeyboardInterrupt:
            ...
        finally:
            avg_count = 10
            acc_avg = np.mean(acc_list[-avg_count:]).item() if acc_list else 0.0
            acc_max = np.max(acc_list).item() if acc_list else 0.0

            self.logger.info("==========Summary==========")
            self.logger.info(f"[Total] Acc: {acc_avg:.2f}\t| Max Acc: {acc_max:.2f}")

            acc_list = np.array(acc_list)
            time_list = np.array(time_list)
            data = np.column_stack((acc_list, time_list))
            csv_path = self.suffix / "acc&time_history.csv"
            np.savetxt(csv_path, data, delimiter=",", header="accuracy,time", comments="", fmt="%.6f")
            self.logger.info(f"Results saved to {csv_path}")


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
