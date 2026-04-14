# Agents Guide for AFL-Benchmark

Purpose: actionable, compact guidance so an AI coding agent can be productive immediately in this repo.

1) Big picture (how things fit together)
- main entry: `main.py` builds alg module `alg.{args.alg}`, constructs Client objects and a Server, and runs `FedSim.simulate()`.
- Algorithm plugins live in `alg/` and must provide `Client` and `Server` classes. Synchronous algorithms extend `alg/base.py` (BaseClient/BaseServer); async algorithms extend `alg/asyncbase.py` (AsyncBaseClient/AsyncBaseServer).
- Models are loaded via `model/config.py::load_model(args)` which imports `model.{args.model}` and instantiates model_{dataset}.
- Datasets and generators live in `dataset/` and dataset-specific generators are `dataset/generate_*.py`. Dataset wiring uses `utils/data_utils.get_dataset`.

2) How to run (concrete commands)
- Generate dataset (edit `dataset/config.yaml` first):
  python dataset/generate_mnist.py
- Run an experiment (example):
  python main.py --cfg config.yaml --alg fedavg --dataset cifar10a1 --model resnet18 --total_num 100 --sr 0.1 --rnd 200 --suffix myrun
- Check logs & checkpoints under `logs/<suffix>/` (logger and checkpoint names built in `main.py`).

3) CLI / config precedence & extension points
- Defaults are loaded from the YAML passed via `--cfg` (see `utils/options.py`). Precedence: CLI args > YAML > code defaults in `utils/options.py`.
- `utils/options.py` pre-parses `--alg` and calls `alg_module.add_args(parser)` if present. To add algorithm-specific CLI flags, implement `add_args(parser)` in your `alg/*.py`.

4) Checkpoints & resume semantics (important when automating experiments)
- Checkpoints saved at `logs/<suffix>/checkpoints/round_<r>.pt` (see `FedSim.save_checkpoint` in `main.py`).
- Checkpoint format keys: `round`, `wall_clock_time`, `server_model` (state_dict), `clients` (mapping id -> {client_personalized_tensor}). An agent reading checkpoints should expect server weights + per-client personalized tensors.
- Resuming: `FedSim` will compare configs using `utils/options.compare_configs`. Resume uses latest round by default (or pass `--resume_round`). If configs differ, a new directory is created.

5) Concurrency & device model (how training is scheduled)
- Clients are light wrappers; actual device placement is handled by server using `utils/run_utils.OnDeviceRun`/`OnDevice` to temporarily move models and optimizers between CPU and GPUs.
- BaseServer uses `max_per_device` (per-server-device serialization cap). AsyncBaseServer exposes `max_concurrent_per_device`, `MAX_CONCURRENCY` (client-level cap), and `aggregation_batch_size`.
- Async flow uses a heap `pending_aggregation_queue` (finish-time ordered) and `Status` enum (IDLE/ACTIVE/ERROR) to manage sampling, training, and aggregation. See `alg/asyncbase.py` for the exact lifecycle.

6) Model personalization & parameter selection
- `BaseClient` supports per-parameter personalization via `parameter_personalized_flag` and `share_flag`. Use `model2shared_tensor()` / `model2personalized_tensor()` and reverse methods to extract/apply subsets of params.
- Async training caches personalized params on train start (`client.cached_personalized_params`) so tests can use the cached personalized state while training continues.

7) Implementation pattern for adding algorithms
- Minimal synchronous plugin (`alg/myalg.py`):
  class Client(BaseClient):
      def run(self):
          self.train()

  class Server(BaseServer):
      def run(self):
          self.sample(); self.downlink(); self.client_update(); self.uplink(); self.aggregate()
- For async algorithms, extend `AsyncBaseClient` / `AsyncBaseServer` and follow the `pending_aggregation_queue` + `clients_to_aggregate` pattern shown in `alg/asyncbase.py`.

8) Quick notes for debugging and automation
- Logs: `main.py` sets a logger writing to stdout and `logs/<suffix>/*.log` (use these files to capture experiment traces).
- When moving models between devices in unit tests, prefer `OnDeviceRun` to mimic actual execution environment used by servers.
- To programmatically resume, ensure the YAML in `logs/<suffix>` matches the CLI `--cfg` (compare uses keys listed in `utils/options.compare_configs`).

Files to inspect first (high signal): `main.py`, `utils/options.py`, `alg/base.py`, `alg/asyncbase.py`, `utils/run_utils.py`, `model/config.py`, `dataset/config.yaml` and `dataset/generate_*.py`.

End.

