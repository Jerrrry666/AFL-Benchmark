"""
Base classes for federated learning algorithms.

This module provides the foundational classes for implementing federated learning
algorithms, including BaseClient and BaseServer classes that handle client-side
and server-side operations respectively.
"""

import random
from concurrent.futures import ThreadPoolExecutor
from threading import Semaphore

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader

from model.config import load_model
from utils.data_utils import get_dataset
from utils.run_utils import OnDeviceRun
from utils.sys_utils import comm_config, device_config


class BaseClient:
    def __init__(self, id, args):
        self.id = id
        self.args = args
        self.dataset_name = args.dataset.split('-')[0]
        self.dataset_path = args.dataset
        self.dataset_train = get_dataset(self.dataset_name)(args, self.id, is_train=True)
        self.dataset_test = get_dataset(self.dataset_name)(args, self.id, is_train=False)
        self.device = assert_device(args.device)
        self.server = None

        self.lr = args.lr
        self.batch_size = args.bs
        self.epoch = args.epoch
        self.model = load_model(args).to(self.device)
        self.loss_func = nn.CrossEntropyLoss()
        self.optim = torch.optim.SGD(self.model.parameters(),
                                     lr=self.lr,
                                     momentum=0.9,
                                     weight_decay=1e-4)
        self.gamma = args.gamma
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optim, gamma=self.gamma)
        self.metric = {'loss': [], 'acc': []}

        # === personalized model ===
        self.p_flag = False
        # keep_local (Ture:local, False:shared)
        self.keep_local = [False for _ in self.model.parameters()] if not self.p_flag \
            else [True for _ in self.model.parameters()]  # default: all global, no personalized
        self.share_flag = [not f for f in self.keep_local]

        if self.dataset_train is not None:
            self.loader_train = DataLoader(
                    dataset=self.dataset_train,
                    batch_size=self.batch_size,
                    shuffle=True,
                    collate_fn=None,
                    num_workers=8,
                    # drop_last=True,
            )
        if self.dataset_test is not None:
            self.loader_test = DataLoader(
                    dataset=self.dataset_test,
                    batch_size=self.batch_size,
                    shuffle=False,
                    collate_fn=None,
                    num_workers=8,
                    # drop_last=True,
            )

        self.task_round = None  # server round when client is sampled
        self.training_time = None

    def run(self):
        raise NotImplementedError

    def train(self):
        self.model.train()
        total_loss = 0.0

        for epoch in range(self.epoch):
            for data in self.loader_train:
                X, y = self.unarchive(data)
                preds = self.model(X)
                loss = self.loss_func(preds, y)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                total_loss += loss.item()

        # === record loss ===
        self.metric['loss'] = total_loss / len(self.loader_train)

    def clone_model(self, source):
        shared_tensor = source.model2shared_tensor()
        self.shared_tensor2model(shared_tensor)

    def unarchive(self, data):
        X, y = data
        if type(X) == type([]):
            X = X[0]
        return X.to(self.device), y.to(self.device)

    def local_test(self):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data in self.loader_test:
                X, y = self.unarchive(data)
                preds = self.model(X)

                _, preds_y = torch.max(preds.data, 1)
                total += y.size(0)
                correct += (preds_y == y).sum().item()
        self.metric['acc'] = 100.00 * correct / total

    def reset_optimizer(self, decay=True):
        if decay and self.task_round > 0:
            self.scheduler.last_epoch = self.task_round - 1
            self.scheduler.step()

    # def model2tensor(self, params=None):
    #     alg_module = importlib.import_module(f'alg.{self.args.alg}')
    #     p_keys = getattr(alg_module, 'p_keys') if hasattr(alg_module, 'p_keys') else []
    #     p_params = [any(key == name.split('.')[0] for key in p_keys)
    #                 for name, _ in self.model.named_parameters()]
    #     selected_params = params if params is not None else [not is_p for is_p in p_params]
    #
    #     return torch.cat([param.detach().view(-1)
    #                       for selected, param in zip(selected_params, self.model.parameters())
    #                       if selected is True], dim=0)

    @staticmethod
    def _model2tensor(model, personalized_flag):
        """
        Only pick the parameters with pick_flag == True
        """
        if not personalized_flag: return None
        selected_param = [p.detach() for pick, p in zip(personalized_flag, model.parameters()) if pick]
        return parameters_to_vector(selected_param).detach()

    def model2shared_tensor(self):
        return self._model2tensor(self.model, self.share_flag)

    def model2personalized_tensor(self):
        if not self.p_flag: return None
        return self._model2tensor(self.model, self.keep_local)

    @staticmethod
    def _tensor2model(tensor, model, personalized_flag):
        if not personalized_flag: return
        selected_params = [p for pick, p in zip(personalized_flag, model.parameters()) if pick]
        with torch.no_grad():
            vector_to_parameters(tensor.to(selected_params[0].device), selected_params)

    def shared_tensor2model(self, tensor):
        self._tensor2model(tensor, self.model, self.share_flag)

    def personalized_tensor2model(self, tensor):
        if not self.p_flag: return
        self._tensor2model(tensor, self.model, self.keep_local)

    def comm_bytes(self):
        model_tensor = self.model2shared_tensor()
        return model_tensor.numel() * model_tensor.element_size()


class BaseServer(BaseClient):
    def __init__(self, args, clients):
        super().__init__(0, args)
        self.dataset_train = None
        self.dataset_test = None
        self.loss_func = None
        self.optim = None

        self.device = None
        self.devices = assert_device(args.device, 's')

        # per-device concurrency control (1 means serialize on each device)
        self.max_per_device = getattr(args, 'max_per_device', 1)
        if isinstance(self.max_per_device, int) and self.max_per_device < 1:
            self.max_per_device = 1

        self.client_num = args.total_num
        self.sample_rate = args.sr
        self.total_round = args.rnd

        self.clients = clients
        self.sampled_clients = []
        self.round = 0
        self.wall_clock_time = 0
        self.received_params = []

        delays = device_config(self.client_num)
        bandwidths = comm_config(self.client_num)

        for client in self.clients:
            client.server = self
            client.delay = delays[client.id]
            client.bandwidth = bandwidths[client.id]

    def run(self):
        raise NotImplementedError

    def sample(self):
        sample_num = int(self.sample_rate * self.client_num)
        self.sampled_clients = sorted(random.sample(self.clients, sample_num), key=lambda x: x.id)

    def downlink(self):
        assert (len(self.sampled_clients) > 0)
        for client in self.sampled_clients:
            client.clone_model(self)
            client.task_round = self.round

    def client_update(self):
        """Threaded client execution with per-device concurrency cap.
        max clients per device = self.max_per_device (1 => serialize on each GPU).
        """
        assert len(self.sampled_clients) > 0
        # normalize device list to non-empty
        devices = self.devices if isinstance(self.devices, list) and len(self.devices) > 0 else ['cpu']

        # build a semaphore per device
        device_semaphores = {device: Semaphore(self.max_per_device) for device in devices}

        def _run_one(client, device):
            semaphore = device_semaphores[device]
            semaphore.acquire()
            try:
                client.reset_optimizer()
                with OnDeviceRun(client, device) as c:
                    c.run()
                return getattr(client, 'training_time', 0.0)
            finally:
                semaphore.release()

        # choose a reasonable pool size: up to sum of per-device caps
        pool_workers = sum(self.max_per_device for _ in devices)
        from concurrent.futures import ThreadPoolExecutor, as_completed
        futures = []
        with ThreadPoolExecutor(max_workers=pool_workers) as ex:
            for idx, client in enumerate(self.sampled_clients):
                device = devices[idx % len(devices)]
                futures.append(ex.submit(_run_one, client, device))
            for f in as_completed(futures):
                _ = f.result()

        # keep original wall-clock accumulation rule
        self.wall_clock_time += max([client.training_time for client in self.sampled_clients])

    def uplink(self):
        assert (len(self.sampled_clients) > 0)

        def nan_to_zero(tensor):
            return torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)

        self.received_params = [nan_to_zero(client.model2shared_tensor()) for client in self.sampled_clients]

    def aggregate(self):
        assert (len(self.sampled_clients) > 0)
        total_samples = sum(len(client.dataset_train) for client in self.sampled_clients)
        weights = [len(client.dataset_train) / total_samples for client in self.sampled_clients]

        self.received_params = [params * weight for weight, params in zip(weights, self.received_params)]
        avg_tensor = sum(self.received_params)
        self.shared_tensor2model(avg_tensor)

    def test_all(self):
        """Evaluate all clients in parallel on available devices to reduce CPU load."""
        self.metric['acc'] = []

        # eval one client
        def _eval_one(client, device):
            client.clone_model(self)
            with torch.no_grad():
                with OnDeviceRun(client, device, 'eval') as c:
                    c.local_test()
            return c.metric['acc']

        max_workers = max(1, len(self.devices))
        acc_results = [None] * len(self.clients)

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = []
            for client in self.clients:
                idx = client.id
                device = self.devices[idx % len(self.devices)] if self.devices else 'cpu'
                futures.append((idx, ex.submit(_eval_one, client, device)))
            for idx, fut in futures:
                acc_results[idx] = fut.result()

        self.metric['acc'] = acc_results
        return {
            'acc'    : np.mean(self.metric['acc']),
            'acc_std': np.std(self.metric['acc']),
        }


def assert_device(deivce_arg, side='c'):
    if side == 's':
        if isinstance(deivce_arg, list):
            if deivce_arg:
                print(deivce_arg)
                assert max(
                        deivce_arg) < torch.cuda.device_count(), f'some device not available! only {torch.cuda.device_count()} cuda devices.'
                return deivce_arg
            return 'cpu'
        if isinstance(deivce_arg, int):
            assert deivce_arg < torch.cuda.device_count(), 'device not available!'
            return [deivce_arg]
        if deivce_arg == 'cpu':
            Warning('All clients work on CPU!')
            return deivce_arg
    elif side == 'c':  # client side
        # return deivce_arg[0] if isinstance(deivce_arg, list) and len(deivce_arg) == 1 else 'cpu'
        return 'cpu'
