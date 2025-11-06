import random
import time
from functools import wraps

import torch

from utils.sys_utils import system_config


def time_record(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        result = func(self, *args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        self.training_time = execution_time * self.delay

        # downlink and uplink
        comm_time = self.comm_bytes() * 8 / (1024 * 1024) / self.bandwidth
        self.training_time += comm_time * 2

        dropout = system_config()['dropout']
        if random.random() < dropout['drop_prob']:
            self.training_time += (random.random() * dropout['drop_latency'])
        return result

    return wrapper


class OnDevice:
    """
    A context manager that temporarily moves a model to a specified device
    and automatically moves it back to the CPU upon exit.

    Usage example:
        model = YourPytorchModel()
        with OnDevice(model, 'cuda:0'):
            # Inside this block, the model is on the 'cuda:0' device
            output = model(data.to('cuda:0'))
        # Outside this block, the model is automatically moved back to the 'cpu'
    """

    def __init__(self,
                 model: list[torch.nn.Module] | torch.nn.Module,
                 device: str | int,
                 model_in_cpu_flag: bool = False):
        self.model = model
        self.target_device = device
        self.original_device = 'cpu' if model_in_cpu_flag else device

    def __enter__(self):
        """When entering the with block, move the model to the target device."""
        if isinstance(self.model, list):
            for m in self.model:
                m.to(self.target_device)
        else:
            self.model.to(self.target_device)
        return self.model

    def __exit__(self, exc_type, exc_val, exc_tb):
        """When exiting the with block, move the model back to CPU, even if an exception occurs."""
        if isinstance(self.model, list):
            for m in self.model:
                m.to(self.original_device)
        else:
            self.model.to(self.original_device)


class OnDeviceRun:
    def __init__(self, client, device: str | int):
        self.client = client
        self.device = device

    def __enter__(self):
        """When entering the with block, move the model to the target device."""
        self.client.device = self.device
        self.client.model.to(self.device)
        self._optim_to(getattr(self.client, 'optim', None), self.device)
        self._scaler_to(getattr(self.client, 'scaler', None), self.device)
        return self.client

    def __exit__(self, exc_type, exc_val, exc_tb):
        """When exiting the with block, move the model back to CPU, even if an exception occurs."""
        self.client.device = None
        self.client.model.to('cpu')
        self._optim_to(getattr(self.client, 'optim', None), 'cpu')
        self._scaler_to(getattr(self.client, 'scaler', None), 'cpu')

    @staticmethod
    def _optim_to(optim: torch.optim.Optimizer | None, device: str | torch.device):
        """Move all tensor states inside optimizer to the target device."""
        if optim is None:
            return
        for state in optim.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device, copy=False)

    @staticmethod
    def _scaler_to(scaler, device: str | torch.device):
        """Move torch.cuda.amp.GradScaler state tensors as well (if present)."""
        if scaler is None:
            return
        # common GradScaler internals
        for attr in ("_scale", "_growth_tracker"):
            if hasattr(scaler, attr):
                t = getattr(scaler, attr)
                if isinstance(t, torch.Tensor):
                    setattr(scaler, attr, t.to(device, copy=False))
        # found_inf bookkeeping can be a dict[device]->Tensor
        if hasattr(scaler, "_found_inf_per_device") and isinstance(scaler._found_inf_per_device, dict):
            for d, t in scaler._found_inf_per_device.items():
                if isinstance(t, torch.Tensor):
                    scaler._found_inf_per_device[d] = t.to(device, copy=False)
