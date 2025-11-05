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

    def __init__(self, model: list[torch.nn.Module] | torch.nn.Module, device: str | int):
        self.model = model
        self.target_device = device
        self.original_device = 'cpu'  # Always move back to CPU after exiting, as required

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
