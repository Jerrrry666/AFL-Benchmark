import heapq
import random
from enum import Enum

import numpy as np
import torch

from alg.base import BaseClient, BaseServer
from utils.run_utils import OnDevice


# Usage of Status
# During training, those training are set to Status.ACTIVE, the active clients will update staleness, and will not be sampled
# After aggregation, the aggregated is set to Status.IDLE

class Status(Enum):
    IDLE = 1
    ACTIVE = 2


class AsyncBaseClient(BaseClient):
    def __init__(self, id, args):
        super().__init__(id, args)
        self.gamma = args.gamma ** (1 / int(self.args.total_num * self.args.sr))  # adapt LR decay gamma for Async
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optim, gamma=self.gamma)

        self.status = Status.IDLE

    def run(self):
        raise NotImplementedError


class AsyncBaseServer(BaseServer):
    def __init__(self, args, clients):
        super().__init__(args, clients)
        self.decay = args.decay

        self.MAX_CONCURRENCY = int(self.client_num * self.sample_rate)
        self.client_queue = []
        # Remove global staleness array - staleness will be calculated on-demand
        # self.staleness = [0 for _ in self.clients]
        self.cur_client = None

    def run(self):
        raise NotImplementedError

    def get_staleness(self, client):
        """
        Calculate staleness for a client based on the difference between current server round
        and the round when client started training.
        
        Args:
            client: The client instance
            
        Returns:
            int: Staleness value (current_round - task_round)
        """
        if client.task_round is None:
            return 0
        return self.round - client.task_round

    def sample(self):
        active_num = len([c for c in self.clients if c.status == Status.ACTIVE])
        if active_num >= self.MAX_CONCURRENCY:
            return

        idle_clients = [c for c in self.clients if c.status != Status.ACTIVE]
        self.sampled_clients = random.sample(idle_clients, self.MAX_CONCURRENCY - active_num)
        # No need to reset staleness here - it will be calculated on-demand

    def downlink(self):
        for c in filter(lambda x: x.status != Status.ACTIVE, self.sampled_clients):
            c.clone_model(self)
            # Record the server round when client starts training
            c.task_round = self.round

    def client_update(self):
        for c in filter(lambda x: x.status != Status.ACTIVE, self.sampled_clients):
            c.model.train()
            c.reset_optimizer(True)
            with OnDevice(c.model, c.target_device):
                c.run()
            heapq.heappush(self.client_queue, (self.wall_clock_time + c.training_time, c))
            c.status = Status.ACTIVE

    def uplink(self):
        self.wall_clock_time, self.cur_client = heapq.heappop(self.client_queue)

    def aggregate(self):
        t_aggr = self.decay * self.cur_client.model2shared_tensor() + (1 - self.decay) * self.model2shared_tensor()
        self.shared_tensor2model(t_aggr)

    def update_status(self):
        # set the current client to idle
        self.cur_client.status = Status.IDLE

        # No need to update staleness here - it's calculated on-demand using get_staleness()
        # The staleness is now determined by the difference between current server round
        # and the round when client started training (stored in client.task_round)

    def test_all(self):
        self.metric['acc'] = []
        for client in self.clients:
            # NOTE: have to store current local model
            context = client.model2shared_tensor()
            client.clone_model(self)
            client.local_test()
            client.shared_tensor2model(context)
            self.metric['acc'].append(client.metric['acc'])

        return {
            'acc'    : np.mean(self.metric['acc']),
            'acc_std': np.std(self.metric['acc']),
        }
