import numpy as np

from alg.asyncbase import AsyncBaseClient, AsyncBaseServer, Status
from utils.run_utils import time_record


def add_args(parser):
    parser.add_argument('--beta', type=float, default=0.5, help='staleness penalty factor')
    parser.add_argument('--b', type=float, default=10, help='determine whether to aggregate')
    return parser.parse_args()


class Client(AsyncBaseClient):
    def __init__(self, id, args):
        super().__init__(id, args)
        self.data_quality = 0

    def train(self):
        # === train ===
        total_loss = 0.0
        data_quality = 0.0

        for epoch in range(self.epoch):
            for data in self.loader_train:
                X, y = self.preprocess(data)
                preds = self.model(X)
                loss = self.loss_func(preds, y)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                total_loss += loss.item()
                data_quality += loss.item() ** 2

        self.metric['loss'] = total_loss / len(self.loader_train)
        self.metric['data_quality'] = data_quality / len(self.loader_train)

    @time_record
    def run(self):
        self.train()
        self.data_quality = len(self.dataset_train) * np.sqrt(self.metric['data_quality'])


class Server(AsyncBaseServer):
    def __init__(self, args, clients):
        super().__init__(args, clients)
        self.beta = args.beta
        self.b = args.b
        self.buffer = []
        self.AGGR = False
        self.last_aggr_time = 0

        self.staleness_history = [[] for _ in self.clients]
        self.profiled_latency = [0 for _ in self.clients]

        self.buffer = []


        
    def run(self):
        self.sample()
        self.downlink()
        self.client_update()
        self.uplink()
        self.aggregate()
        self.update_status()
    
    def sample(self):
        active_num = len([c for c in self.clients if c.status == Status.ACTIVE])
        if active_num >= self.MAX_CONCURRENCY:
            return

        idle_clients = [c for c in self.clients if c.status != Status.ACTIVE]

        for c in idle_clients:
            # Use get_staleness method to calculate current staleness
            current_staleness = self.get_staleness(c)
            avg_staleness = sum(self.staleness_history[c.id]) / len(self.staleness_history[c.id]) \
                if len(self.staleness_history[c.id]) > 0 else 0
            # Combine current staleness with historical average for sampling decision
            combined_staleness = (current_staleness + avg_staleness) / 2
            c.u = c.data_quality / pow(1 + combined_staleness, self.beta)
        self.sampled_clients = sorted(idle_clients, key=lambda c: c.u, reverse=True)[:self.MAX_CONCURRENCY - active_num]

        # No need to reset staleness here - it will be calculated on-demand

    
    def aggregate(self):
        self.buffer.append(self.cur_client.model2shared_tensor())
        
        slowest_time = max(self.profiled_latency)
        current_b = max(5, min(self.b, 15))  
        
        AGGR = (self.wall_clock_time - self.last_aggr_time > slowest_time / current_b)
        
        if AGGR and len(self.buffer) >= 3: 
            self.last_aggr_time = self.wall_clock_time
            self.AGGR = True
            self.shared_tensor2model(sum(self.buffer) / len(self.buffer))
        else:
            self.AGGR = False

    def update_status(self):
        # set the current client to idle
        self.cur_client.status = Status.IDLE

        # add staleness to historical information - use get_staleness instead of staleness array
        current_staleness = self.get_staleness(self.cur_client)
        self.staleness_history[self.cur_client.id].append(current_staleness)

        # profile latency
        self.profiled_latency[self.cur_client.id] = self.cur_client.training_time

        if self.AGGR:
            self.buffer.clear()
