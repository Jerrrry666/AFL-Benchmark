import heapq
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from threading import Semaphore

import numpy as np
import torch

from alg.base import BaseClient, BaseServer, assert_device
from utils.run_utils import OnDeviceRun


def compute_staleness_weight(staleness, strategy='hinge', a=1, b=4):
    """计算staleness权重，支持多种策略
    
    Args:
        staleness: staleness值
        strategy: 权重策略 ('constant', 'poly', 'hinge')
        a: poly策略的指数参数，hinge策略的分子参数
        b: hinge策略的分母参数
    
    Returns:
        float: staleness权重
    """
    if strategy == 'poly':
        return 1 / ((staleness + 1) ** abs(a))
    elif strategy == 'hinge':
        return 1 / (a * (staleness + b) + 1) if staleness > b else 1
    else:  # constant
        return 1


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

        # 新增多卡并行相关属性
        self.max_concurrent_per_device = getattr(args, 'max_concurrent_per_device', 2)
        self.aggregation_batch_size = getattr(args, 'aggregation_batch_size', 4)
        self.devices = assert_device(args.device, 's') if hasattr(args, 'device') else ['cpu']
        if isinstance(self.devices, str):
            self.devices = [self.devices]
        self.max_total_concurrent = len(self.devices) * self.max_concurrent_per_device
        self.pending_aggregation_queue = []  # 等待聚合的客户端队列
        self.clients_to_aggregate = []  # 当前批次要聚合的客户端

    def run(self):
        raise NotImplementedError
        # self.sample()
        # self.downlink()
        # self.client_update()
        # self.uplink()
        # self.aggregate()
        # self.update_status()

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
        """并发训练客户端，支持多GPU"""

        # 构建设备信号量，控制每个GPU的并发数
        device_semaphores = {}
        for device in self.devices:
            device_semaphores[device] = Semaphore(self.max_concurrent_per_device)

        def train_client_on_device(client, device):
            """在指定设备上训练客户端"""
            with device_semaphores[device]:
                try:
                    client.reset_optimizer(True)
                    with OnDeviceRun(client, device, 'train') as c:
                        c.run()
                    
                    # 确保模型回到CPU，避免设备不一致问题
                    client.model.to('cpu')
                    
                    # 训练完成后立即加入待聚合队列
                    heapq.heappush(self.pending_aggregation_queue,
                                   (self.wall_clock_time + client.training_time, client))
                    client.status = Status.ACTIVE
                    
                except Exception as e:
                    # 出错时也要确保模型回到CPU
                    client.model.to('cpu')
                    client.status = Status.IDLE
                    print(f"Client {client.id} training failed on device {device}: {e}")
                    raise e

        # 并发启动所有采样客户端的训练
        with ThreadPoolExecutor(max_workers=self.max_total_concurrent) as executor:
            futures = []
            for idx, client in enumerate(self.sampled_clients):
                if client.status != Status.ACTIVE:
                    device = self.devices[idx % len(self.devices)]
                    futures.append(executor.submit(train_client_on_device, client, device))

            # 等待所有训练任务完成（不阻塞，只是确保任务启动）
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Client training failed: {e}")

    def uplink(self):
        """获取一批已完成训练的客户端"""
        self.clients_to_aggregate = []

        # 从待聚合队列中获取一批客户端（按完成时间排序）
        batch_size = min(self.aggregation_batch_size, len(self.pending_aggregation_queue))

        for _ in range(batch_size):
            if self.pending_aggregation_queue:
                _, client = heapq.heappop(self.pending_aggregation_queue)
                self.clients_to_aggregate.append(client)

        # 更新墙钟时间到这批客户端的最晚完成时间
        if self.clients_to_aggregate:
            max_time = max(self.wall_clock_time + client.training_time for client in self.clients_to_aggregate)
            self.wall_clock_time = max(self.wall_clock_time, max_time)

    def aggregate(self):
        """批量聚合多个客户端的更新"""
        if not self.clients_to_aggregate:
            return

        # 确保所有客户端模型参数都在CPU上，避免设备不一致问题
        for client in self.clients_to_aggregate:
            client.model.to('cpu')

        # 计算加权聚合权重（考虑staleness和数据量）
        total_weight = 0
        weighted_params = None

        for client in self.clients_to_aggregate:
            # 计算staleness权重（基础权重）
            staleness = self.get_staleness(client)
            staleness_weight = 1.0 / (staleness + 1.0)  # staleness越大权重越小

            # 考虑数据量权重
            data_weight = len(client.dataset_train) if client.dataset_train else 1.0

            # 组合权重
            combined_weight = staleness_weight * data_weight

            # 现在可以安全地提取参数，确保模型在CPU上
            client_params = client.model2shared_tensor()

            if weighted_params is None:
                weighted_params = combined_weight * client_params
            else:
                weighted_params += combined_weight * client_params

            total_weight += combined_weight

        # 执行聚合
        if weighted_params is not None and total_weight > 0:
            aggregated_params = weighted_params / total_weight
            # 与服务器模型混合
            mixed_params = self.decay * aggregated_params + (1 - self.decay) * self.model2shared_tensor()
            self.shared_tensor2model(mixed_params)

    def update_status(self):
        """批量更新客户端状态"""
        for client in self.clients_to_aggregate:
            client.status = Status.IDLE

        # 清空当前批次的聚合列表
        self.clients_to_aggregate = []

    def test_all(self):
        """Evaluate all clients in parallel on available devices to reduce CPU load."""
        self.metric['acc'] = []

        # eval one client
        def _eval_one(client, device):
            # NOTE: have to store current local model
            context = client.model2shared_tensor()
            client.clone_model(self)
            with torch.no_grad():
                with OnDeviceRun(client, device, 'eval') as c:
                    c.local_test()
            # restore client's original model
            client.shared_tensor2model(context)
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
