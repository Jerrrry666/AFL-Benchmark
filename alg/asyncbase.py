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
    ERROR = 3


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

        self.max_concurrent_per_device = getattr(args, 'max_concurrent_per_device', 2)
        self.aggregation_batch_size = getattr(args, 'aggregation_batch_size', 1)
        self.devices = assert_device(args.device, 's') if hasattr(args, 'device') else ['cpu']
        if isinstance(self.devices, str):
            self.devices = [self.devices]
        self.test_devices = assert_device(args.test_device, 's') if hasattr(args, 'test_device') else ['cpu']
        if isinstance(self.test_devices, str):
            self.test_devices = [self.test_devices]
        self.max_total_concurrent = len(self.devices) * self.max_concurrent_per_device
        self.pending_aggregation_queue = []  # 等待聚合的客户端队列
        self.clients_to_aggregate = []

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
        """并发训练客户端，支持多GPU，集成训练-测试分离"""

        # 构建设备信号量，控制每个GPU的并发数
        device_semaphores = {}
        for device in self.devices:
            device_semaphores[device] = Semaphore(self.max_concurrent_per_device)

        def train_client_on_device(client, device):
            """在指定设备上训练客户端"""
            with device_semaphores[device]:
                try:
                    # 注意：client.status应该在调用此函数前已经设置为ACTIVE
                    # 这里只负责训练和缓存，不负责状态管理

                    client.reset_optimizer(True)

                    # 缓存个性化参数用于测试（训练开始前的状态）
                    # 使用model2personalized_tensor保存个性化参数，更高效
                    client.cached_personalized_params = client.model2personalized_tensor()

                    with OnDeviceRun(client, device, 'train') as c:
                        c.run()

                    # 训练完成后立即加入待聚合队列（此时client仍然是ACTIVE状态）
                    heapq.heappush(self.pending_aggregation_queue,
                                   (self.wall_clock_time + client.training_time, client))

                except Exception as e:
                    client.status = Status.ERROR
                    # client.model.to('cpu')
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

        # todo 半异步下，如何计算权重？
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
        """批量更新客户端状态，在聚合完成后才设为IDLE"""
        # 只有在聚合完成后，才将这些客户端设为IDLE
        for client in self.clients_to_aggregate:
            client.status = Status.IDLE
            # 清理个性化参数缓存以节省内存
            if hasattr(client, 'cached_personalized_params'):
                client.cached_personalized_params = None

        # 清空当前批次的聚合列表
        self.clients_to_aggregate = []

    def test_all(self):
        """使用个性化参数的并行测试，避免影响正在训练的客户端"""
        self.metric['acc'] = []

        def _eval_one(client, device):
            """测试单个客户端，根据状态选择测试策略"""
            if client.status == Status.ACTIVE and hasattr(client,
                                                          'cached_personalized_params') and client.cached_personalized_params is not None:
                # 客户端正在训练中，使用缓存的个性化参数恢复测试模型状态
                with torch.no_grad():
                    # 创建测试模型：服务器模型 + 个性化参数
                    import copy
                    test_model = copy.deepcopy(client.model)  # 当前服务器模型状态
                    test_model.to('cpu')  # 确保在CPU上

                    # 保存当前个性化参数
                    current_personalized = client.model2personalized_tensor()

                    # 应用缓存的个性化参数到测试模型
                    if client.cached_personalized_params is not None:
                        client.personalized_tensor2model(client.cached_personalized_params)

                    # 使用恢复的测试模型进行测试
                    test_model.eval()
                    correct = 0
                    total = 0

                    for data in client.loader_test:
                        X, y = client.preprocess(data)
                        preds = test_model(X)
                        _, preds_y = torch.max(preds.data, 1)
                        total += y.size(0)
                        correct += (preds_y == y).sum().item()

                    client.metric['acc'] = 100.00 * correct / total

                    # 恢复当前个性化参数
                    if current_personalized is not None:
                        client.personalized_tensor2model(current_personalized)

                    # 清理临时测试模型
                    del test_model

                    return client.metric['acc']
            else:
                # 客户端不在训练中，使用正常测试流程
                return _test_client_normal(self, client, device)

        def _test_client_normal(self, client, device):
            """正常的客户端测试流程"""
            context = client.model2shared_tensor()
            client.clone_model(self)
            with OnDeviceRun(client, device, 'eval') as c:
                c.local_test()
            client.shared_tensor2model(context)
            return client.metric['acc']

        # 并行测试所有客户端
        max_workers = max(1, len(self.test_devices))
        acc_results = [None] * len(self.clients)

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = []
            for client in self.clients:
                idx = client.id
                device = self.test_devices[idx % len(self.test_devices)] if self.test_devices else 'cpu'
                futures.append((idx, ex.submit(_eval_one, client, device)))

            for idx, fut in futures:
                acc_results[idx] = fut.result()

        self.metric['acc'] = acc_results
        return {
            'acc'    : np.mean(self.metric['acc']),
            'acc_std': np.std(self.metric['acc']),
        }
