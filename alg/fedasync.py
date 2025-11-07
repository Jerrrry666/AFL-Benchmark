from alg.asyncbase import AsyncBaseClient, AsyncBaseServer, compute_staleness_weight
from utils.run_utils import time_record


def add_args(parser):
    parser.add_argument('--a', type=int, default=1)
    parser.add_argument('--b', type=int, default=4)
    parser.add_argument('--strategy', type=str, default='hinge', help='constant/poly/hinge')
    return parser.parse_args()


class Client(AsyncBaseClient):
    @time_record
    def run(self):
        self.train()


class Server(AsyncBaseServer):
    def __init__(self, args, clients):
        super().__init__(args, clients)
        self.decay = args.decay

    def run(self):
        self.sample()
        self.downlink()
        self.client_update()
        self.uplink()
        self.aggregate()
        self.update_status()

    def aggregate(self):
        """批量聚合多个客户端的更新，支持FedAsync的staleness策略"""
        if not self.clients_to_aggregate:
            return
        
        # 计算加权聚合权重（考虑staleness策略和数据量）
        total_weight = 0
        weighted_params = None
        
        for client in self.clients_to_aggregate:
            # 使用FedAsync的staleness权重计算，固定参数值
            staleness = self.get_staleness(client)
            staleness_weight = compute_staleness_weight(staleness, strategy='hinge', a=1, b=4)
            
            # 考虑数据量权重
            data_weight = len(client.dataset_train) if client.dataset_train else 1.0
            
            # 组合权重
            combined_weight = staleness_weight * data_weight
            
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
