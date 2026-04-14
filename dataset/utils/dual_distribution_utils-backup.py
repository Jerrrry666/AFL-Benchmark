"""
通用双层分布控制工具
支持标签层面和特征层面的独立分布控制
适用于DomainNet等多维特征数据集
"""

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
from typing import Tuple, List, Dict, Any


def split_dual_distribution(data: np.ndarray, 
                          labels: np.ndarray, 
                          features: np.ndarray, 
                          config: Dict[str, Any]) -> Tuple[List, List, List]:
    """
    通用双层分布切分工具
    
    Args:
        data: 原始数据 [N, ...]
        labels: 标签分布 [N] (类别索引)
        features: 特征分布 [N] (风格等特征索引)  
        config: 配置参数字典
    
    Returns:
        X: 每个client的数据列表
        y: 每个client的标签列表
        statistic: 统计信息列表
    """
    num_clients = config['client_num']
    batch_size = config['batch_size']
    train_ratio = config['train_ratio']
    
    # 分布控制参数
    label_partition = config.get('label_partition', 'uni')  # 'uni' 或 'dir'
    label_alpha = config.get('label_alpha', 10000)  # 标签分布的alpha参数
    
    feature_partition = config.get('feature_partition', 'uni')  # 'uni', 'dir', 或 'single'
    feature_alpha = config.get('feature_alpha', 10000)  # 特征分布的alpha参数
    
    num_classes = len(np.unique(labels))
    num_features = len(np.unique(features))
    
    print(f"Dataset info: {len(data)} samples, {num_classes} classes, {num_features} features")
    print(f"Label distribution: {label_partition} (alpha={label_alpha})")
    print(f"Feature distribution: {feature_partition} (alpha={feature_alpha})")
    
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    f = [[] for _ in range(num_clients)]
    statistic = [[] for _ in range(num_clients)]
    
    # 确保最小样本数
    least_samples = int(min(batch_size / (1 - train_ratio), len(labels) / num_clients / 2))
    
    # 创建数据索引映射
    dataidx_map = {}
    
    if feature_partition == 'single':
        # 极致异质：每个client只持有1种风格
        dataidx_map = _split_single_feature(data, labels, features, num_clients, num_features)
        
    elif feature_partition == 'dir':
        # 特征层面狄利克雷分布
        dataidx_map = _split_feature_dirichlet(data, labels, features, num_clients, feature_alpha)
        
    elif feature_partition == 'uni':  # feature_partition == 'uni'
        # 特征层面平均分布
        dataidx_map = _split_feature_uniform(data, labels, features, num_clients)
    else:
        raise ValueError(f"Unknown feature_partition: {feature_partition}")
    
    # 在特征分布的基础上应用标签分布
    dataidx_map = _apply_label_distribution(dataidx_map, labels, num_clients, num_classes, 
                                          label_partition, label_alpha, least_samples)
    
    # 分配数据
    for client in range(num_clients):
        if client in dataidx_map:
            idxs = dataidx_map[client]
            X[client] = data[idxs]
            y[client] = labels[idxs]  
            f[client] = features[idxs]
            
            # 统计信息
            unique_labels, counts = np.unique(y[client], return_counts=True)
            for label, count in zip(unique_labels, counts):
                statistic[client].append((int(label), int(count)))
    
    # 打印统计信息
    _print_client_statistics(X, y, f, statistic, num_clients)
    
    return X, y, statistic


def _split_single_feature(data: np.ndarray, 
                         labels: np.ndarray, 
                         features: np.ndarray, 
                         num_clients: int, 
                         num_features: int) -> Dict[int, List[int]]:
    """每个client只持有1种风格"""
    dataidx_map = {}
    
    # 为每个client分配一个特征类型
    feature_per_client = num_features // num_clients
    if feature_per_client == 0:
        feature_per_client = 1
    
    clients_per_feature = num_clients // num_features
    
    feature_idx = 0
    client_idx = 0
    
    for feature_id in range(num_features):
        # 找到该特征类型的所有样本
        feature_mask = features == feature_id
        feature_indices = np.where(feature_mask)[0]
        
        # 将这些样本分配给下一个可用的clients
        for _ in range(min(clients_per_feature, num_clients - client_idx)):
            dataidx_map[client_idx] = feature_indices.tolist()
            client_idx += 1
            if client_idx >= num_clients:
                break
        
        if client_idx >= num_clients:
            break
    
    return dataidx_map


def _split_feature_dirichlet(data: np.ndarray, 
                           labels: np.ndarray, 
                           features: np.ndarray, 
                           num_clients: int, 
                           alpha: float) -> Dict[int, List[int]]:
    """特征层面狄利克雷分布"""
    dataidx_map = {i: [] for i in range(num_clients)}
    num_features = len(np.unique(features))
    
    for feature_id in range(num_features):
        feature_mask = features == feature_id
        feature_indices = np.where(feature_mask)[0]
        np.random.shuffle(feature_indices)
        
        # 使用狄利克雷分布分配特征到clients
        proportions = np.random.dirichlet([alpha] * num_clients)
        split_points = (np.cumsum(proportions) * len(feature_indices)).astype(int)[:-1]
        split_idxs = np.split(feature_indices, split_points)
        
        for client_id, idxs in enumerate(split_idxs):
            dataidx_map[client_id].extend(idxs.tolist())
    
    return dataidx_map


def _split_feature_uniform(data: np.ndarray, 
                         labels: np.ndarray, 
                         features: np.ndarray, 
                         num_clients: int) -> Dict[int, List[int]]:
    """特征层面平均分布"""
    dataidx_map = {i: [] for i in range(num_clients)}
    num_features = len(np.unique(features))
    
    for feature_id in range(num_features):
        feature_mask = features == feature_id
        feature_indices = np.where(feature_mask)[0]
        np.random.shuffle(feature_indices)
        
        # 平均分配每个特征到所有clients
        split_size = len(feature_indices) // num_clients
        remainder = len(feature_indices) % num_clients
        
        start_idx = 0
        for client_id in range(num_clients):
            end_idx = start_idx + split_size + (1 if client_id < remainder else 0)
            dataidx_map[client_id].extend(feature_indices[start_idx:end_idx].tolist())
            start_idx = end_idx
    
    return dataidx_map


def _apply_label_distribution(dataidx_map: Dict[int, List[int]], 
                            labels: np.ndarray, 
                            num_clients: int, 
                            num_classes: int, 
                            label_partition: str, 
                            label_alpha: float, 
                            least_samples: int) -> Dict[int, List[int]]:
    """在特征分布的基础上应用标签分布"""
    
    if label_partition == 'dir':
        # 标签层面狄利克雷分布
        return _apply_label_dirichlet(dataidx_map, labels, num_clients, num_classes, label_alpha, least_samples)
    else:  # label_partition == 'uni'
        # 标签层面平均分布
        return _apply_label_uniform(dataidx_map, labels, num_clients, num_classes)


def _apply_label_dirichlet(dataidx_map: Dict[int, List[int]], 
                         labels: np.ndarray, 
                         num_clients: int, 
                         num_classes: int, 
                         alpha: float, 
                         least_samples: int) -> Dict[int, List[int]]:
    """应用标签层面狄利克雷分布"""
    new_dataidx_map = {i: [] for i in range(num_clients)}
    
    for class_id in range(num_classes):
        # 找到该类别的所有样本
        class_mask = labels == class_id
        class_indices = np.where(class_mask)[0]
        
        # 为每个class创建client分配比例
        proportions = np.random.dirichlet([alpha] * num_clients)
        split_points = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        split_idxs = np.split(class_indices, split_points)
        
        # 分配到clients
        for client_id, idxs in enumerate(split_idxs):
            new_dataidx_map[client_id].extend(idxs.tolist())
    
    # 移除样本数过少的client
    min_size = least_samples
    filtered_map = {}
    for client_id, idxs in new_dataidx_map.items():
        if len(idxs) >= min_size:
            filtered_map[client_id] = idxs
    
    # 如果过滤后client数量太少，重新分配
    if len(filtered_map) < num_clients // 2:
        # 简化分配策略
        filtered_map = {i: [] for i in range(num_clients)}
        for class_id in range(num_classes):
            class_mask = labels == class_id
            class_indices = np.where(class_mask)[0]
            np.random.shuffle(class_indices)
            
            split_size = len(class_indices) // num_clients
            remainder = len(class_indices) % num_clients
            
            start_idx = 0
            for client_id in range(num_clients):
                end_idx = start_idx + split_size + (1 if client_id < remainder else 0)
                filtered_map[client_id].extend(class_indices[start_idx:end_idx].tolist())
                start_idx = end_idx
    
    return filtered_map


def _apply_label_uniform(dataidx_map: Dict[int, List[int]], 
                       labels: np.ndarray, 
                       num_clients: int, 
                       num_classes: int) -> Dict[int, List[int]]:
    """应用标签层面平均分布"""
    new_dataidx_map = {i: [] for i in range(num_clients)}
    
    for class_id in range(num_classes):
        # 找到该类别的所有样本
        class_mask = labels == class_id
        class_indices = np.where(class_mask)[0]
        np.random.shuffle(class_indices)
        
        # 平均分配到所有clients
        split_size = len(class_indices) // num_clients
        remainder = len(class_indices) % num_clients
        
        start_idx = 0
        for client_id in range(num_clients):
            end_idx = start_idx + split_size + (1 if client_id < remainder else 0)
            new_dataidx_map[client_id].extend(class_indices[start_idx:end_idx].tolist())
            start_idx = end_idx
    
    return new_dataidx_map


def _print_client_statistics(X: List, y: List, f: List, statistic: List, num_clients: int):
    """打印client统计信息"""
    print("\n" + "="*80)
    print("CLIENT DISTRIBUTION STATISTICS")
    print("="*80)
    
    for client in range(num_clients):
        if len(X[client]) > 0:
            print(f"Client {client}:")
            print(f"  Data size: {len(X[client])}")
            print(f"  Labels: {np.unique(y[client])}")
            print(f"  Features: {np.unique(f[client])}")
            print(f"  Label distribution: {[(int(label), int(count)) for label, count in statistic[client]]}")
            print("-" * 40)
        else:
            print(f"Client {client}: No data assigned")
            print("-" * 40)
    
    print("="*80)
