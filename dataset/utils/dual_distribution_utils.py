"""
通用双层分布控制工具
支持标签层面和特征层面的独立分布控制
适用于DomainNet等多维特征数据集
"""

import json
from typing import Any, Dict, List, Tuple

import numpy as np


def split_dual_distribution(
    data: np.ndarray, labels: np.ndarray, features: np.ndarray, config: Dict[str, Any]
) -> Tuple[List, List, Dict]:
    """
    通用双层分布切分工具

    Args:
        data: 原始数据 [N, ...]
        labels: 标签分布 [N] (类别索引)
        features: 特征分布 [N] (风格等特征索引)
        config: 配置参数字典（可包含 verbose 参数控制是否打印详细信息）

    Returns:
        X: 每个client的数据列表
        y: 每个client的标签列表
        statistic: 详细统计信息字典，包含label和feature分布
    """
    num_clients = config["client_num"]
    batch_size = config["batch_size"]
    train_ratio = config["train_ratio"]
    verbose = config.get("verbose", False)  # 从config中读取verbose参数

    # 分布控制参数
    label_partition = config.get("label_partition", "uni")  # 'uni' 或 'dir'
    label_alpha = config.get("label_alpha", 10000)  # 标签分布的alpha参数

    feature_partition = config.get("feature_partition", "uni")  # 'uni', 'dir', 或 'pat'
    feature_p = config.get("feature_p", 2)  # pat分布中每个client的特征种类数
    feature_pat_mode = config.get(
        "feature_pat_mode", "proportional"
    )  # pat分布的分配模式：'proportional' 或 'uniform'

    # feature_alpha只在特征狄利克雷分布时需要
    feature_alpha = None
    if feature_partition == "dir":
        feature_alpha = config.get("feature_alpha", 10000)  # 特征分布的alpha参数
    elif feature_partition == "pat":
        feature_alpha = config.get(
            "feature_alpha", 0.5
        )  # pat分布内部样本分配的alpha参数

    num_classes = len(np.unique(labels))
    num_features = len(np.unique(features))

    print(
        f"Dataset info: {len(data)} samples, {num_classes} classes, {num_features} features"
    )
    print(f"Label distribution: {label_partition} (alpha={label_alpha})")
    print(f"Feature distribution: {feature_partition}")
    if feature_alpha is not None:
        print(f"Feature alpha: {feature_alpha}")
    if feature_partition == "pat":
        print(f"Pathological distribution: each client has {feature_p} features")
        print(f"Pathological allocation mode: {feature_pat_mode}")

    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    f = [[] for _ in range(num_clients)]

    # 详细统计信息结构
    statistic = {
        "label_distribution": [[] for _ in range(num_clients)],
        "feature_distribution": [[] for _ in range(num_clients)],
        "client_stats": [{} for _ in range(num_clients)],
    }

    # 确保最小样本数
    least_samples = int(
        min(batch_size / (1 - train_ratio), len(labels) / num_clients / 2)
    )

    # 创建数据索引映射
    dataidx_map = {}

    if feature_partition == "pat":
        # Pathological分布：每个client只有p种特征
        # 首先生成特征拓扑结构
        topology = _generate_pathological_topology(
            num_clients, num_features, features, feature_p, feature_pat_mode
        )
        # 然后在特征约束下分配样本和标签
        dataidx_map = _distribute_pathological_samples(
            topology,
            labels,
            features,
            num_clients,
            num_classes,
            label_partition,
            label_alpha,
        )

    elif feature_partition == "dir":
        # 特征层面狄利克雷分布
        dataidx_map = _split_feature_dirichlet(
            data, labels, features, num_clients, feature_alpha
        )

    elif feature_partition == "uni":  # feature_partition == 'uni'
        # 特征层面平均分布
        dataidx_map = _split_feature_uniform(data, labels, features, num_clients)
    else:
        raise ValueError(f"Unknown feature_partition: {feature_partition}")

    # 只有在非pat模式下才需要额外的标签分布处理
    # pat模式下标签分布已经在_distribute_pathological_samples中处理过了
    if feature_partition != "pat":
        dataidx_map = _apply_label_distribution(
            dataidx_map,
            labels,
            num_clients,
            num_classes,
            label_partition,
            label_alpha,
            least_samples,
            feature_partition,
            features,
            dataidx_map,
        )

    # 分配数据并收集统计信息
    for client in range(num_clients):
        if client in dataidx_map:
            idxs = dataidx_map[client]
            X[client] = data[idxs]
            y[client] = labels[idxs]
            f[client] = features[idxs]

            # 标签分布统计
            unique_labels, label_counts = np.unique(y[client], return_counts=True)
            for label, count in zip(unique_labels, label_counts):
                statistic["label_distribution"][client].append(
                    {"label": int(label), "count": int(count)}
                )

            # 特征分布统计
            unique_features, feature_counts = np.unique(f[client], return_counts=True)
            for feature, count in zip(unique_features, feature_counts):
                statistic["feature_distribution"][client].append(
                    {"feature": int(feature), "count": int(count)}
                )

            # 客户端总体统计
            statistic["client_stats"][client] = {
                "total_samples": len(X[client]),
                "num_labels": len(unique_labels),
                "num_features": len(unique_features),
                "unique_labels": unique_labels.tolist(),
                "unique_features": unique_features.tolist(),
            }

    # 打印详细统计信息
    _print_detailed_statistics(X, y, f, statistic, num_clients, verbose)

    # 如果是pat分布，额外打印全局分布信息
    if feature_partition == "pat":
        _print_pathological_global_distribution(f, num_features, num_clients, verbose)

    return X, y, statistic


def _generate_pathological_topology(
    num_clients: int,
    num_features: int,
    features: np.ndarray,
    feature_p: int,
    mode: str = "uniform",
) -> Dict[int, List[int]]:
    """
    生成Pathological分布的特征拓扑结构（即每个client拥有哪些特征）
    """
    # 统计每个特征的样本数量
    feature_sample_counts = {}
    for feature_id in range(num_features):
        feature_mask = features == feature_id
        feature_sample_counts[feature_id] = np.sum(feature_mask)

    # 计算每个特征应该被多少个client覆盖
    total_samples = sum(feature_sample_counts.values())
    feature_coverage = {}

    if mode == "proportional":
        # 智能比例分配：大样本域分配更多client
        for feature_id, sample_count in feature_sample_counts.items():
            coverage = max(1, round((sample_count / total_samples) * num_clients))
            feature_coverage[feature_id] = coverage
    else:  # mode == 'uniform'
        # 均匀分配：每个特征分配相同数量的client
        avg_coverage = max(1, num_clients // num_features)
        for feature_id in range(num_features):
            feature_coverage[feature_id] = avg_coverage

    # 为每个特征分配client
    feature_client_assignment = {}
    for feature_id in range(num_features):
        # 随机选择coverage个client
        available_clients = list(range(num_clients))
        np.random.shuffle(available_clients)
        assigned_clients = available_clients[: feature_coverage[feature_id]]
        feature_client_assignment[feature_id] = assigned_clients

    # 为每个client分配特征
    client_features = {i: [] for i in range(num_clients)}
    for feature_id, clients in feature_client_assignment.items():
        for client_id in clients:
            client_features[client_id].append(feature_id)

    # 为每个client随机选择p个特征
    for client_id in range(num_clients):
        if len(client_features[client_id]) > feature_p:
            # 如果特征过多，随机选择p个
            available_features = client_features[client_id]
            np.random.shuffle(available_features)
            client_features[client_id] = available_features[:feature_p]
        elif len(client_features[client_id]) < feature_p:
            # 如果特征不足，从未分配的特征中补充
            current_features = set(client_features[client_id])
            unassigned_features = [
                f for f in range(num_features) if f not in current_features
            ]
            if unassigned_features:
                np.random.shuffle(unassigned_features)
                needed = feature_p - len(client_features[client_id])
                client_features[client_id].extend(unassigned_features[:needed])

    return client_features


def _distribute_pathological_samples(
    client_features: Dict[int, List[int]],
    labels: np.ndarray,
    features: np.ndarray,
    num_clients: int,
    num_classes: int,
    label_partition: str,
    label_alpha: float,
) -> Dict[int, List[int]]:
    """
    根据特征拓扑和标签分布策略分配样本
    在这里实现了真正的双层分布：
    1. 首先确保每个client只有指定的feature
    2. 然后在feature约束下按label_partition策略分配标签
    """
    dataidx_map = {i: [] for i in range(num_clients)}

    # 反向映射：特征 -> 拥有该特征的Clients
    feature_to_clients = {}
    for client_id, feats in client_features.items():
        for f_id in feats:
            if f_id not in feature_to_clients:
                feature_to_clients[f_id] = []
            feature_to_clients[f_id].append(client_id)

    num_features = len(np.unique(features))

    for feature_id in range(num_features):
        if feature_id not in feature_to_clients:
            continue

        clients_with_feature = feature_to_clients[feature_id]
        if not clients_with_feature:
            continue

        # 获取该特征的所有样本索引
        feature_mask = features == feature_id
        feature_indices = np.where(feature_mask)[0]

        # 按类别分配
        for class_id in range(num_classes):
            class_mask = labels[feature_indices] == class_id
            class_indices_in_feature = feature_indices[class_mask]

            if len(class_indices_in_feature) == 0:
                continue

            np.random.shuffle(class_indices_in_feature)

            # 确定分配比例
            if label_partition == "dir":
                proportions = np.random.dirichlet(
                    [label_alpha] * len(clients_with_feature)
                )
            else:  # 'uni'
                proportions = np.array(
                    [1.0 / len(clients_with_feature)] * len(clients_with_feature)
                )

            # 计算分割点
            split_points = (
                np.cumsum(proportions) * len(class_indices_in_feature)
            ).astype(int)[:-1]
            split_idxs = np.split(class_indices_in_feature, split_points)

            # 分配给Client
            for i, client_id in enumerate(clients_with_feature):
                dataidx_map[client_id].extend(split_idxs[i].tolist())

    return dataidx_map


def _split_feature_dirichlet(
    data: np.ndarray,
    labels: np.ndarray,
    features: np.ndarray,
    num_clients: int,
    alpha: float,
) -> Dict[int, List[int]]:
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


def _split_feature_uniform(
    data: np.ndarray, labels: np.ndarray, features: np.ndarray, num_clients: int
) -> Dict[int, List[int]]:
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


def _apply_label_distribution(
    dataidx_map: Dict[int, List[int]],
    labels: np.ndarray,
    num_clients: int,
    num_classes: int,
    label_partition: str,
    label_alpha: float,
    least_samples: int,
    feature_partition: str,
    features: np.ndarray,
    original_dataidx_map: Dict[int, List[int]],
) -> Dict[int, List[int]]:
    """在特征分布的基础上应用标签分布"""

    if label_partition == "dir":
        # 标签层面狄利克雷分布
        # 其他情况使用全局dirichlet分布
        return _apply_label_dirichlet(
            dataidx_map, labels, num_clients, num_classes, label_alpha, least_samples
        )
    else:  # label_partition == 'uni'
        # 如果已经进行了特征层面的分配（dataidx_map不为空），且要求标签uni，
        # 则应保留现有的特征分配，而不是重新进行全局uni分配。
        if (
            dataidx_map
            and len(dataidx_map) > 0
            and any(len(v) > 0 for v in dataidx_map.values())
        ):
            return dataidx_map

        # 标签层面平均分布
        return _apply_label_uniform(dataidx_map, labels, num_clients, num_classes)


def _apply_label_dirichlet(
    dataidx_map: Dict[int, List[int]],
    labels: np.ndarray,
    num_clients: int,
    num_classes: int,
    alpha: float,
    least_samples: int,
) -> Dict[int, List[int]]:
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
                filtered_map[client_id].extend(
                    class_indices[start_idx:end_idx].tolist()
                )
                start_idx = end_idx

    return filtered_map


def _apply_label_uniform(
    dataidx_map: Dict[int, List[int]],
    labels: np.ndarray,
    num_clients: int,
    num_classes: int,
) -> Dict[int, List[int]]:
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


def _print_detailed_statistics(
    X: List, y: List, f: List, statistic: Dict, num_clients: int, verbose: bool = False
):
    """打印详细的client统计信息"""

    if verbose:
        print("\n" + "=" * 80)
        print("DETAILED CLIENT DISTRIBUTION STATISTICS")
        print("=" * 80)

        for client in range(num_clients):
            if len(X[client]) > 0:
                print(f"Client {client}:")
                print(f"  📊 Total samples: {len(X[client])}")
                print(
                    f"  🏷️  Unique labels: {statistic['client_stats'][client]['num_labels']} (IDs: {statistic['client_stats'][client]['unique_labels']})"
                )
                print(
                    f"  🎨 Unique features: {statistic['client_stats'][client]['num_features']} (IDs: {statistic['client_stats'][client]['unique_features']})"
                )

                # 标签分布详情
                print(f"  📈 Label distribution:")
                for label_info in statistic["label_distribution"][client]:
                    print(
                        f"     Label {label_info['label']}: {label_info['count']} samples"
                    )

                # 特征分布详情
                print(f"  🎯 Feature distribution:")
                for feature_info in statistic["feature_distribution"][client]:
                    print(
                        f"     Feature {feature_info['feature']}: {feature_info['count']} samples"
                    )

                print("-" * 60)
            else:
                print(f"Client {client}: ❌ No data assigned")
                print("-" * 60)

        print("=" * 80)

    # 全局统计摘要
    total_samples = sum(stat["total_samples"] for stat in statistic["client_stats"])
    print(f"\n📊 GLOBAL SUMMARY:")
    print(f"   Total samples across all clients: {total_samples}")
    print(f"   Average samples per client: {total_samples / num_clients:.1f}")
    print(
        f"   Number of clients with data: {sum(1 for stat in statistic['client_stats'] if stat['total_samples'] > 0)}"
    )
    print("=" * 80)


def _print_pathological_global_distribution(
    f: List, num_features: int, num_clients: int, verbose: bool = False
):
    """打印Pathological分布的全局信息"""
    if not verbose:
        return

    print("\n" + "=" * 80)
    print("PATHOLOGICAL DISTRIBUTION GLOBAL VIEW")
    print("=" * 80)

    # 统计每个特征被哪些client拥有
    feature_clients = {}
    for feature_id in range(num_features):
        feature_clients[feature_id] = []

    for client_id in range(num_clients):
        if len(f[client_id]) > 0:
            unique_features = np.unique(f[client_id])
            for feature_id in unique_features:
                if feature_id not in feature_clients:
                    feature_clients[feature_id] = []
                feature_clients[feature_id].append(client_id)

    # 打印每个特征的分布
    print("🔍 FEATURE VIEW (which clients have each feature):")
    for feature_id in range(num_features):
        clients_with_feature = feature_clients.get(feature_id, [])
        print(
            f"Feature {feature_id}: covered by {len(clients_with_feature)} clients -> {clients_with_feature}"
        )

    print("\n🔍 CLIENT VIEW (which features does each client have):")
    # 打印每个client拥有哪些特征
    for client_id in range(num_clients):
        if len(f[client_id]) > 0:
            unique_features = np.unique(f[client_id])
            print(
                f"Client {client_id}: has {len(unique_features)} features -> {unique_features.tolist()}"
            )
        else:
            print(f"Client {client_id}: ❌ No data assigned")

    print("=" * 80)


def save_detailed_statistics(
    statistic: Dict, save_path: str, domain_names: List[str] = None
):
    """
    保存详细统计信息到文件

    Args:
        statistic: 统计信息字典
        save_path: 保存路径
        domain_names: 域名列表（用于特征ID到名称的映射）
    """
    # 如果提供了域名列表，转换特征ID为名称
    if domain_names:
        for client_stats in statistic["feature_distribution"]:
            for feature_info in client_stats:
                if feature_info["feature"] < len(domain_names):
                    feature_info["feature_name"] = domain_names[feature_info["feature"]]

    # 保存为JSON格式
    json_path = save_path.replace(".yaml", ".json").replace(".yml", ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(statistic, f, indent=2, ensure_ascii=False)

    # 保存为YAML格式
    yaml_path = save_path
    try:
        import yaml

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(
                statistic, f, default_flow_style=False, allow_unicode=True, indent=2
            )
    except ImportError:
        print("Warning: yaml module not available, skipping YAML export")

    print(f"📁 Detailed statistics saved to:")
    print(f"   JSON: {json_path}")
    if domain_names:
        print(f"   YAML: {yaml_path}")

    return json_path, yaml_path if domain_names else None
