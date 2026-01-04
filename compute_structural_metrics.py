#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_structural_metrics.py

计算聚类的结构指标（不仅仅是 C_v）：
- Silhouette coefficient
- kNN mixing（跨板块邻居比例）  
- 拓扑稳定性（seed 鲁棒性）

这些指标比 C_v 更能说明"拓扑分裂/桥梁/孤岛"
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from sklearn.metrics import silhouette_score, silhouette_samples
import sys
import json

sys.path.insert(0, str(Path(__file__).resolve().parent))


def compute_silhouette(embeddings: np.ndarray, labels: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    计算 Silhouette coefficient
    
    范围 [-1, 1]：
      1.0  = 簇非常紧凑，样本与簇内其他样本相似，与其他簇不同
      0.0  = 样本在簇边界上，或多个簇的聚类无效
     -1.0  = 样本被分配到错误的簇
    
    Returns:
      overall_score: 全局 silhouette score
      sample_scores: 每个样本的 silhouette score
    """
    # 计算全局 score
    overall = silhouette_score(embeddings, labels)
    
    # 计算每个样本的 score
    samples = silhouette_samples(embeddings, labels)
    
    return overall, samples


def compute_knn_mixing(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k: int = 20
) -> Dict[str, Any]:
    """
    计算 k-NN mixing：在 k 个最近邻中，有多少比例来自其他簇
    
    定义：
      kNN_mixing = (跨簇邻居数) / (总邻居数 k)
    
    解释：
      mixing = 0    : 聚类完美（邻居都来自同簇）
      mixing = 1    : 完全混乱（邻居都来自其他簇）
      mixing = 0.5  : 聚类有问题
    """
    n_samples = embeddings.shape[0]
    n_clusters = len(np.unique(labels))
    
    # 计算欧氏距离矩阵（用于 kNN）
    # 对于大数据集，应该用 scipy.spatial.distance.cdist
    from scipy.spatial.distance import cdist
    distances = cdist(embeddings, embeddings, metric='euclidean')
    
    # 对每个样本找 k 个最近邻（包括自己，所以是 k+1）
    knn_indices = np.argsort(distances, axis=1)[:, :k+1]
    
    # 计算 mixing
    mixing_rates = []
    for i in range(n_samples):
        neighbors = knn_indices[i, 1:]  # 排除自己
        neighbor_labels = labels[neighbors]
        my_label = labels[i]
        
        # 有多少邻居来自其他簇
        cross_cluster = np.sum(neighbor_labels != my_label)
        mixing_rate = cross_cluster / k
        mixing_rates.append(mixing_rate)
    
    mixing_rates = np.array(mixing_rates)
    
    return {
        "mean_mixing": float(np.mean(mixing_rates)),
        "std_mixing": float(np.std(mixing_rates)),
        "min_mixing": float(np.min(mixing_rates)),
        "max_mixing": float(np.max(mixing_rates)),
        "purity": float(1 - np.mean(mixing_rates)),  # 反指标：纯度
        "details": {
            "k": k,
            "n_clusters": n_clusters,
            "interpretation": "mixing 越低越好（接近 0），表示聚类越纯"
        }
    }


def compute_cluster_isolation(
    embeddings: np.ndarray,
    labels: np.ndarray
) -> Dict[str, Any]:
    """
    计算簇的隔离度：簇内距离 vs 簇间距离
    
    inter-intra distance ratio = mean_inter_cluster_dist / mean_intra_cluster_dist
    
    解释：
      ratio > 1   : 簇之间距离大于簇内距离（聚类好）
      ratio ≈ 1   : 簇之间距离接近簇内距离（聚类有问题）
      ratio < 1   : 簇之间距离小于簇内距离（聚类很差）
    """
    unique_labels = np.unique(labels)
    
    # 计算簇内距离（intra-cluster）
    intra_distances = []
    for label in unique_labels:
        mask = labels == label
        cluster_points = embeddings[mask]
        if len(cluster_points) > 1:
            # 计算这个簇内所有点之间的平均距离
            from scipy.spatial.distance import pdist
            dists = pdist(cluster_points, metric='euclidean')
            intra_distances.extend(dists)
    
    mean_intra = np.mean(intra_distances) if intra_distances else 0
    
    # 计算簇间距离（inter-cluster）
    inter_distances = []
    for i, label_i in enumerate(unique_labels):
        for label_j in unique_labels[i+1:]:
            mask_i = labels == label_i
            mask_j = labels == label_j
            cluster_i = embeddings[mask_i]
            cluster_j = embeddings[mask_j]
            
            # 计算两个簇之间的平均距离
            from scipy.spatial.distance import cdist
            dists = cdist(cluster_i, cluster_j, metric='euclidean')
            inter_distances.extend(dists.flatten())
    
    mean_inter = np.mean(inter_distances) if inter_distances else 0
    
    ratio = mean_inter / (mean_intra + 1e-10)
    
    return {
        "mean_intra_cluster_distance": float(mean_intra),
        "mean_inter_cluster_distance": float(mean_inter),
        "isolation_ratio": float(ratio),
        "interpretation": "ratio 越大越好（> 2），表示簇之间分离良好"
    }


def load_bertopic_results(result_dir: Path) -> Tuple[np.ndarray, List[int]]:
    """
    从 BERTopic 结果目录加载聚类结果
    
    Expected files:
      - documents_*.csv（含 document_topic 列）
      - best_mc_by_method.json（或从目录名推断 mc）
    """
    import pandas as pd
    import glob
    
    # 查找 documents CSV
    csv_files = list(result_dir.glob("*documents*.csv"))
    if not csv_files:
        # 如果没有，尝试从标准位置
        csv_files = list(result_dir.glob("*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"找不到 documents CSV 文件: {result_dir}")
    
    csv_file = csv_files[0]
    df = pd.read_csv(csv_file)
    
    if "document_topic" not in df.columns:
        raise ValueError(f"CSV 中缺少 'document_topic' 列，有列: {df.columns.tolist()}")
    
    # 获取 topic 标签（-1 是噪音）
    labels = df["document_topic"].values.astype(int)
    
    return labels


def print_structural_metrics_report(metrics: Dict[str, Any]):
    """打印结构指标报告"""
    print("\n" + "="*70)
    print("📊 聚类结构指标")
    print("="*70)
    
    print("\n【Silhouette Coefficient】")
    print(f"  全局 score: {metrics['silhouette']['overall']:.4f}")
    print(f"  说明: 范围 [-1, 1]，1.0 最优，0 聚类无效，-1 聚类失败")
    
    print("\n【kNN Mixing（k=20）】")
    print(f"  平均 mixing: {metrics['knn_mixing']['mean_mixing']:.4f}")
    print(f"  聚类纯度: {metrics['knn_mixing']['purity']:.4f}")
    print(f"  说明: 纯度 1.0 最优（邻居都来自同簇），0 最差（完全混乱）")
    
    print("\n【簇隔离度】")
    print(f"  簇间/簇内距离比: {metrics['cluster_isolation']['isolation_ratio']:.4f}")
    print(f"  说明: 比值 > 2 表示聚类好，> 1 表示可接受")
    
    print("\n" + "="*70)


def main():
    """主流程：计算所有消融实验的结构指标"""
    
    # 注意：这个脚本需要配合 BERTopic 的聚类结果
    # 目前演示计算方法，具体集成需要在 BERTopic 完成后进行
    
    print("="*70)
    print("📊 聚类结构指标计算")
    print("="*70)
    
    print("\n【说明】")
    print("此脚本需要 BERTopic 聚类完成后才能运行。")
    print("使用方法：")
    print("  1. 用 4 个消融实验的向量分别跑 BERTopic")
    print("  2. 在每个输出目录下运行此脚本")
    print("  3. 对比四个版本的结构指标")
    
    print("\n【计算的指标】")
    print("  • Silhouette coefficient - 簇的紧凑性")
    print("  • kNN mixing - 跨簇邻居比例（纯度反指标）")
    print("  • 簇隔离度 - 簇间距离 / 簇内距离比")
    
    print("\n【预期输出】")
    print("  输出文件: ablation_outputs/<config>/structural_metrics.json")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
