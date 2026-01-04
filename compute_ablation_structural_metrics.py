#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_ablation_structural_metrics.py
对消融实验结果计算结构指标

输入：
- 07_topic_models/ABLATION_{version}/helicobacter_pylori_mc39_*
- ablation_outputs/{version}/c_{version}_final_clean_vectors.npz

输出：
- 07_topic_models/ABLATION_{version}/structural_metrics.json
- ablation_outputs/ablation_structural_metrics_summary.csv
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.spatial.distance import pdist, squareform, cdist

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# 配置
# ============================================================================

VERSIONS = ["baseline", "M_S", "M_S_B", "M_S_B_Anatomy"]
PROJECT_PREFIX = "helicobacter_pylori"

# ============================================================================
# 核心指标函数
# ============================================================================

def compute_silhouette(embeddings: np.ndarray, labels: np.ndarray, sample_size: int = 5000) -> float:
    """
    计算 Silhouette 系数
    
    对于大数据集，抽样计算以加速
    """
    if len(np.unique(labels[labels != -1])) < 2:
        return -1.0  # 少于2个簇
    
    # 抽样
    if len(embeddings) > sample_size:
        idx = np.random.choice(len(embeddings), size=sample_size, replace=False)
        embeddings_sample = embeddings[idx]
        labels_sample = labels[idx]
    else:
        embeddings_sample = embeddings
        labels_sample = labels
    
    # 只计算非噪声点
    mask = labels_sample != -1
    if mask.sum() < 2:
        return -1.0
    
    try:
        score = silhouette_score(embeddings_sample[mask], labels_sample[mask], metric="cosine")
        return float(score)
    except Exception as e:
        logger.warning(f"Silhouette 计算失败: {e}")
        return -1.0


def compute_knn_mixing(embeddings: np.ndarray, labels: np.ndarray, k: int = 20) -> float:
    """
    计算 k-NN 混合率（跨簇邻居比例）
    
    低值表示簇分离良好
    """
    if len(np.unique(labels[labels != -1])) < 2:
        return 1.0  # 无效
    
    # 计算距离矩阵（仅非噪声点）
    mask = labels != -1
    embeddings_clean = embeddings[mask]
    labels_clean = labels[mask]
    
    if len(embeddings_clean) < k + 1:
        k = max(1, len(embeddings_clean) - 1)
    
    # 计算每个点到其他点的距离
    distances = cdist(embeddings_clean, embeddings_clean, metric="cosine")
    
    mixing_count = 0
    for i in range(len(embeddings_clean)):
        # 找到 k 个最近邻
        neighbors_idx = np.argsort(distances[i])[1:k+1]  # 排除自己
        neighbors_labels = labels_clean[neighbors_idx]
        
        # 统计与自己不同簇的邻居
        diff_cluster = (neighbors_labels != labels_clean[i]).sum()
        mixing_count += diff_cluster
    
    mixing_rate = mixing_count / (len(embeddings_clean) * k)
    return float(mixing_rate)


def compute_cluster_isolation(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """
    计算簇隔离指数（inter-cluster / intra-cluster distance ratio）
    
    > 1 表示簇分离良好
    """
    if len(np.unique(labels[labels != -1])) < 2:
        return 1.0
    
    mask = labels != -1
    embeddings_clean = embeddings[mask]
    labels_clean = labels[mask]
    
    # 计算簇内距离平均值
    intra_distances = []
    for label in np.unique(labels_clean):
        cluster_mask = labels_clean == label
        if cluster_mask.sum() > 1:
            cluster_embeddings = embeddings_clean[cluster_mask]
            distances = pdist(cluster_embeddings, metric="cosine")
            intra_distances.extend(distances)
    
    if not intra_distances:
        return 1.0
    
    intra_dist_mean = np.mean(intra_distances)
    
    # 计算簇间距离平均值
    unique_labels = np.unique(labels_clean)
    if len(unique_labels) < 2:
        return 1.0
    
    inter_distances = []
    for i in range(len(unique_labels)):
        for j in range(i + 1, len(unique_labels)):
            label_i = unique_labels[i]
            label_j = unique_labels[j]
            cluster_i = embeddings_clean[labels_clean == label_i]
            cluster_j = embeddings_clean[labels_clean == label_j]
            distances = cdist(cluster_i, cluster_j, metric="cosine")
            inter_distances.extend(distances.flatten())
    
    if not inter_distances:
        return 1.0
    
    inter_dist_mean = np.mean(inter_distances)
    
    # 隔离指数 = 簇间 / 簇内
    isolation_ratio = inter_dist_mean / max(intra_dist_mean, 1e-6)
    return float(isolation_ratio)


def compute_davies_bouldin(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """
    计算 Davies-Bouldin 指数（越小越好）
    """
    if len(np.unique(labels[labels != -1])) < 2:
        return float('inf')
    
    mask = labels != -1
    try:
        score = davies_bouldin_score(embeddings[mask], labels[mask])
        return float(score)
    except:
        return float('inf')


# ============================================================================
# 主函数
# ============================================================================

def process_version(version: str) -> Dict[str, Any]:
    """处理单个版本"""
    logger.info(f"\n[{version}] 计算结构指标...")
    
    # 1. 加载文档-主题映射
    output_dir = Path(f"07_topic_models/ABLATION_{version}")
    topic_file = output_dir / f"{PROJECT_PREFIX}_mc39_doc_topic_mapping.csv"
    
    if not topic_file.exists():
        logger.warning(f"[{version}] 主题文件不存在: {topic_file}")
        return {"version": version, "status": "missing"}
    
    df_topics = pd.read_csv(topic_file)
    labels = df_topics["Topic"].values
    
    logger.info(f"[{version}] 加载主题标签: {len(labels)} 个文档, {len(np.unique(labels))} 个簇")
    
    # 2. 加载向量
    vector_file = Path(f"ablation_outputs/{version}/c_{version}_final_clean_vectors.npz")
    if not vector_file.exists():
        logger.warning(f"[{version}] 向量文件不存在: {vector_file}")
        return {"version": version, "status": "missing"}
    
    embeddings = np.load(vector_file)["embeddings"]
    logger.info(f"[{version}] 加载向量: {embeddings.shape}")
    
    # 3. 计算指标
    result = {
        "version": version,
        "num_docs": int(len(labels)),
        "num_clusters": int(len(np.unique(labels[labels != -1]))),
        "num_noise_docs": int((labels == -1).sum()),
    }
    
    # Silhouette
    silhouette = compute_silhouette(embeddings, labels)
    result["silhouette"] = silhouette
    logger.info(f"[{version}] Silhouette: {silhouette:.4f}")
    
    # kNN 混合率
    knn_mixing = compute_knn_mixing(embeddings, labels, k=20)
    result["knn_mixing"] = knn_mixing
    logger.info(f"[{version}] kNN Mixing (k=20): {knn_mixing:.4f}")
    
    # 簇隔离
    isolation = compute_cluster_isolation(embeddings, labels)
    result["cluster_isolation"] = isolation
    logger.info(f"[{version}] 簇隔离: {isolation:.4f}")
    
    # Davies-Bouldin
    db_index = compute_davies_bouldin(embeddings, labels)
    result["davies_bouldin"] = db_index
    logger.info(f"[{version}] Davies-Bouldin: {db_index:.4f}")
    
    result["status"] = "ok"
    
    # 4. 保存结果
    metrics_file = output_dir / "structural_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"[{version}] 保存指标到: {metrics_file}")
    
    return result


def main():
    logger.info("=" * 70)
    logger.info("消融实验：计算结构指标")
    logger.info("=" * 70)
    
    all_results = {}
    
    for version in VERSIONS:
        try:
            result = process_version(version)
            all_results[version] = result
            logger.info(f"[{version}] ✓ 完成")
        except Exception as e:
            logger.error(f"[{version}] ✗ 失败: {e}")
            all_results[version] = {"version": version, "error": str(e)}
    
    # 汇总结果
    logger.info("\n" + "=" * 70)
    logger.info("结构指标对比")
    logger.info("=" * 70 + "\n")
    
    summary_data = []
    for version in VERSIONS:
        if all_results[version].get("status") == "ok":
            r = all_results[version]
            summary_data.append({
                "版本": version,
                "簇数": r.get("num_clusters", "-"),
                "噪声数": r.get("num_noise_docs", "-"),
                "Silhouette": f"{r.get('silhouette', 0):.4f}",
                "kNN混合": f"{r.get('knn_mixing', 0):.4f}",
                "隔离度": f"{r.get('cluster_isolation', 0):.4f}",
                "DB指数": f"{r.get('davies_bouldin', 0):.4f}",
            })
        else:
            summary_data.append({
                "版本": version,
                "簇数": "ERROR",
                "噪声数": "-",
                "Silhouette": "-",
                "kNN混合": "-",
                "隔离度": "-",
                "DB指数": "-",
            })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # 保存汇总表
    summary_file = Path("ablation_outputs/ablation_structural_metrics_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    logger.info(f"\n汇总表已保存到: {summary_file}")
    
    # 保存完整 JSON
    json_file = Path("ablation_outputs/ablation_structural_metrics.json")
    with open(json_file, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info(f"详细结果已保存到: {json_file}")


if __name__ == "__main__":
    main()
