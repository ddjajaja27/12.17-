#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 doc-topic 映射和 2D 可视化计算 C_V 和 kNN mixing
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def load_data(version_dir: Path):
    """加载版本的数据"""
    
    # 加载主题分配
    topics_file = version_dir / "topics.csv"
    df = pd.read_csv(topics_file)
    topics = df["topic"].values
    
    # 加载 2D embeddings 用于可视化相似度计算
    embeddings_2d_file = version_dir / "embeddings_2d.csv"
    embeddings_2d = None
    if embeddings_2d_file.exists():
        df_2d = pd.read_csv(embeddings_2d_file)
        embeddings_2d = df_2d[["umap_x", "umap_y"]].values
    
    return topics, embeddings_2d

def calculate_coherence_from_topics(topics: np.ndarray) -> dict:
    """
    从主题分配计算相关指标
    """
    results = {}
    
    # 计算主题数（不包括噪声）
    unique_topics = np.unique(topics)
    valid_topics = unique_topics[unique_topics >= 0]
    results["num_topics"] = len(valid_topics)
    
    # 计算噪声比例
    noise_count = (topics == -1).sum()
    noise_ratio = noise_count / len(topics)
    results["noise_ratio"] = noise_ratio
    results["noise_percent"] = noise_ratio * 100
    
    # 计算主题大小分布（作为质量指标）
    topic_sizes = []
    for t in valid_topics:
        size = (topics == t).sum()
        topic_sizes.append(size)
    
    results["avg_topic_size"] = np.mean(topic_sizes) if topic_sizes else 0
    results["median_topic_size"] = np.median(topic_sizes) if topic_sizes else 0
    
    return results

def calculate_clustering_quality(topics: np.ndarray, embeddings_2d: np.ndarray) -> dict:
    """
    基于 2D 可视化嵌入计算聚类质量
    """
    results = {}
    
    if embeddings_2d is None:
        return results
    
    unique_topics = np.unique(topics)
    valid_topics = unique_topics[unique_topics >= 0]
    
    # 计算主题内和主题间的距离
    intra_distances = []
    inter_distances = []
    
    for topic_id in valid_topics:
        mask = topics == topic_id
        if mask.sum() <= 1:
            continue
        
        topic_embeddings = embeddings_2d[mask]
        
        # 主题内距离（使用欧氏距离）
        if len(topic_embeddings) > 1:
            pairwise_dist = np.sqrt(((topic_embeddings[:, None, :] - topic_embeddings[None, :, :]) ** 2).sum(axis=2))
            # 只计算上三角（避免重复）
            intra_distances.extend(pairwise_dist[np.triu_indices_from(pairwise_dist, k=1)])
    
    # 简化的聚类质量指标：主题内紧密度
    # 使用主题文档与主题中心的距离
    coherence_scores = []
    for topic_id in valid_topics:
        mask = topics == topic_id
        if mask.sum() <= 1:
            continue
        
        topic_embeddings = embeddings_2d[mask]
        center = topic_embeddings.mean(axis=0)
        distances = np.sqrt(((topic_embeddings - center) ** 2).sum(axis=1))
        coherence_scores.append(1.0 / (1.0 + distances.mean()))  # 转换为 0-1 分数
    
    if coherence_scores:
        results["avg_coherence"] = np.mean(coherence_scores)
        results["median_coherence"] = np.median(coherence_scores)
    
    return results

def calculate_knn_mixing(topics: np.ndarray, embeddings_2d: np.ndarray, k: int = 5) -> float:
    """
    计算 kNN mixing：相同主题的文档在嵌入空间中有多接近
    """
    if embeddings_2d is None:
        return None
    
    n_docs = len(embeddings_2d)
    
    # 使用 2D embeddings 找邻居
    nbrs = NearestNeighbors(n_neighbors=min(k+1, n_docs), algorithm='auto').fit(embeddings_2d)
    distances, indices = nbrs.kneighbors(embeddings_2d)
    
    # 计算混合指标
    mixing_scores = []
    for i in range(n_docs):
        doc_topic = topics[i]
        if doc_topic == -1:  # 噪声文档
            continue
        
        # 找 k 个邻居中有多少属于同一主题（不包括自己）
        actual_k = min(k, len(indices[i]) - 1)
        if actual_k <= 0:
            continue
        
        neighbor_topics = topics[indices[i][1:actual_k+1]]
        same_topic_count = (neighbor_topics == doc_topic).sum()
        mixing_score = same_topic_count / actual_k
        mixing_scores.append(mixing_score)
    
    if mixing_scores:
        return np.mean(mixing_scores)
    else:
        return None

def main():
    print("\n" + "="*80)
    print("聚类质量指标计算")
    print("="*80 + "\n")
    
    versions = {
        "Baseline": Path("ablation_outputs/baseline/bertopic_results"),
        "VPD": Path("ablation_outputs/M_S_B_Anatomy/bertopic_results"),
    }
    
    all_results = {}
    
    for version_name, output_dir in versions.items():
        print(f"Processing {version_name}...")
        
        try:
            topics, embeddings_2d = load_data(output_dir)
            
            # 基础指标
            coherence_info = calculate_coherence_from_topics(topics)
            quality_info = calculate_clustering_quality(topics, embeddings_2d)
            knn_mixing = calculate_knn_mixing(topics, embeddings_2d, k=5)
            
            results = {
                **coherence_info,
                **quality_info,
                "knn_mixing_k5": knn_mixing,
            }
            
            all_results[version_name] = results
            
            print(f"  Topics: {results['num_topics']}")
            print(f"  Noise: {results['noise_percent']:.2f}%")
            print(f"  Avg topic size: {results['avg_topic_size']:.1f}")
            if 'avg_coherence' in results:
                print(f"  Avg coherence: {results['avg_coherence']:.4f}")
            if knn_mixing is not None:
                print(f"  kNN Mixing (k=5): {knn_mixing:.4f}")
            print()
            
        except Exception as e:
            print(f"  Error: {e}\n")
    
    # 汇总表
    print("="*80)
    print("FINAL COMPARISON TABLE")
    print("="*80)
    
    headers = ["Metric", "Baseline", "VPD"]
    metrics = [
        ("Topics", "num_topics"),
        ("Noise (%)", "noise_percent"),
        ("Avg Topic Size", "avg_topic_size"),
        ("Median Topic Size", "median_topic_size"),
        ("Avg Coherence", "avg_coherence"),
        ("Median Coherence", "median_coherence"),
        ("kNN Mixing (k=5)", "knn_mixing_k5"),
    ]
    
    print(f"{headers[0]:<30} {headers[1]:<25} {headers[2]:<25}")
    print("-" * 80)
    
    for metric_name, metric_key in metrics:
        baseline_val = all_results.get("Baseline", {}).get(metric_key, "N/A")
        vpd_val = all_results.get("VPD", {}).get(metric_key, "N/A")
        
        if isinstance(baseline_val, float):
            if metric_key == "noise_percent":
                baseline_str = f"{baseline_val:.2f}%"
            else:
                baseline_str = f"{baseline_val:.4f}"
        else:
            baseline_str = str(baseline_val)
        
        if isinstance(vpd_val, float):
            if metric_key == "noise_percent":
                vpd_str = f"{vpd_val:.2f}%"
            else:
                vpd_str = f"{vpd_val:.4f}"
        else:
            vpd_str = str(vpd_val)
        
        print(f"{metric_name:<30} {baseline_str:<25} {vpd_str:<25}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
