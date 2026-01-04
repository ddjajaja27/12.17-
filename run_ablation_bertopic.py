#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_ablation_bertopic.py
消融实验：对 4 个向量版本各运行一次 BERTopic，计算结构指标

输入：
  ablation_outputs/{baseline,M_S,M_S_B,M_S_B_Anatomy}/embeddings_*.npz

输出：
  ablation_outputs/{version}/bertopic_results/
    ├─ topic_model.pkl          (保存的 BERTopic 模型)
    ├─ topics.csv               (文档-主题映射)
    ├─ topic_info.csv           (主题信息表)
    ├─ c_v_score.txt            (C_v 得分)
    ├─ noise_ratio.txt          (噪声文档比例)
    └─ structural_metrics.json   (Silhouette / kNN mixing / 隔离度)

参数：
  所有参数从 experiment_config.yaml 读取（确保一致性）

用法：
  python run_ablation_bertopic.py
  python run_ablation_bertopic.py --version baseline  (只运行某个版本)
  python run_ablation_bertopic.py --force             (强制重新运行)
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import argparse
import pickle
import logging

import numpy as np
import pandas as pd
import yaml

try:
    from sentence_transformers import SentenceTransformer
    from bertopic import BERTopic
    from umap import UMAP
    from hdbscan import HDBSCAN
except ImportError as e:
    print(f"缺少必要的包: {e}")
    sys.exit(1)

# ============================================================================
# 配置和日志
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# 从 experiment_config.yaml 读取配置
CONFIG_FILE = Path("experiment_config.yaml")
if not CONFIG_FILE.exists():
    logger.error("实验配置文件不存在: experiment_config.yaml")
    sys.exit(1)

with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

VERSIONS = ["baseline", "M_S", "M_S_B", "M_S_B_Anatomy"]
OUTPUT_BASE = Path(CONFIG.get("output_base_dir", "ablation_outputs"))
RAW_DATA_DIR = Path(CONFIG.get("raw_data_dir", "01_raw_data"))

# 读取文档（数据集）
# 使用处理过的数据文件，而不是原始数据（确保文档数与向量数一致）
DOCS_FILE = Path("06_denoised_data/helicobacter_pylori_topic_modeling_denoised.csv")

# ============================================================================
# 核心函数
# ============================================================================

def load_documents() -> list[str]:
    """加载处理过的文档"""
    if not DOCS_FILE.exists():
        logger.error(f"文档文件不存在: {DOCS_FILE}")
        sys.exit(1)
    
    df = pd.read_csv(DOCS_FILE)
    logger.info(f"加载的列: {list(df.columns)[:5]}")
    
    # 使用 Abstract 列（经过处理的摘要）
    if "Abstract" in df.columns:
        docs = df["Abstract"].fillna("").astype(str).tolist()
    elif "abstract" in df.columns:
        docs = df["abstract"].fillna("").astype(str).tolist()
    elif "Title" in df.columns:
        docs = df["Title"].fillna("").astype(str).tolist()
    else:
        logger.warning("未找到预期的文本列，使用第一列")
        docs = df.iloc[:, 0].fillna("").astype(str).tolist()
    
    logger.info(f"加载了 {len(docs)} 个文档")
    return docs


def load_embeddings(version: str) -> np.ndarray:
    """加载指定版本的向量"""
    emb_file = OUTPUT_BASE / version / f"embeddings_{version}.npz"
    if not emb_file.exists():
        raise FileNotFoundError(f"向量文件不存在: {emb_file}")
    
    data = np.load(emb_file)
    embeddings = data["embeddings"]
    logger.info(f"[{version}] 加载向量: shape={embeddings.shape}")
    return embeddings


def run_bertopic(
    docs: list[str],
    embeddings: np.ndarray,
    version: str,
    force: bool = False
) -> Dict[str, Any]:
    """
    运行 BERTopic 聚类
    
    参数：
    - docs: 文档列表
    - embeddings: 向量数组 (N × D)
    - version: 版本名称 (baseline/M_S/...)
    - force: 是否强制重新运行
    
    返回：
    - results_dict: 包含 C_v、噪声比例等信息
    """
    logger.info(f"[{version}] 开始 BERTopic 聚类...")
    
    output_dir = OUTPUT_BASE / version / "bertopic_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查是否已完成
    model_file = output_dir / "topic_model.pkl"
    if model_file.exists() and not force:
        logger.info(f"[{version}] 检测到已完成的模型，跳过")
        return load_previous_results(output_dir)
    
    # ========================================================================
    # 初始化 UMAP 和 HDBSCAN（使用配置中的参数）
    # ========================================================================
    # 聚类用 5D UMAP（重要：只用于聚类，不用于可视化）
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,  # 聚类维度：5D
        metric="cosine",
        min_dist=0.1,
        random_state=42,
    )
    
    # 尝试用多个 min_cluster_size 值
    mc_values = [20, 30, 39, 50, 70]  # 从小到大尝试
    hdbscan_model = None
    
    for mc in mc_values:
        try:
            logger.info(f"[{version}] 尝试 HDBSCAN min_cluster_size={mc}")
            hdbscan_model = HDBSCAN(
                min_cluster_size=mc,
                metric="euclidean",
                prediction_data=True,  # 确保生成预测数据
            )
            
            # ========================================================================
            # 初始化 BERTopic
            # ========================================================================
            embedding_model = "all-MiniLM-L6-v2"
            
            topic_model = BERTopic(
                embedding_model=embedding_model,
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                calculate_probabilities=True,
                verbose=True,
            )
            
            # ========================================================================
            # 拟合模型
            # ========================================================================
            logger.info(f"[{version}] 拟合 BERTopic...")
            topics, probs = topic_model.fit_transform(docs, embeddings)
            logger.info(f"[{version}] 拟合完成（mc={mc}）")
            break  # 成功则跳出循环
            
        except Exception as e:
            logger.warning(f"[{version}] mc={mc} 失败: {e}")
            if mc == mc_values[-1]:
                logger.error(f"[{version}] 所有 mc 值都失败")
                raise
            continue
    
    # ========================================================================
    # 保存模型
    # ========================================================================
    with open(model_file, "wb") as f:
        pickle.dump(topic_model, f)
    logger.info(f"[{version}] 保存模型到 {model_file}")
    
    # ========================================================================
    # 提取结果
    # ========================================================================
    # 主题信息表
    topic_info = topic_model.get_topic_info()
    topic_info_file = output_dir / "topic_info.csv"
    topic_info.to_csv(topic_info_file, index=False)
    logger.info(f"[{version}] 保存主题信息到 {topic_info_file}")
    
    # 文档-主题映射
    doc_topic_df = pd.DataFrame({
        "document_id": range(len(docs)),
        "topic": topics,
        "probability": probs.max(axis=1),
    })
    doc_topic_file = output_dir / "topics.csv"
    doc_topic_df.to_csv(doc_topic_file, index=False)
    logger.info(f"[{version}] 保存文档-主题映射到 {doc_topic_file}")
    
    # ========================================================================
    # 生成 2D UMAP 用于可视化（不用于聚类，仅用于绘图）
    # ========================================================================
    try:
        logger.info(f"[{version}] 生成 2D UMAP 可视化...")
        umap_2d = UMAP(
            n_neighbors=CONFIG.get("umap_n_neighbors", 15),
            n_components=2,  # 可视化维度：2D
            metric=CONFIG.get("umap_metric", "cosine"),
            min_dist=CONFIG.get("umap_min_dist", 0.1),
            random_state=CONFIG.get("umap_random_state", 42),
        )
        embeddings_2d = umap_2d.fit_transform(embeddings)
        
        # 保存 2D 嵌入供可视化使用
        embeddings_2d_df = pd.DataFrame(
            embeddings_2d,
            columns=["umap_x", "umap_y"]
        )
        embeddings_2d_df["topic"] = topics
        embeddings_2d_file = output_dir / "embeddings_2d.csv"
        embeddings_2d_df.to_csv(embeddings_2d_file, index=False)
        logger.info(f"[{version}] 保存 2D UMAP 到 {embeddings_2d_file}")
    except Exception as e:
        logger.warning(f"[{version}] 生成 2D UMAP 失败: {e}")
    
    # ========================================================================
    # 计算评价指标
    # ========================================================================
    try:
        # 确保 topics 是 numpy 数组
        if not isinstance(topics, np.ndarray):
            topics = np.array(topics)
        
        num_topics = len(topic_info[topic_info["Topic"] >= 0])
        num_noise = int((topics == -1).sum())
        noise_ratio = num_noise / len(docs)
        
        logger.info(f"[{version}] 主题数: {num_topics}, 噪声文档: {num_noise} ({noise_ratio:.2%})")
        
        # C_v 分数（如果可用）
        c_v_score = None
        try:
            c_v_score = topic_model.calculate_topic_model_quality()
            logger.info(f"[{version}] C_v 分数: {c_v_score:.4f}")
        except Exception as e:
            logger.warning(f"[{version}] 无法计算 C_v 分数: {e}")
        
        # 保存 C_v 和噪声比例
        with open(output_dir / "c_v_score.txt", "w") as f:
            if c_v_score is not None:
                f.write(f"{c_v_score:.4f}\n")
            else:
                f.write("N/A\n")
        
        with open(output_dir / "noise_ratio.txt", "w") as f:
            f.write(f"{noise_ratio:.4f}\n")
        
        results = {
            "version": version,
            "num_topics": num_topics,
            "num_noise_docs": int(num_noise),
            "noise_ratio": float(noise_ratio),
            "c_v_score": float(c_v_score) if c_v_score is not None else None,
            "topics": topics.tolist(),
            "probabilities": probs.tolist() if hasattr(probs, "tolist") else probs,
        }
        
        return results, embeddings
        
    except Exception as e:
        logger.error(f"[{version}] 计算指标失败: {e}")
        import traceback
        traceback.print_exc()
        raise


def load_previous_results(output_dir: Path) -> Dict[str, Any]:
    """从已保存的结果加载数据"""
    with open(output_dir / "topic_info.csv") as f:
        topic_info = pd.read_csv(f)
    
    with open(output_dir / "topics.csv") as f:
        doc_topic = pd.read_csv(f)
    
    with open(output_dir / "c_v_score.txt") as f:
        c_v_text = f.read().strip()
        c_v_score = float(c_v_text) if c_v_text != "N/A" else None
    
    with open(output_dir / "noise_ratio.txt") as f:
        noise_ratio = float(f.read().strip())
    
    return {
        "num_topics": int(len(topic_info[topic_info["Topic"] >= 0])),
        "num_noise_docs": int((doc_topic["topic"] == -1).sum()),
        "noise_ratio": noise_ratio,
        "c_v_score": c_v_score,
    }


def compute_structural_metrics(
    embeddings: np.ndarray,
    topics: np.ndarray,
    output_dir: Path
) -> Dict[str, float]:
    """
    计算结构指标：Silhouette / kNN mixing / 隔离度
    
    这个函数稍后由 compute_structural_metrics.py 处理
    """
    # 这里留作占位符，实际计算由后续脚本处理
    return {}


def main():
    parser = argparse.ArgumentParser(
        description="消融实验：对 4 个向量版本运行 BERTopic"
    )
    parser.add_argument(
        "--version",
        choices=VERSIONS,
        default=None,
        help="只运行指定版本（默认运行所有）"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新运行，即使已存在结果"
    )
    
    args = parser.parse_args()
    
    # 选择要运行的版本
    versions_to_run = [args.version] if args.version else VERSIONS
    
    # 加载文档（共用）
    logger.info("加载文档...")
    docs = load_documents()
    
    # ========================================================================
    # 对每个版本运行 BERTopic
    # ========================================================================
    summary = {}
    
    for version in versions_to_run:
        logger.info(f"\n{'='*70}")
        logger.info(f"处理版本: {version}")
        logger.info(f"{'='*70}")
        
        try:
            # 加载向量
            embeddings = load_embeddings(version)
            
            # 运行 BERTopic
            results, saved_embeddings = run_bertopic(docs, embeddings, version, force=args.force)
            
            summary[version] = results
            
            logger.info(f"[{version}] ✓ 完成")
            
        except Exception as e:
            logger.error(f"[{version}] ✗ 失败: {e}")
            summary[version] = {"error": str(e)}
    
    # ========================================================================
    # 生成对比表
    # ========================================================================
    logger.info(f"\n{'='*70}")
    logger.info("消融实验对比")
    logger.info(f"{'='*70}\n")
    
    comparison_df = pd.DataFrame([
        {
            "版本": v,
            "主题数": summary[v].get("num_topics", "N/A"),
            "噪声文档": summary[v].get("num_noise_docs", "N/A"),
            "噪声比例": f"{summary[v].get('noise_ratio', 0):.2%}",
            "C_v 分数": f"{summary[v].get('c_v_score', 0):.4f}" if summary[v].get("c_v_score") else "N/A",
        }
        for v in versions_to_run
    ])
    
    print(comparison_df.to_string(index=False))
    
    # 保存对比表
    comparison_file = OUTPUT_BASE / "ablation_comparison_table.csv"
    comparison_df.to_csv(comparison_file, index=False)
    logger.info(f"\n对比表已保存到: {comparison_file}")
    
    # 保存完整的 JSON 结果
    results_file = OUTPUT_BASE / "ablation_bertopic_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"详细结果已保存到: {results_file}")
    
    logger.info("\n✓ 所有版本处理完成")


if __name__ == "__main__":
    main()
