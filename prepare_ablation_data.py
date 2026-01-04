#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_ablation_data.py
为 4 个消融版本准备输入数据和向量文件

目标：
1. 从基础数据生成 4 个版本的 topic_modeling_*.csv
2. 从投影向量生成 4 个版本的 embedding NPZ（格式兼容 step07）
3. 设置好目录结构，使得可以直接调用 step07/run_model_engine

输入：
- 06_denoised_data/helicobacter_pylori_topic_modeling_baseline.csv
- ablation_outputs/{version}/embeddings_{version}.npz

输出：
- 06_denoised_data/helicobacter_pylori_topic_modeling_{version}.csv（符号链接到baseline）
- ablation_outputs/{version}/c_{version}_final_clean_vectors.npz（投影后的向量）
- ablation_outputs/{version}/bertopic_input/（输入数据包）
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, Any
import logging

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# 配置
# ============================================================================

CONFIG_FILE = Path("experiment_config.yaml")
with open(CONFIG_FILE) as f:
    CONFIG = yaml.safe_load(f)

VERSIONS = ["baseline", "M_S", "M_S_B", "M_S_B_Anatomy"]
OUTPUT_BASE = Path(CONFIG.get("output_base_dir", "ablation_outputs"))
DENOISED_DATA_DIR = Path("06_denoised_data")

# ============================================================================
# 核心逻辑
# ============================================================================

def prepare_ablation_version(version: str) -> Dict[str, Any]:
    """为单个版本准备数据"""
    logger.info(f"\n[{version}] 准备数据...")
    
    # 1. 确保输入向量存在
    embedding_file = OUTPUT_BASE / version / f"embeddings_{version}.npz"
    if not embedding_file.exists():
        raise FileNotFoundError(f"向量文件不存在: {embedding_file}")
    
    embeddings = np.load(embedding_file)["embeddings"]
    logger.info(f"[{version}] 加载向量: shape={embeddings.shape}")
    
    # 2. 创建输入 CSV（直接使用 baseline 的 CSV，文本相同，只是向量不同）
    baseline_csv = DENOISED_DATA_DIR / "helicobacter_pylori_topic_modeling_baseline.csv"
    version_csv = DENOISED_DATA_DIR / f"helicobacter_pylori_topic_modeling_{version}.csv"
    
    if not baseline_csv.exists():
        raise FileNotFoundError(f"Baseline CSV 不存在: {baseline_csv}")
    
    # 读取 baseline CSV
    df = pd.read_csv(baseline_csv)
    num_docs_csv = len(df)
    num_docs_emb = embeddings.shape[0]
    
    if num_docs_csv != num_docs_emb:
        logger.warning(f"[{version}] CSV 文档数 {num_docs_csv} != 向量数 {num_docs_emb}")
        # 取交集
        num_docs = min(num_docs_csv, num_docs_emb)
        df = df.iloc[:num_docs]
        embeddings = embeddings[:num_docs]
        logger.info(f"[{version}] 调整到 {num_docs} 个文档")
    
    # 保存版本 CSV
    df.to_csv(version_csv, index=False)
    logger.info(f"[{version}] 保存 CSV: {version_csv}")
    
    # 3. 保存向量（兼容 step07 的格式）
    # step07 期望的文件位置：output_dir / "c_final_clean_vectors.npz"
    # 需要包含 'embeddings'、'pmids'、'description' 三个字段
    vector_dir = OUTPUT_BASE / version
    vector_dir.mkdir(parents=True, exist_ok=True)
    vector_file = vector_dir / f"c_{version}_final_clean_vectors.npz"
    
    # 读取 pmids 和文本
    pmids = df["PMID"].values if "PMID" in df.columns else np.arange(len(df))
    titles = df["Title"].values if "Title" in df.columns else np.array([""] * len(df))
    abstracts = df["Abstract"].values if "Abstract" in df.columns else np.array([""] * len(df))
    descriptions = np.array([f"{t[:50]} ... {a[:100]}" for t, a in zip(titles, abstracts)])
    
    np.savez_compressed(
        vector_file,
        embeddings=embeddings,
        pmids=pmids,
        description=descriptions
    )
    logger.info(f"[{version}] 保存向量: {vector_file} (含 pmids 和 description)")
    
    # 4. 生成元数据
    result = {
        "version": version,
        "csv_file": str(version_csv),
        "vector_file": str(vector_file),
        "num_docs": int(len(df)),
        "embedding_dim": int(embeddings.shape[1]),
        "embedding_norm_mean": float(np.linalg.norm(embeddings, axis=1).mean()),
    }
    
    return result


def main():
    logger.info("=" * 70)
    logger.info("准备消融实验数据")
    logger.info("=" * 70)
    
    results = {}
    
    for version in VERSIONS:
        try:
            result = prepare_ablation_version(version)
            results[version] = result
            logger.info(f"[{version}] ✓ 完成")
        except Exception as e:
            logger.error(f"[{version}] ✗ 失败: {e}")
            results[version] = {"error": str(e)}
    
    # 保存汇总结果
    summary_file = OUTPUT_BASE / "ablation_data_manifest.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"\n汇总结果已保存到: {summary_file}")
    
    # 打印对比表
    logger.info("\n" + "=" * 70)
    logger.info("数据准备对比")
    logger.info("=" * 70)
    
    for version in VERSIONS:
        if "error" not in results[version]:
            r = results[version]
            print(f"{version:20s} docs={r['num_docs']:6d} dim={r['embedding_dim']:3d} norm={r['embedding_norm_mean']:.4f}")
        else:
            print(f"{version:20s} ERROR: {results[version]['error']}")


if __name__ == "__main__":
    main()
