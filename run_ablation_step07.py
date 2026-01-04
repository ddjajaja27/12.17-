#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_ablation_step07.py
运行消融实验的 step07 处理（BERTopic 聚类）

策略：
1. 对每个版本，调用 step07 的核心逻辑
2. 每个版本单独处理，生成独立的结果目录
3. 最后对比 C_v 分数和其他指标

输入：
- 06_denoised_data/helicobacter_pylori_topic_modeling_{version}.csv
- ablation_outputs/{version}/c_{version}_final_clean_vectors.npz

输出：
- 07_topic_models/ABLATION_{version}/
  ├─ helicobacter_pylori_mc39_topic_info.csv
  ├─ helicobacter_pylori_mc39_doc_topic_mapping.csv
  ├─ ...
  └─ manifest.json
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any
import logging

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# 配置
# ============================================================================

VERSIONS = ["baseline", "M_S", "M_S_B", "M_S_B_Anatomy"]
PROJECT_PREFIX = "helicobacter_pylori"

# ============================================================================
# 核心函数
# ============================================================================

def run_step07_for_version(version: str, force: bool = False) -> Dict[str, Any]:
    """
    为单个版本运行 step07 的主题建模
    
    策略：调用 _engine_bertopic.py（step07 的核心）
    """
    logger.info(f"\n[{version}] 运行 step07 处理...")
    
    base_dir = Path(".").resolve()
    input_file = base_dir / "06_denoised_data" / f"{PROJECT_PREFIX}_topic_modeling_{version}.csv"
    output_dir = base_dir / "07_topic_models" / f"ABLATION_{version}"
    
    # 检查输入
    if not input_file.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_file}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 准备向量文件路径
    vector_file = base_dir / "ablation_outputs" / version / f"c_{version}_final_clean_vectors.npz"
    if not vector_file.exists():
        raise FileNotFoundError(f"向量文件不存在: {vector_file}")
    
    # 构建命令：调用 _engine_bertopic.py
    cmd = [
        sys.executable,
        "_engine_bertopic.py",
        "--input", str(input_file),
        "--output_dir", str(output_dir),
        "--embedding_npz", str(vector_file),
    ]
    
    logger.info(f"[{version}] 执行命令: {' '.join(cmd)}")
    
    # 捕获输出以便诊断
    result = subprocess.run(cmd, cwd=str(base_dir), capture_output=True, text=True, timeout=600)
    
    if result.returncode != 0:
        logger.error(f"[{version}] step07 处理失败: returncode={result.returncode}")
        logger.error(f"stdout: {result.stdout[-500:]}")  # 最后 500 字
        logger.error(f"stderr: {result.stderr[-500:]}")
        raise RuntimeError(f"step07 处理失败: returncode={result.returncode}")
    
    logger.info(f"[{version}] step07 处理完成")
    
    # 读取输出的 topic_info 获取指标
    topic_info_file = output_dir / f"{PROJECT_PREFIX}_mc39_topic_info.csv"
    if topic_info_file.exists():
        topic_info = pd.read_csv(topic_info_file)
        num_topics = int((topic_info["Topic"] >= 0).sum())
        logger.info(f"[{version}] 主题数: {num_topics}")
        
        return {
            "version": version,
            "output_dir": str(output_dir),
            "num_topics": num_topics,
            "status": "ok",
        }
    else:
        logger.warning(f"[{version}] 找不到输出文件，可能处理部分失败")
        return {
            "version": version,
            "output_dir": str(output_dir),
            "status": "partial",
        }


def read_results_from_output(version: str) -> Dict[str, Any]:
    """从输出目录读取聚类结果"""
    output_dir = Path(f"07_topic_models/ABLATION_{version}")
    
    topic_info_file = output_dir / f"{PROJECT_PREFIX}_mc39_topic_info.csv"
    doc_topic_file = output_dir / f"{PROJECT_PREFIX}_mc39_doc_topic_mapping.csv"
    
    result = {"version": version}
    
    if topic_info_file.exists():
        topic_info = pd.read_csv(topic_info_file)
        result["num_topics"] = int((topic_info["Topic"] >= 0).sum())
    
    if doc_topic_file.exists():
        doc_topic = pd.read_csv(doc_topic_file)
        noise_docs = int((doc_topic["Topic"] == -1).sum())
        result["num_noise"] = noise_docs
        result["noise_ratio"] = float(noise_docs / len(doc_topic))
    
    return result


def main():
    logger.info("=" * 70)
    logger.info("消融实验：Step07 BERTopic 处理")
    logger.info("=" * 70)
    
    results = {}
    
    # 处理每个版本
    for version in VERSIONS:
        try:
            result = run_step07_for_version(version)
            # 读取实际输出
            detailed_result = read_results_from_output(version)
            result.update(detailed_result)
            results[version] = result
            logger.info(f"[{version}] ✓ 完成")
        except Exception as e:
            logger.error(f"[{version}] ✗ 失败: {e}")
            results[version] = {"error": str(e)}
    
    # 保存结果汇总
    summary_file = Path("ablation_outputs/ablation_step07_results.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"\n结果已保存到: {summary_file}")
    
    # 打印对比表
    logger.info("\n" + "=" * 70)
    logger.info("消融实验对比")
    logger.info("=" * 70 + "\n")
    
    for version in VERSIONS:
        if "error" not in results[version]:
            r = results[version]
            topics = r.get("num_topics", "N/A")
            noise = r.get("num_noise", "N/A")
            noise_ratio = f"{r.get('noise_ratio', 0):.2%}" if "noise_ratio" in r else "N/A"
            print(f"{version:20s} topics={topics:4} noise={noise:5} ratio={noise_ratio}")
        else:
            print(f"{version:20s} ERROR")


if __name__ == "__main__":
    main()
