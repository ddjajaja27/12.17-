#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_ablation_comparison.py
生成最终的消融实验对比表

汇总：
1. BERTopic C_v 分数（从 step07 结果读取）
2. 结构指标（Silhouette/kNN/隔离度）
3. 生成可直接用于论文的对比表
"""

from __future__ import annotations

import json
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
# 读取函数
# ============================================================================

def read_c_v_from_bertopic(version: str) -> float | None:
    """从 BERTopic 结果读取 C_v 分数"""
    # step07 会生成一个结果文件，其中包含 C_v 信息
    # 假设保存在 07_topic_models/ABLATION_{version}/ 中
    
    output_dir = Path(f"07_topic_models/ABLATION_{version}")
    
    # 尝试从 topic_info.csv 读取 (BERTopic 自带的质量度量)
    topic_info_file = output_dir / f"{PROJECT_PREFIX}_mc39_topic_info.csv"
    if topic_info_file.exists():
        try:
            df = pd.read_csv(topic_info_file)
            # 有些版本的 BERTopic 会在 topic_info 中包含 Top_n_words_per_topic 等信息
            # 但 C_v 通常需要手动计算或从日志中提取
            # 这里先返回 None，后续可从日志或其他文件补充
            return None
        except:
            return None
    
    return None


def read_structural_metrics(version: str) -> Dict[str, Any]:
    """读取结构指标"""
    metrics_file = Path(f"07_topic_models/ABLATION_{version}/structural_metrics.json")
    
    if metrics_file.exists():
        try:
            with open(metrics_file) as f:
                return json.load(f)
        except:
            pass
    
    return {}


def read_bertopic_summary() -> Dict[str, Any]:
    """读取 BERTopic 处理的汇总信息"""
    summary_file = Path("ablation_outputs/ablation_step07_results.json")
    
    if summary_file.exists():
        try:
            with open(summary_file) as f:
                return json.load(f)
        except:
            pass
    
    return {}


def main():
    logger.info("=" * 70)
    logger.info("消融实验：生成最终对比表")
    logger.info("=" * 70)
    
    # 1. 读取 BERTopic 结果汇总
    bertopic_summary = read_bertopic_summary()
    logger.info(f"读取 BERTopic 汇总: {len(bertopic_summary)} 个版本")
    
    # 2. 汇总数据
    comparison_data = []
    
    for version in VERSIONS:
        row = {"版本": version}
        
        # BERTopic 信息
        if version in bertopic_summary:
            bert_info = bertopic_summary[version]
            if "error" not in bert_info:
                row["主题数"] = bert_info.get("num_topics", "-")
                row["噪声数"] = bert_info.get("num_noise", "-")
                row["噪声%"] = f"{bert_info.get('noise_ratio', 0) * 100:.1f}%"
            else:
                row["主题数"] = "ERROR"
                row["噪声数"] = "-"
                row["噪声%"] = "-"
        else:
            row["主题数"] = "-"
            row["噪声数"] = "-"
            row["噪声%"] = "-"
        
        # 结构指标
        metrics = read_structural_metrics(version)
        if metrics:
            row["Silhouette"] = f"{metrics.get('silhouette', 0):.4f}"
            row["kNN混合"] = f"{metrics.get('knn_mixing', 0):.4f}"
            row["隔离度"] = f"{metrics.get('cluster_isolation', 0):.4f}"
            row["DB指数"] = f"{metrics.get('davies_bouldin', 0):.4f}"
        else:
            row["Silhouette"] = "-"
            row["kNN混合"] = "-"
            row["隔离度"] = "-"
            row["DB指数"] = "-"
        
        comparison_data.append(row)
    
    # 3. 生成对比表
    comparison_df = pd.DataFrame(comparison_data)
    
    logger.info("\n" + "=" * 70)
    logger.info("消融实验最终对比")
    logger.info("=" * 70 + "\n")
    
    print(comparison_df.to_string(index=False))
    
    # 4. 保存为 CSV
    output_file = Path("ablation_outputs/ABLATION_COMPARISON_FINAL.csv")
    comparison_df.to_csv(output_file, index=False)
    logger.info(f"\n对比表已保存到: {output_file}")
    
    # 5. 保存为 JSON（便于进一步处理）
    json_file = Path("ablation_outputs/ABLATION_COMPARISON_FINAL.json")
    with open(json_file, "w", encoding="utf-8") as f:
        # 转换为适合 JSON 的格式
        json_data = {
            "versions": VERSIONS,
            "timestamp": pd.Timestamp.now().isoformat(),
            "comparison": comparison_data,
        }
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    logger.info(f"详细数据已保存到: {json_file}")
    
    # 6. 分析结论
    logger.info("\n" + "=" * 70)
    logger.info("分析结论")
    logger.info("=" * 70)
    
    # 查看哪个版本效果最好
    try:
        # 按某个指标排序（例如隔离度越高越好）
        isolation_values = []
        for row in comparison_data:
            try:
                val = float(row["隔离度"])
                isolation_values.append((row["版本"], val))
            except:
                pass
        
        if isolation_values:
            isolation_values.sort(key=lambda x: x[1], reverse=True)
            best = isolation_values[0]
            worst = isolation_values[-1]
            improvement = (best[1] - worst[1]) / worst[1] * 100
            logger.info(f"隔离度最高: {best[0]} ({best[1]:.4f})")
            logger.info(f"隔离度最低: {worst[0]} ({worst[1]:.4f})")
            logger.info(f"改进幅度: {improvement:.1f}%")
    except:
        pass
    
    logger.info("\n✓ 消融实验对比完成")


if __name__ == "__main__":
    main()
