#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_umap_fix.py
验证 UMAP n_components=5 修复的效果

检查项：
1. 噪声比例算术正确性
2. mc39 文件的主题数和C_V
3. baseline vs VPD 对比
"""

import pandas as pd
import numpy as np
from pathlib import Path

def check_noise_ratio(version: str, mc: int = 39) -> dict:
    """检查指定版本和mc的噪声比例"""
    try:
        # 尝试多个可能的路径
        paths_to_try = [
            f"ablation_outputs/{version}/bertopic_results/topics.csv",
            f"07_topic_models/ABLATION_{version}/helicobacter_pylori_mc{mc}_doc_topic_mapping.csv",
            f"07_topic_models/{version.upper()}/helicobacter_pylori_mc{mc}_doc_topic_mapping.csv",
        ]
        
        mapping_file = None
        for path in paths_to_try:
            if Path(path).exists():
                mapping_file = path
                break
        
        if not mapping_file:
            return {"error": f"找不到 {version} 的 doc_topic_mapping 文件"}
        
        df = pd.read_csv(mapping_file)
        
        # 第二列通常是 Topic 列
        topic_col = df.columns[1] if len(df.columns) > 1 else "Topic"
        
        noise_count = (df[topic_col] == -1).sum()
        total_count = len(df)
        noise_ratio = noise_count / total_count
        
        return {
            "version": version,
            "mc": mc,
            "file": mapping_file,
            "total_docs": total_count,
            "noise_docs": noise_count,
            "noise_ratio": f"{noise_ratio:.4f}",
            "noise_percent": f"{noise_ratio * 100:.2f}%",
            "correct_calc": f"{noise_count} / {total_count} = {noise_ratio:.4f}",
        }
    except Exception as e:
        return {"error": str(e), "version": version}

def check_topics_and_cv(version: str, mc: int = 39) -> dict:
    """检查指定版本和mc的主题数和C_V"""
    try:
        paths_to_try = [
            f"ablation_outputs/{version}/bertopic_results/topic_info.csv",
            f"07_topic_models/ABLATION_{version}/helicobacter_pylori_mc{mc}_topic_info.csv",
            f"07_topic_models/{version.upper()}/helicobacter_pylori_mc{mc}_topic_info.csv",
        ]
        
        info_file = None
        for path in paths_to_try:
            if Path(path).exists():
                info_file = path
                break
        
        if not info_file:
            return {"error": f"找不到 {version} 的 topic_info 文件"}
        
        df = pd.read_csv(info_file)
        
        # 第一列通常是 Topic，最后一列通常是 c_v
        topic_col = df.columns[0]
        cv_col = df.columns[-1]
        
        valid_topics = df[df[topic_col] >= 0]
        topic_count = len(valid_topics)
        mean_cv = valid_topics[cv_col].mean() if cv_col in valid_topics.columns else None
        
        return {
            "version": version,
            "mc": mc,
            "file": info_file,
            "topic_count": topic_count,
            "mean_cv": f"{mean_cv:.4f}" if mean_cv is not None else "N/A",
            "cv_col": cv_col,
        }
    except Exception as e:
        return {"error": str(e), "version": version}

def main():
    print("=" * 80)
    print("UMAP n_components=5 修复验证")
    print("=" * 80)
    print()
    
    versions = ["baseline", "VPD"]
    mc = 39
    
    print(f"【检查点 1】噪声比例（mc={mc}）")
    print("-" * 80)
    for v in versions:
        result = check_noise_ratio(v, mc)
        if "error" in result:
            print(f"✗ {v}: {result['error']}")
        else:
            print(f"✓ {v}:")
            for key, val in result.items():
                if key != "file":
                    print(f"  {key:20} = {val}")
    print()
    
    print(f"【检查点 2】主题数和平均C_V（mc={mc}）")
    print("-" * 80)
    for v in versions:
        result = check_topics_and_cv(v, mc)
        if "error" in result:
            print(f"✗ {v}: {result['error']}")
        else:
            print(f"✓ {v}:")
            for key, val in result.items():
                if key != "file":
                    print(f"  {key:20} = {val}")
    print()
    
    print("【检查点 3】对比表")
    print("-" * 80)
    comparison = {}
    for v in versions:
        noise_r = check_noise_ratio(v, mc)
        topics_r = check_topics_and_cv(v, mc)
        
        if "error" not in noise_r and "error" not in topics_r:
            comparison[v] = {
                "Topics": topics_r.get("topic_count", "?"),
                "Mean_CV": topics_r.get("mean_cv", "?"),
                "Noise_Ratio": noise_r.get("noise_percent", "?"),
            }
    
    if comparison:
        df_comp = pd.DataFrame(comparison).T
        print(df_comp)
    else:
        print("✗ 无法生成对比表（检查文件是否存在）")
    print()
    
    print("=" * 80)
    print("验证完成")
    print("=" * 80)

if __name__ == "__main__":
    main()
