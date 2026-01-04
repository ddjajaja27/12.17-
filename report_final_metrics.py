#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提取 Baseline 和 VPD 的关键指标
"""
import pandas as pd
import json
from pathlib import Path
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def extract_metrics(version_name: str, output_base: Path = Path("ablation_outputs")):
    """提取指定版本的指标"""
    
    output_dir = output_base / version_name / "bertopic_results"
    
    results = {"version": version_name}
    
    # 1. 主题数
    topic_info_file = output_dir / "topic_info.csv"
    if topic_info_file.exists():
        df = pd.read_csv(topic_info_file)
        num_topics = len(df[df["Topic"] >= 0])
        results["num_topics"] = num_topics
    
    # 2. 噪声比例
    topics_file = output_dir / "topics.csv"
    if topics_file.exists():
        df = pd.read_csv(topics_file)
        noise_count = (df["topic"] == -1).sum()
        noise_ratio = noise_count / len(df)
        results["noise_docs"] = int(noise_count)
        results["noise_ratio"] = round(noise_ratio, 4)
        results["noise_percent"] = round(noise_ratio * 100, 2)
    
    # 3. C_V 分数
    cv_file = output_dir / "c_v_score.txt"
    if cv_file.exists():
        with open(cv_file) as f:
            cv_text = f.read().strip()
            if cv_text != "N/A":
                results["c_v_score"] = float(cv_text)
            else:
                results["c_v_score"] = None
    
    # 4. 检查 kNN mixing 指标（如果有的话）
    # 这需要计算或从已保存的指标中读取
    # 目前假设没有保存这个指标，需要计算
    
    return results

def main():
    print("\n" + "="*80)
    print("BASELINE vs VPD (M_S_B_Anatomy) - 关键指标对比")
    print("="*80 + "\n")
    
    baseline = extract_metrics("baseline")
    vpd = extract_metrics("M_S_B_Anatomy")
    
    print("[BASELINE]")
    print(f"  Version: {baseline['version']}")
    print(f"  Topics: {baseline.get('num_topics', 'N/A')}")
    print(f"  Noise documents: {baseline.get('noise_docs', 'N/A')}")
    print(f"  Noise ratio: {baseline.get('noise_ratio', 'N/A')}")
    print(f"  Noise percentage: {baseline.get('noise_percent', 'N/A')}%")
    print(f"  C_V score: {baseline.get('c_v_score', 'N/A')}")
    
    print("\n[VPD (M_S_B_Anatomy)]")
    print(f"  Version: {vpd['version']}")
    print(f"  Topics: {vpd.get('num_topics', 'N/A')}")
    print(f"  Noise documents: {vpd.get('noise_docs', 'N/A')}")
    print(f"  Noise ratio: {vpd.get('noise_ratio', 'N/A')}")
    print(f"  Noise percentage: {vpd.get('noise_percent', 'N/A')}%")
    print(f"  C_V score: {vpd.get('c_v_score', 'N/A')}")
    
    # 对比表
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    print(f"{'Metric':<30} {'Baseline':<25} {'VPD':<25}")
    print("-" * 80)
    print(f"{'Topics (mc=20)':<30} {baseline.get('num_topics', 'N/A'):<25} {vpd.get('num_topics', 'N/A'):<25}")
    print(f"{'Noise documents':<30} {baseline.get('noise_docs', 'N/A'):<25} {vpd.get('noise_docs', 'N/A'):<25}")
    print(f"{'Noise ratio':<30} {baseline.get('noise_ratio', 'N/A'):<25} {vpd.get('noise_ratio', 'N/A'):<25}")
    
    baseline_pct = baseline.get('noise_percent', 'N/A')
    vpd_pct = vpd.get('noise_percent', 'N/A')
    if isinstance(baseline_pct, (int, float)):
        baseline_pct_str = f"{baseline_pct:.2f}%"
    else:
        baseline_pct_str = str(baseline_pct)
    if isinstance(vpd_pct, (int, float)):
        vpd_pct_str = f"{vpd_pct:.2f}%"
    else:
        vpd_pct_str = str(vpd_pct)
    print(f"{'Noise percentage':<30} {baseline_pct_str:<25} {vpd_pct_str:<25}")
    
    baseline_cv = baseline.get('c_v_score', 'N/A')
    vpd_cv = vpd.get('c_v_score', 'N/A')
    print(f"{'C_V score':<30} {str(baseline_cv):<25} {str(vpd_cv):<25}")
    
    # kNN mixing 指标
    print("\n" + "="*80)
    print("AI BOARD kNN MIXING METRICS (to be calculated)")
    print("="*80)
    print("Note: kNN mixing metrics require deeper analysis of topic embeddings")
    print("These will be computed from the saved models and embeddings")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
