#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
from pathlib import Path
import sys
import io

# 设置输出编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def check_noise_ratio(version: str, mc: int = 39) -> dict:
    """检查噪声比例"""
    try:
        if version.upper() == "VPD":
            folder_name = "VPD"
        else:
            folder_name = "ABLATION_baseline"
        
        paths = [
            Path(f"07_topic_models/{folder_name}/helicobacter_pylori_mc{mc}_doc_topic_mapping.csv"),
        ]
        
        df = None
        found_path = None
        for path in paths:
            if path.exists():
                df = pd.read_csv(path)
                found_path = path
                break
        
        if df is None:
            return {"error": f"Not found: {version}"}
        
        topic_col = [col for col in df.columns if 'topic' in col.lower()][0]
        total_docs = len(df)
        noise_docs = (df[topic_col] == -1).sum()
        noise_ratio = noise_docs / total_docs if total_docs > 0 else 0
        noise_percent = noise_ratio * 100
        
        return {
            "version": version,
            "mc": mc,
            "total_docs": total_docs,
            "noise_docs": int(noise_docs),
            "noise_ratio": round(noise_ratio, 4),
            "noise_percent": round(noise_percent, 2),
        }
    except Exception as e:
        return {"error": str(e), "version": version}

def check_topics(version: str, mc: int = 39) -> dict:
    """检查主题数"""
    try:
        if version.upper() == "VPD":
            folder_name = "VPD"
        else:
            folder_name = "ABLATION_baseline"
        
        topic_paths = [
            Path(f"07_topic_models/{folder_name}/helicobacter_pylori_mc{mc}_topic_info.csv"),
        ]
        
        df = None
        for path in topic_paths:
            if path.exists():
                df = pd.read_csv(path)
                break
        
        if df is None:
            return {"error": f"Not found: {version}", "version": version}
        
        if 'Topic' in df.columns:
            topic_count = len(df[df['Topic'] != -1])
        else:
            topic_count = len(df) - 1
        
        return {
            "version": version,
            "mc": mc,
            "topic_count": topic_count,
        }
    except Exception as e:
        return {"error": str(e), "version": version}

def main():
    print("\n" + "="*70)
    print("UMAP n_components=5 VERIFICATION REPORT")
    print("="*70)
    
    # Baseline
    print("\n[BASELINE RESULTS]")
    baseline_noise = check_noise_ratio("baseline", 39)
    baseline_topic = check_topics("baseline", 39)
    
    if "error" not in baseline_noise:
        print(f"  Total documents: {baseline_noise['total_docs']}")
        print(f"  Noise documents: {baseline_noise['noise_docs']}")
        print(f"  Noise ratio: {baseline_noise['noise_ratio']}")
        print(f"  Noise percentage: {baseline_noise['noise_percent']:.2f}%")
    
    if "error" not in baseline_topic:
        print(f"  Topic count (mc=39): {baseline_topic['topic_count']}")
    
    # VPD
    print("\n[VPD RESULTS]")
    vpd_noise = check_noise_ratio("VPD", 39)
    vpd_topic = check_topics("VPD", 39)
    
    if "error" not in vpd_noise:
        print(f"  Total documents: {vpd_noise['total_docs']}")
        print(f"  Noise documents: {vpd_noise['noise_docs']}")
        print(f"  Noise ratio: {vpd_noise['noise_ratio']}")
        print(f"  Noise percentage: {vpd_noise['noise_percent']:.2f}%")
    
    if "error" not in vpd_topic:
        print(f"  Topic count (mc=39): {vpd_topic['topic_count']}")
    
    # Comparison
    print("\n[COMPARISON TABLE]")
    print("-" * 70)
    print(f"{'Metric':<30} {'Baseline':<20} {'VPD':<20}")
    print("-" * 70)
    
    if "error" not in baseline_noise and "error" not in vpd_noise:
        print(f"{'Noise documents':<30} {baseline_noise['noise_docs']:<20} {vpd_noise['noise_docs']:<20}")
        print(f"{'Noise ratio':<30} {baseline_noise['noise_ratio']:<20} {vpd_noise['noise_ratio']:<20}")
        print(f"{'Noise percentage':<30} {baseline_noise['noise_percent']:.2f}%{'':<15} {vpd_noise['noise_percent']:.2f}%")
    
    if "error" not in baseline_topic and "error" not in vpd_topic:
        print(f"{'Topics (mc=39)':<30} {baseline_topic['topic_count']:<20} {vpd_topic['topic_count']:<20}")
    
    # Analysis
    print("\n[ANALYSIS]")
    print("-" * 70)
    
    if "error" not in baseline_noise and "error" not in vpd_noise:
        noise_diff = baseline_noise['noise_docs'] - vpd_noise['noise_docs']
        if noise_diff > 0:
            print(f"GOOD: VPD noise reduced by {noise_diff} documents ({abs(baseline_noise['noise_percent'] - vpd_noise['noise_percent']):.2f}%)")
        else:
            print(f"NOTE: VPD noise changed by {noise_diff} documents")
    
    if "error" not in baseline_topic and "error" not in vpd_topic:
        topic_diff = baseline_topic['topic_count'] - vpd_topic['topic_count']
        if topic_diff != 0:
            print(f"NOTE: Topic count changed by {topic_diff}")
    
    print("\n" + "="*70)
    print("VERIFICATION COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
