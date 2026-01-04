#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
monitor_ablation_progress.py
监控消融实验的 BERTopic 处理进度

实时显示：
1. 哪些版本已完成
2. 哪些版本进行中
3. 完整性汇总
"""

from pathlib import Path
import pandas as pd
import numpy as np
import time

versions = ["baseline", "M_S", "M_S_B", "M_S_B_Anatomy"]
PROJECT_PREFIX = "helicobacter_pylori"

def check_version_status(version):
    """检查版本处理状态"""
    output_dir = Path(f"07_topic_models/ABLATION_{version}")
    
    # 查找 mc*_topic_info.csv 文件
    topic_files = sorted(output_dir.glob("*_topic_info.csv"))
    
    if not topic_files:
        return {"status": "not_started"}
    
    result = {"status": "completed", "files": []}
    
    for topic_file in topic_files:
        try:
            df = pd.read_csv(topic_file)
            mc = int(topic_file.name.split("_mc")[1].split("_")[0])
            num_topics = int((df["Topic"] >= 0).sum())
            
            # 查找对应的 doc_topic_mapping.csv
            doc_topic_file = output_dir / f"{PROJECT_PREFIX}_mc{mc}_doc_topic_mapping.csv"
            if doc_topic_file.exists():
                df_docs = pd.read_csv(doc_topic_file)
                num_noise = int((df_docs["Topic"] == -1).sum())
                noise_ratio = num_noise / len(df_docs)
            else:
                num_noise = -1
                noise_ratio = -1
            
            result["files"].append({
                "mc": mc,
                "num_topics": num_topics,
                "num_noise": num_noise,
                "noise_ratio": noise_ratio,
            })
        except Exception as e:
            pass
    
    # 优先使用 mc39（标准参数）
    mc39_file = [f for f in result["files"] if f["mc"] == 39]
    if mc39_file:
        result["primary"] = mc39_file[0]
    elif result["files"]:
        result["primary"] = result["files"][-1]  # 用最后一个（通常是 mc22）
    else:
        result["status"] = "incomplete"
    
    return result

def main():
    print("\n" + "="*70)
    print("消融实验 BERTopic 处理监控")
    print("="*70 + "\n")
    
    # 检查向量文件
    print("【向量文件状态】")
    for version in versions:
        vector_file = Path(f"ablation_outputs/{version}/c_{version}_final_clean_vectors.npz")
        if vector_file.exists():
            size = vector_file.stat().st_size / (1024*1024)
            print(f"  ✓ {version:20s} {size:6.1f} MB")
        else:
            print(f"  ✗ {version:20s} 缺失")
    
    print("\n【BERTopic 处理进度】\n")
    
    completed = 0
    summary = []
    
    for version in versions:
        status = check_version_status(version)
        
        if status["status"] == "completed" and "primary" in status:
            completed += 1
            primary = status["primary"]
            print(f"  ✓ {version:20s} mc={primary['mc']} topics={primary['num_topics']:4d} noise={primary['noise_ratio']*100:5.2f}%")
            summary.append({
                "版本": version,
                "状态": "✓ 完成",
                "mc": primary['mc'],
                "主题数": primary['num_topics'],
                "噪声比例": f"{primary['noise_ratio']*100:.2f}%",
            })
        elif status["status"] == "incomplete":
            print(f"  ⏳ {version:20s} 部分完成 ({len(status['files'])} 个 mc 参数)")
            summary.append({
                "版本": version,
                "状态": "⏳ 进行中",
                "mc": "-",
                "主题数": "-",
                "噪声比例": "-",
            })
        else:
            print(f"  ⏳ {version:20s} 未开始")
            summary.append({
                "版本": version,
                "状态": "⏳ 等待中",
                "mc": "-",
                "主题数": "-",
                "噪声比例": "-",
            })
    
    print(f"\n【总体进度】 {completed}/{len(versions)} 版本完成\n")
    
    # 汇总表
    if summary:
        summary_df = pd.DataFrame(summary)
        print(summary_df.to_string(index=False))
    
    print(f"\n【下一步】")
    if completed == len(versions):
        print("  ✓ 所有版本 BERTopic 处理已完成")
        print("  执行：python compute_ablation_structural_metrics.py")
    else:
        remaining = len(versions) - completed
        print(f"  ⏳ 等待剩余 {remaining} 个版本完成...")
        print(f"  预计时间：~{remaining * 30} 分钟")
    
    print()

if __name__ == "__main__":
    main()
