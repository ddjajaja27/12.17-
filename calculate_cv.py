"""
从已有数据计算结构化指标
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def calculate_structural_metrics(version: str, mc: int = 39):
    """计算结构化指标"""
    
    if version.upper() == "VPD":
        model_dir = Path(f"07_topic_models/VPD")
    else:
        model_dir = Path(f"07_topic_models/ABLATION_baseline")
    
    output_file = model_dir / f"helicobacter_pylori_mc{mc}_doc_topic_mapping.csv"
    topic_info_file = model_dir / f"helicobacter_pylori_mc{mc}_topic_info.csv"
    
    if not output_file.exists():
        print(f"❌ 文件不存在: {output_file}")
        return None
    
    print(f"📊 分析 {version} (mc={mc})")
    
    try:
        # 读取数据
        df = pd.read_csv(output_file)
        topic_info = pd.read_csv(topic_info_file)
        
        # 找到主题列
        topic_col = None
        for col in df.columns:
            if 'topic' in col.lower() and col != 'primary_topic':
                topic_col = col
                break
        
        if not topic_col:
            topic_col = 'Topic' if 'Topic' in df.columns else df.columns[0]
        
        # 基础指标
        total_docs = len(df)
        topic_count = len(topic_info[topic_info['Topic'] != -1]) if 'Topic' in topic_info.columns else len(topic_info) - 1
        noise_docs = (df[topic_col] == -1).sum() if topic_col in df.columns else 0
        noise_ratio = noise_docs / total_docs
        
        results = {
            "version": version,
            "mc": mc,
            "total_docs": total_docs,
            "topic_count": topic_count,
            "noise_docs": int(noise_docs),
            "noise_ratio": round(noise_ratio, 4),
            "noise_percent": round(noise_ratio * 100, 2),
        }
        
        print(f"  ✓ 主题数: {topic_count}")
        print(f"  ✓ 噪声比例: {results['noise_percent']:.2f}%")
        
        return results
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("\n" + "="*70)
    print("="*15 + " 结构化指标计算" + "="*40)
    print("="*70 + "\n")
    
    results = {}
    
    for version in ["baseline", "VPD"]:
        for mc in [39]:
            r = calculate_structural_metrics(version, mc)
            if r:
                results[version] = r
            print()
    
    # 生成对比表
    if len(results) >= 2:
        print("="*70)
        print("="*15 + " 对比表" + "="*50)
        print("="*70)
        
        baseline = results.get("baseline", {})
        vpd = results.get("VPD", {})
        
        print(f"\n{'指标':<25} {'Baseline':<25} {'VPD':<25}")
        print("-" * 70)
        print(f"{'主题数（mc=39）':<25} {baseline.get('topic_count', 'N/A'):<25} {vpd.get('topic_count', 'N/A'):<25}")
        print(f"{'噪声文档数':<25} {baseline.get('noise_docs', 'N/A'):<25} {vpd.get('noise_docs', 'N/A'):<25}")
        print(f"{'噪声比例':<25} {baseline.get('noise_ratio', 'N/A'):<25} {vpd.get('noise_ratio', 'N/A'):<25}")
        print(f"{'噪声百分比':<25} {baseline.get('noise_percent', 'N/A'):.2f}%{'':<15} {vpd.get('noise_percent', 'N/A'):.2f}%")
        
        # 分析结果
        print("\n" + "="*70)
        print("="*15 + " 分析结果" + "="*45)
        print("="*70)
        
        if baseline.get('topic_count') and vpd.get('topic_count'):
            if vpd['topic_count'] < baseline['topic_count']:
                print(f"✓ VPD 主题数减少: {baseline['topic_count']} → {vpd['topic_count']} (-{baseline['topic_count'] - vpd['topic_count']})")
            else:
                print(f"ℹ️  VPD 主题数增加: {baseline['topic_count']} → {vpd['topic_count']} (+{vpd['topic_count'] - baseline['topic_count']})")
        
        if baseline.get('noise_percent') is not None and vpd.get('noise_percent') is not None:
            diff = baseline['noise_percent'] - vpd['noise_percent']
            if diff > 0:
                print(f"✓ VPD 噪声减少: {baseline['noise_percent']:.2f}% → {vpd['noise_percent']:.2f}% (-{abs(diff):.2f}%)")
            else:
                print(f"ℹ️  VPD 噪声增加: {baseline['noise_percent']:.2f}% → {vpd['noise_percent']:.2f}% (+{abs(diff):.2f}%)")

if __name__ == "__main__":
    main()
