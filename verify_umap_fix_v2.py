import pandas as pd
import json
from pathlib import Path

def check_noise_ratio(version: str, mc: int = 39) -> dict:
    """检查噪声比例（应为 (Topic == -1).count() / 31617）"""
    try:
        # 确定版本的文件夹名称
        if version.upper() == "VPD":
            folder_name = "VPD"
        else:
            folder_name = "ABLATION_baseline"
        
        # 尝试多个可能的路径
        paths = [
            Path(f"07_topic_models/{folder_name}/helicobacter_pylori_mc{mc}_doc_topic_mapping.csv"),
            Path(f"ablation_outputs/{version.lower()}/bertopic_results/helicobacter_pylori_mc{mc}_doc_topic_mapping.csv"),
        ]
        
        df = None
        found_path = None
        for path in paths:
            if path.exists():
                df = pd.read_csv(path)
                found_path = path
                break
        
        if df is None:
            return {"error": f"找不到 {version} 的 doc_topic_mapping 文件"}
        
        # 计算噪声比例
        topic_col = [col for col in df.columns if 'topic' in col.lower()][0]
        total_docs = len(df)
        noise_docs = (df[topic_col] == -1).sum()
        noise_ratio = noise_docs / total_docs if total_docs > 0 else 0
        noise_percent = noise_ratio * 100
        
        return {
            "version": version,
            "mc": mc,
            "file": str(found_path),
            "total_docs": total_docs,
            "noise_docs": int(noise_docs),
            "noise_ratio": round(noise_ratio, 4),
            "noise_percent": round(noise_percent, 2),
            "correct_calc": f"{int(noise_docs)} / {total_docs} = {round(noise_ratio, 4)}"
        }
    except Exception as e:
        return {"error": str(e), "version": version}

def check_topics_and_cv(version: str, mc: int = 39) -> dict:
    """检查主题数和 C_V 得分"""
    try:
        # 确定版本的文件夹名称
        if version.upper() == "VPD":
            folder_name = "VPD"
        else:
            folder_name = "ABLATION_baseline"
        
        # 尝试多个可能的路径
        topic_paths = [
            Path(f"07_topic_models/{folder_name}/helicobacter_pylori_mc{mc}_topic_info.csv"),
            Path(f"ablation_outputs/{version.lower()}/bertopic_results/helicobacter_pylori_mc{mc}_topic_info.csv"),
        ]
        
        df = None
        found_path = None
        for path in topic_paths:
            if path.exists():
                df = pd.read_csv(path)
                found_path = path
                break
        
        if df is None:
            return {"error": f"找不到 {version} 的 topic_info 文件", "version": version}
        
        # 获取主题数（Topic 列中不包括 -1）
        if 'Topic' in df.columns:
            topic_count = len(df[df['Topic'] != -1])
        else:
            topic_count = len(df) - 1
        
        # 尝试从 c_v_score.txt 读取 C_V 分数
        cv_paths = [
            Path(f"07_topic_models/{folder_name}/c_v_score.txt"),
            Path(f"ablation_outputs/{version.lower()}/bertopic_results/c_v_score.txt"),
        ]
        
        mean_cv = None
        for path in cv_paths:
            if path.exists():
                try:
                    with open(path) as f:
                        mean_cv = float(f.read().strip())
                    break
                except:
                    pass
        
        return {
            "version": version,
            "mc": mc,
            "file": str(found_path),
            "topic_count": topic_count,
            "mean_cv": round(mean_cv, 4) if mean_cv is not None else "N/A",
        }
    except Exception as e:
        return {"error": str(e), "version": version}

def main():
    print("\n" + "="*70)
    print("="*15 + " UMAP n_components=5 修复验证 "*2 + "="*10)
    print("="*70)
    
    # 检查点 1：噪声比例
    print("\n【检查点 1】噪声比例（mc=39）")
    print("-" * 70)
    
    for version in ["baseline", "VPD"]:
        result = check_noise_ratio(version, 39)
        if "error" not in result:
            print(f"\n✓ {version}:")
            for key, value in result.items():
                print(f"  {key:20s} = {value}")
        else:
            print(f"\n✗ {version}: {result['error']}")
    
    # 检查点 2：主题数和平均 C_V
    print("\n\n【检查点 2】主题数和平均C_V（mc=39）")
    print("-" * 70)
    
    for version in ["baseline", "VPD"]:
        result = check_topics_and_cv(version, 39)
        if "error" not in result:
            print(f"\n✓ {version}:")
            for key, value in result.items():
                print(f"  {key:20s} = {value}")
        else:
            print(f"\n✗ {version}: {result['error']}")
    
    # 检查点 3：对比表
    print("\n\n【检查点 3】对比表")
    print("-" * 70)
    baseline_noise = check_noise_ratio("baseline", 39)
    vpd_noise = check_noise_ratio("VPD", 39)
    baseline_cv = check_topics_and_cv("baseline", 39)
    vpd_cv = check_topics_and_cv("VPD", 39)
    
    if "error" not in baseline_noise and "error" not in vpd_noise:
        print(f"\n{'指标':<25} {'Baseline':<25} {'VPD':<25}")
        print("-" * 70)
        print(f"{'噪声文档数':<25} {baseline_noise['noise_docs']:<25} {vpd_noise['noise_docs']:<25}")
        print(f"{'噪声比例':<25} {baseline_noise['noise_ratio']:<25} {vpd_noise['noise_ratio']:<25}")
        print(f"{'噪声百分比':<25} {baseline_noise['noise_percent']:.2f}%{'':<22} {vpd_noise['noise_percent']:.2f}%")
        
        if "error" not in baseline_cv and "error" not in vpd_cv:
            print(f"{'主题数（mc=39）':<25} {baseline_cv['topic_count']:<25} {vpd_cv['topic_count']:<25}")
            print(f"{'平均C_V':<25} {str(baseline_cv['mean_cv']):<25} {str(vpd_cv['mean_cv']):<25}")
    else:
        print("✗ 无法生成对比表（检查文件是否存在）")
    
    print("\n" + "="*70)
    print("="*20 + " 验证完成 " + "="*39)
    print("="*70)

if __name__ == "__main__":
    main()
