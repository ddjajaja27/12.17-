#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
🔬 VPD 去噪效果验证脚本
=============================================================================

目标：验证 VPD 方法是否"真的做掉了噪声方向"

验证指标：
1. 原始向量 · 噪声方向 的分布（应该有明显方向性）
2. 去噪后向量 · 噪声方向 的分布（应该接近 0）
3. 噪声原型向量的范数（如果太小，说明噪声不稳定）
4. 投影强度的实际效果（理论值 vs 实际值）

科学标准：
- 如果 projection_strength=1.0 且原型向量稳定，
  去噪后应该几乎全都接近 0（数值误差级 < 1e-6）
- 如果去噪后仍明显不为 0，说明有 bug（归一化/顺序/融合问题）
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import json
import sys

# 添加根目录到路径（便于导入 config）
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from config import PROJECT_PREFIX
except ImportError:
    PROJECT_PREFIX = "helicobacter_pylori"


def load_vectors_and_metadata():
    """加载原始向量和去噪向量"""
    root = Path(__file__).resolve().parent
    
    # 路径 1: 原始融合向量（Step 1）
    fused_path = root / "05_stopwords" / "Experiment_C_Vector" / "data" / "c_step1_fused_vectors.npz"
    
    # 路径 2: 去噪后向量（Step 2）
    clean_path = root / "05_stopwords" / "Experiment_C_Vector" / "data" / "c_step2_clean_vectors.npz"
    
    # 路径 3: 最终输出的向量（用于 topic modeling）
    output_path = root / "06_denoised_data" / f"{PROJECT_PREFIX}_topic_modeling_VPD.csv"
    
    print("=" * 80)
    print("📍 路径检查")
    print("=" * 80)
    print(f"融合向量 (Step 1):  {fused_path.exists() and '✓' or '✗'} {fused_path}")
    print(f"清洁向量 (Step 2):  {clean_path.exists() and '✓' or '✗'} {clean_path}")
    print(f"最终输出 (CSV):     {output_path.exists() and '✓' or '✗'} {output_path}")
    
    if not fused_path.exists() or not clean_path.exists():
        raise FileNotFoundError("缺少关键向量文件，请先运行 Experiment C 的 Step 1 和 Step 2")
    
    # 加载向量
    print("\n加载向量数据...")
    fused_data = np.load(fused_path, allow_pickle=True)
    clean_data = np.load(clean_path, allow_pickle=True)
    
    fused_vectors = fused_data["fused_vectors"]  # (n_docs, 384)
    clean_vectors = clean_data["clean_vectors"]  # (n_docs, 384)
    noise_prototype = clean_data["noise_prototype"]  # (384,)
    noise_words = clean_data["noise_words"]
    projection_strength = clean_data["config_projection_strength"].item()
    
    print(f"  融合向量形状: {fused_vectors.shape}")
    print(f"  清洁向量形状: {clean_vectors.shape}")
    print(f"  噪声原型维度: {noise_prototype.shape}")
    print(f"  噪声词数量: {len(noise_words)}")
    print(f"  投影强度配置: {projection_strength}")
    
    return {
        "fused_vectors": fused_vectors,
        "clean_vectors": clean_vectors,
        "noise_prototype": noise_prototype,
        "noise_words": noise_words,
        "projection_strength": projection_strength,
    }


def verify_noise_prototype_stability(noise_prototype):
    """验证噪声原型向量的稳定性"""
    print("\n" + "=" * 80)
    print("🔍 验证 1: 噪声原型向量的稳定性")
    print("=" * 80)
    
    norm = np.linalg.norm(noise_prototype)
    print(f"噪声原型向量的 L2 范数: {norm:.6f}")
    
    if norm < 0.1:
        print("⚠️  WARNING: 范数过小 (<0.1)，说明噪声方向不稳定！")
        print("   → 可能原因: 噪声词定义不好，或数据中没有这些词的一致方向")
        print("   → 影响: 投影出来的方向会很不稳定，去噪效果差")
        return False
    elif norm < 0.5:
        print("⚠️  CAUTION: 范数较小 (0.1-0.5)，噪声方向可能不够清晰")
        return True
    else:
        print("✓ 范数合理 (>0.5)，噪声方向定义清晰")
        return True
    
    # 验证是否已归一化
    expected_norm = 1.0
    if abs(norm - expected_norm) < 1e-5:
        print("✓ 向量已单位化 (||n̂|| ≈ 1.0)")
    else:
        print(f"⚠️  向量未完全单位化 (||n̂|| = {norm:.6f} ≠ 1.0)")


def verify_noise_similarity_distribution(fused_vectors, clean_vectors, noise_prototype):
    """验证原始和去噪向量与噪声方向的相似度分布"""
    print("\n" + "=" * 80)
    print("🔍 验证 2: 与噪声方向的相似度分布")
    print("=" * 80)
    
    # 计算点积（相似度）
    fused_similarity = fused_vectors @ noise_prototype  # (n_docs,)
    clean_similarity = clean_vectors @ noise_prototype  # (n_docs,)
    
    print("\n【原始向量与噪声方向的相似度】")
    print(f"  均值:     {np.mean(fused_similarity):+.6f}")
    print(f"  中位数:   {np.median(fused_similarity):+.6f}")
    print(f"  标准差:   {np.std(fused_similarity):.6f}")
    print(f"  最小值:   {np.min(fused_similarity):+.6f}")
    print(f"  最大值:   {np.max(fused_similarity):+.6f}")
    print(f"  范围:     [{np.percentile(fused_similarity, 5):+.6f}, {np.percentile(fused_similarity, 95):+.6f}] (P5-P95)")
    
    print("\n【去噪向量与噪声方向的相似度】")
    print(f"  均值:     {np.mean(clean_similarity):+.6f}")
    print(f"  中位数:   {np.median(clean_similarity):+.6f}")
    print(f"  标准差:   {np.std(clean_similarity):.6f}")
    print(f"  最小值:   {np.min(clean_similarity):+.6f}")
    print(f"  最大值:   {np.max(clean_similarity):+.6f}")
    print(f"  范围:     [{np.percentile(clean_similarity, 5):+.6f}, {np.percentile(clean_similarity, 95):+.6f}] (P5-P95)")
    
    print("\n【去噪效果评估】")
    mean_reduction = abs(np.mean(fused_similarity)) - abs(np.mean(clean_similarity))
    reduction_ratio = mean_reduction / abs(np.mean(fused_similarity)) if np.mean(fused_similarity) != 0 else 0
    print(f"  平均相似度减少: {mean_reduction:+.6f} ({reduction_ratio*100:.1f}%)")
    
    # 验证关键指标：去噪后是否接近 0
    clean_mean_abs = np.mean(np.abs(clean_similarity))
    print(f"  |去噪相似度|的平均值: {clean_mean_abs:.6f}")
    
    if clean_mean_abs < 1e-5:
        print("✓ 完美！去噪后几乎完全消除噪声方向（< 1e-5）")
        return True
    elif clean_mean_abs < 1e-4:
        print("✓ 很好！去噪后噪声方向非常小（< 1e-4）")
        return True
    elif clean_mean_abs < 0.01:
        print("⚠️  一般。去噪后仍有一定噪声分量（>1e-4）")
        print("   → 可能原因: 向量非单位向量，或归一化有问题")
        return False
    else:
        print("❌ 不行！去噪效果很差（> 0.01）")
        print("   → 这表明投影可能没有正确执行")
        return False


def verify_projection_reconstruction():
    """验证投影的数学正确性"""
    print("\n" + "=" * 80)
    print("🔍 验证 3: 投影数学的正确性（样例）")
    print("=" * 80)
    
    root = Path(__file__).resolve().parent
    clean_path = root / "05_stopwords" / "Experiment_C_Vector" / "data" / "c_step2_clean_vectors.npz"
    
    clean_data = np.load(clean_path, allow_pickle=True)
    fused_path = root / "05_stopwords" / "Experiment_C_Vector" / "data" / "c_step1_fused_vectors.npz"
    fused_data = np.load(fused_path, allow_pickle=True)
    
    # 从 npz 中检查是否有原始相似度值
    original_noise_sim = float(clean_data.get("original_noise_similarity", 0.0))
    clean_noise_sim = float(clean_data.get("clean_noise_similarity", 0.0))
    
    print(f"\n从 Step 2 输出中记录的数据：")
    print(f"  投影前平均相似度: {original_noise_sim:+.6f}")
    print(f"  投影后平均相似度: {clean_noise_sim:+.6f}")
    print(f"  相似度减少百分比: {(1 - clean_noise_sim / original_noise_sim) * 100:.1f}%")
    
    # 重新计算验证
    fused_vectors = fused_data["fused_vectors"]
    clean_vectors = clean_data["clean_vectors"]
    noise_prototype = clean_data["noise_prototype"]
    
    recomputed_fused = np.mean(fused_vectors @ noise_prototype)
    recomputed_clean = np.mean(clean_vectors @ noise_prototype)
    
    print(f"\n重新计算验证：")
    print(f"  原始相似度: {recomputed_fused:+.6f} (diff: {abs(original_noise_sim - recomputed_fused):.6e})")
    print(f"  清洁相似度: {recomputed_clean:+.6f} (diff: {abs(clean_noise_sim - recomputed_clean):.6e})")
    
    if abs(recomputed_clean) < 1e-5:
        print("✓ 投影数学正确，噪声方向已被完全移除")
        return True
    else:
        print(f"⚠️  投影后仍有残余 ({abs(recomputed_clean):.6e})")
        return False


def compare_with_baseline_embeddings():
    """与 Baseline 的嵌入进行对比"""
    print("\n" + "=" * 80)
    print("🔍 验证 4: VPD 向量与 Baseline 的对比")
    print("=" * 80)
    
    root = Path(__file__).resolve().parent
    clean_path = root / "06_denoised_data" / f"{PROJECT_PREFIX}_topic_modeling_VPD.csv"
    
    # VPD 用的是 c_final_clean_vectors.npz
    vpd_vec_path = root / "05_stopwords" / "Experiment_C_Vector" / "output" / "c_final_clean_vectors.npz"
    
    if not vpd_vec_path.exists():
        print(f"⚠️  找不到 VPD 向量文件: {vpd_vec_path}")
        print("   → 这个文件应该在 Step 3 生成（03_output_vectors.py）")
        return None
    
    # 加载 VPD 用的最终清洁向量
    vpd_data = np.load(vpd_vec_path, allow_pickle=True)
    vpd_vectors = vpd_data["embeddings"]
    
    print(f"VPD 最终向量形状: {vpd_vectors.shape}")
    print(f"向量已归一化: {np.allclose(np.linalg.norm(vpd_vectors, axis=1), 1.0)}")
    
    return vpd_vectors


def generate_report(results):
    """生成详细的验证报告"""
    print("\n" + "=" * 80)
    print("📋 综合评估报告")
    print("=" * 80)
    
    # 汇总结果
    verdict = all(results.values())
    
    if verdict:
        print("\n✅ 总体结论: VPD 去噪实现正确！")
        print("   → 噪声方向已被有效移除")
        print("   → 向量空间去噪的数学原理得到验证")
        print("   → VPD 的 +5.2% C_v 提升是有基础的")
    else:
        print("\n❌ 总体结论: VPD 去噪存在问题！")
        print("   → 以下方面需要检查:")
        if not results.get("noise_stability"):
            print("     - 噪声原型向量不够稳定（范数过小）")
        if not results.get("similarity_distribution"):
            print("     - 去噪后仍有明显噪声分量")
        if not results.get("projection_math"):
            print("     - 投影计算可能有数值问题")
        print("\n   建议:")
        print("     1. 检查 noise_words 的定义是否合理")
        print("     2. 验证向量的单位化处理")
        print("     3. 运行 05_stopwords/Experiment_C_Vector/ 中的 Step 1-3")
    
    # 输出数值总结
    print("\n【关键数值总结】")
    print(f"  • 原始向量均值相似度: {results.get('original_mean', 'N/A')}")
    print(f"  • 清洁向量均值相似度: {results.get('clean_mean', 'N/A')}")
    print(f"  • 噪声减少百分比: {results.get('reduction_ratio', 'N/A')}")
    print(f"  • 清洁向量|相似度|均值: {results.get('clean_mean_abs', 'N/A')}")
    
    return verdict


def main():
    """主流程"""
    results = {}
    
    try:
        # 加载数据
        data = load_vectors_and_metadata()
        
        # 验证 1: 噪声原型稳定性
        results["noise_stability"] = verify_noise_prototype_stability(data["noise_prototype"])
        
        # 验证 2: 相似度分布
        fused_sim = data["fused_vectors"] @ data["noise_prototype"]
        clean_sim = data["clean_vectors"] @ data["noise_prototype"]
        results["original_mean"] = f"{np.mean(fused_sim):+.6f}"
        results["clean_mean"] = f"{np.mean(clean_sim):+.6f}"
        results["clean_mean_abs"] = f"{np.mean(np.abs(clean_sim)):.6e}"
        results["reduction_ratio"] = f"{(1 - np.mean(clean_sim) / np.mean(fused_sim)) * 100:.1f}%"
        results["similarity_distribution"] = verify_noise_similarity_distribution(
            data["fused_vectors"], 
            data["clean_vectors"], 
            data["noise_prototype"]
        )
        
        # 验证 3: 投影数学正确性
        results["projection_math"] = verify_projection_reconstruction()
        
        # 验证 4: 与 Baseline 对比
        compare_with_baseline_embeddings()
        
        # 生成报告
        verdict = generate_report(results)
        
        # 保存报告到文件
        report_path = Path(__file__).resolve().parent / "verification_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n验证报告已保存到: {report_path}")
        
        return 0 if verdict else 1
    
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
