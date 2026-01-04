#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_ablation_experiments.py

执行消融实验：4 个版本的投影（无投影/M+S/M+S+B/全部）
对每个版本跑 BERTopic，对比 C_v、主题词质量、拓扑结构稳定性
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from ablation_experiment_config import ABLATION_CONFIGS, NoiseWordGroups
    from config import PROJECT_PREFIX
except ImportError as e:
    print(f"缺少配置模块: {e}")
    sys.exit(1)


def load_raw_embeddings() -> Tuple[np.ndarray, np.ndarray]:
    """加载原始融合向量"""
    root = Path(__file__).resolve().parent
    fused_path = root / "05_stopwords" / "Experiment_C_Vector" / "data" / "c_step1_fused_vectors.npz"
    
    if not fused_path.exists():
        raise FileNotFoundError(f"融合向量文件不存在: {fused_path}")
    
    data = np.load(fused_path, allow_pickle=True)
    return data["pmids"], data["fused_vectors"]


def build_noise_prototype(embedding_model, noise_words: List[str]) -> np.ndarray:
    """构建噪声原型向量"""
    if not noise_words:
        return None
    
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise RuntimeError("缺少 sentence_transformers，请先 pip install sentence-transformers")
    
    if embedding_model is None:
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # 编码噪声词
    noise_embeddings = embedding_model.encode(
        noise_words,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True
    )
    
    # 平均 + 归一化
    noise_prototype = np.mean(noise_embeddings, axis=0)
    noise_prototype = noise_prototype / (np.linalg.norm(noise_prototype) + 1e-10)
    
    return noise_prototype


def project_vectors(vectors: np.ndarray, noise_direction: np.ndarray) -> np.ndarray:
    """正交投影（完全移除噪声方向）"""
    if noise_direction is None:
        return vectors.copy()
    
    projection_lengths = vectors @ noise_direction
    projection_vectors = np.outer(projection_lengths, noise_direction)
    clean_vectors = vectors - projection_vectors
    
    # 重新归一化
    norms = np.linalg.norm(clean_vectors, axis=1, keepdims=True)
    clean_vectors = clean_vectors / (norms + 1e-10)
    
    return clean_vectors


def run_single_ablation(
    config_name: str,
    config: Dict[str, Any],
    pmids: np.ndarray,
    fused_vectors: np.ndarray,
    embedding_model=None,
) -> Dict[str, Any]:
    """运行单个消融配置"""
    print(f"\n{'='*70}")
    print(f"🧪 运行: {config['name']}")
    print(f"   噪声词数: {len(config['noise_words'])}")
    print(f"{'='*70}")
    
    # 构建噪声方向
    if config['noise_words']:
        noise_direction = build_noise_prototype(embedding_model, config['noise_words'])
        print(f"噪声原型范数: {np.linalg.norm(noise_direction):.4f}")
    else:
        noise_direction = None
        print("（无投影，baseline）")
    
    # 投影
    clean_vectors = project_vectors(fused_vectors, noise_direction)
    
    # 计算投影效果
    if noise_direction is not None:
        original_sim = np.mean(fused_vectors @ noise_direction)
        clean_sim = np.mean(clean_vectors @ noise_direction)
        reduction = (1 - clean_sim / original_sim) * 100 if original_sim != 0 else 0
        print(f"噪声相似度: {original_sim:.4f} → {clean_sim:.6f} (减少 {reduction:.1f}%)")
    
    # 保存投影后的向量
    output_dir = Path(__file__).resolve().parent / "ablation_outputs" / config_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    vector_file = output_dir / f"embeddings_{config_name}.npz"
    np.savez_compressed(
        vector_file,
        pmids=pmids,
        embeddings=clean_vectors.astype(np.float32),
    )
    print(f"✓ 向量已保存: {vector_file}")
    
    return {
        "config_name": config_name,
        "config_name_display": config['name'],
        "noise_words_count": len(config['noise_words']),
        "vector_file": str(vector_file),
        "original_noise_sim": float(np.mean(fused_vectors @ noise_direction)) if noise_direction is not None else None,
        "clean_noise_sim": float(np.mean(clean_vectors @ noise_direction)) if noise_direction is not None else None,
    }


def main():
    """主流程"""
    print("=" * 70)
    print("🧪 VPD 消融实验 - 向量投影阶段")
    print("=" * 70)
    
    # 加载原始向量
    print("\n加载原始融合向量...")
    pmids, fused_vectors = load_raw_embeddings()
    print(f"  文档数: {len(pmids)}")
    print(f"  维度: {fused_vectors.shape[1]}")
    
    # 加载 embedding 模型（只加载一次，后续复用）
    print("\n加载 Sentence Transformer 模型...")
    try:
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("  ✓ 模型加载成功")
    except Exception as e:
        print(f"  ✗ 加载失败: {e}")
        embedding_model = None
    
    # 运行 4 个消融配置
    results = []
    for config_name, config in ABLATION_CONFIGS.items():
        result = run_single_ablation(
            config_name,
            config,
            pmids,
            fused_vectors,
            embedding_model
        )
        results.append(result)
    
    # 保存结果汇总
    output_dir = Path(__file__).resolve().parent / "ablation_outputs"
    summary_file = output_dir / "ablation_summary.json"
    
    summary = {
        "timestamp": str(Path(__file__).resolve().parent),
        "configs": ABLATION_CONFIGS,
        "results": results,
        "total_docs": len(pmids),
        "embedding_dim": fused_vectors.shape[1],
    }
    
    with open(summary_file, "w", encoding="utf-8") as f:
        # 简化 ABLATION_CONFIGS 以便 JSON 序列化
        summary["configs"] = {
            k: {
                "name": v["name"],
                "noise_words_count": len(v["noise_words"]),
                "description": v["description"]
            }
            for k, v in ABLATION_CONFIGS.items()
        }
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 消融实验汇总已保存: {summary_file}")
    
    # 打印关键问题清单
    print("\n" + "=" * 70)
    print("📋 下一步：用这些向量跑 BERTopic")
    print("=" * 70)
    print("\n【4 个版本的向量文件】")
    for result in results:
        print(f"  {result['config_name']}: {result['vector_file']}")
    
    print("\n【关键对比维度】")
    print("  1. C_v 一致性 (原有指标)")
    print("  2. 主题词质量 (新增：是否更生物学意义)")
    print("  3. Silhouette coefficient (新增：簇的紧凑性)")
    print("  4. kNN mixing (新增：跨板块的邻居比例)")
    print("  5. 拓扑稳定性 (新增：不同 seed 下结构一致性)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
