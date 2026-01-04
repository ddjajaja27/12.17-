#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_engine_bertopic.py
BERTopic 主题建模引擎（被 step07_topic_model.py 调用）

功能：
- 接收去噪后的文本数据，执行 BERTopic 主题建模
- 支持预计算 embedding（C方案向量投影）
- 自适应 min_cluster_size 计算
- 多阶段 reduce_outliers 降噪

参考文献：
[1] Campello, R. J., Moulavi, D., & Sander, J. (2013). Density-based clustering based on 
    hierarchical density estimates. PAKDD 2013. (HDBSCAN min_cluster_size 选择)
[2] McInnes, L., Healy, J., & Astels, S. (2017). hdbscan: Hierarchical density based clustering. 
    JOSS. (min_cluster_size 经验公式: sqrt(N) 作为上界)
[3] Small, H., Boyack, K. W., & Klavans, R. (2014). Identifying emerging topics in science 
    and technology. Research Policy, 43(8), 1450-1467. (新兴主题识别)
[4] Chen, C. (2006). CiteSpace II: Detecting and visualizing emerging trends and transient 
    patterns in scientific literature. JASIST. (研究前沿分类)
[5] Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation 
    of cluster analysis. Journal of Computational and Applied Mathematics. (聚类质量评估)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, cast


import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP
except ImportError:
    print("请安装必要的包: pip install bertopic sentence-transformers umap-learn hdbscan")
    sys.exit(1)

try:
    from config import PATHS, MODEL_CONFIG, PROJECT_PREFIX, get_project_name
except ImportError:
    print("请确保 config.py 存在且配置正确")
    sys.exit(1)

# 说明：原始版本依赖 05_denoise/main/denoise_controller.py。
# 本仓库已将第5步目录统一为 05_stopwords/，为保证流程可运行，这里内置一个最小 read_json。
import json


def read_json(path):
    """最小 JSON 读取函数（兼容旧脚本依赖）"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_custom_stopwords(base_dir: Path, method: str) -> list:
    """
    加载停用词列表。
    
    设计说明（2025-12-28 简化版）：
    - 项目已简化为只有 baseline 和 VPD 两种方法
    - 两者都只使用 sklearn 基础英文停用词（318 词）
    - VPD 的"去噪"体现在 embedding 层面（通过 --embedding_npz 传入预计算的清洁向量）
    - 停用词仅用于 CountVectorizer 提取主题关键词时过滤 the/is/and 等功能词
    
    A/B/AB/ABC 方法已归档于 07_topic_models/_archived_AB_methods
    """
    # 仅加载 sklearn 内置英文停用词
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    stopwords = set(ENGLISH_STOP_WORDS)
    
    return sorted(list(stopwords))


@dataclass(frozen=True)
class FrontierConfig:
    """研究前沿识别配置"""
    model_name: str = "all-MiniLM-L6-v2"
    min_cluster_size: int = 60
    ngram_range: tuple = (1, 1)
    recent_years: int = 3
    high_cited_quantile: float = 0.99


class TopicModeler:
    """主题建模与研究前沿识别"""

    def __init__(self):
        import argparse, os, json, re
        parser = argparse.ArgumentParser(description="BERTopic主题建模主控脚本")
        parser.add_argument('--input', type=str, default=None, help='指定输入csv文件（去噪产物）')
        parser.add_argument('--output_dir', type=str, default=None, help='指定输出目录')
        parser.add_argument('--model_name', type=str, default=None, help='指定embedding模型名')
        parser.add_argument('--min_cluster_size', type=int, default=None, help='指定min_cluster_size')
        parser.add_argument('--embedding_npz', type=str, default=None, help='指定预计算embedding文件')
        args, _ = parser.parse_known_args()

        # ========== 1. 确定输入文件（优先级：命令行 > 环境变量 > lock > denoised > fallback） ==========
        self.embedding_npz: Optional[Path] = None
        input_file_arg = args.input or os.environ.get('TOPIC_MODEL_INPUT', None)
        fallback = Path(PATHS["file_04_topic"])
        base_dir = Path(__file__).resolve().parent
        denoise_dir = base_dir / "05_denoise"
        
        if input_file_arg and Path(input_file_arg).exists():
            self.input_file = Path(input_file_arg)
            
            # 关键修复：从 manifest 中查找对应的 embedding_npz
            manifest_path = denoise_dir / "denoised_data" / f"{PROJECT_PREFIX}_denoise_manifest_part1.json"
            if manifest_path.exists():
                try:
                    manifest = read_json(manifest_path)
                    candidates = manifest.get("candidates", [])
                    input_basename = self.input_file.name
                    for cand in candidates:
                        cand_file = cand.get("file", "")
                        if Path(cand_file).name == input_basename:
                            npz = cand.get("embedding_npz")
                            if npz:
                                npz_path = denoise_dir / npz
                                if npz_path.exists():
                                    self.embedding_npz = npz_path
                                    print(f"[INFO] 从 manifest 加载 C 方案清洁向量: {npz_path}")
                            break
                except Exception as e:
                    print(f"[WARN] 读取 manifest 失败: {e}")
        else:
            self.input_file = fallback
            # 尝试从lock读取
            lock_path_str = PATHS.get("file_05_denoise_lock", "")
            if lock_path_str:
                lock_path = Path(lock_path_str)
                if lock_path.exists():
                    try:
                        lock = read_json(lock_path)
                        cand = lock.get("candidate") or {}
                        cand_file = cand.get("file")
                        if cand_file and Path(str(cand_file)).exists():
                            self.input_file = Path(str(cand_file))
                        npz = cand.get("embedding_npz")
                        if npz:
                            npz_path = base_dir / str(npz)
                            if npz_path.exists():
                                self.embedding_npz = npz_path
                    except Exception:
                        pass
            # 尝试标准去噪文件
            if self.input_file == fallback:
                denoised_str = PATHS.get("file_05_denoised_topic", "")
                if denoised_str:
                    denoised = Path(denoised_str)
                    if denoised.exists():
                        self.input_file = denoised

        # 命令行/环境变量指定的 embedding_npz 优先级最高
        if args.embedding_npz:
            npz_path = Path(args.embedding_npz)
            if npz_path.exists():
                self.embedding_npz = npz_path
        elif os.environ.get('TOPIC_MODEL_EMBEDDING_NPZ'):
            npz_path = Path(os.environ['TOPIC_MODEL_EMBEDDING_NPZ'])
            if npz_path.exists():
                self.embedding_npz = npz_path

        # ========== 2. 确定输出目录 ===========
        # 约定：
        # - Step07 会按方法传入 --output_dir=07_topic_models/<METHOD>，此时不得再追加 /<METHOD>，否则会出现双层目录。
        # - 若未显式指定 --output_dir，则在 out_dir_base 下按 method 自动分目录。
        out_dir_base = os.environ.get("TOPIC_MODEL_OUTPUT", None) or PATHS.get("dir_06_results")
        if not out_dir_base:
            out_dir_base = str(Path(__file__).resolve().parent / "06_topic_model_results")
        
        # 从输入文件名提取方法类型（如A、B、C、AB、ABC、baseline）
        input_name = str(self.input_file)
        match = re.search(r"topic_modeling(?:_part\d+)?_([A-Za-z]+)\.csv", input_name, re.IGNORECASE)
        method = match.group(1).upper() if match else "DEFAULT"
        self.method = method  # 保存方法名供后续使用

        if args.output_dir:
            self.output_dir = Path(args.output_dir)
        else:
            self.output_dir = Path(out_dir_base) / method
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # ========== 2.5 加载自定义停用词（用于 CountVectorizer） ==========
        # 关键设计：A/B 停用词在这里用于 TF-IDF 关键词提取，而非文本删词
        # 这样可以保持 embedding 语义完整，同时过滤无意义的主题关键词
        self.custom_stopwords = load_custom_stopwords(base_dir, method)

        # ========== 3. 配置参数 ==========
        self.cfg = FrontierConfig(
            model_name=args.model_name or os.environ.get("TOPIC_MODEL_EMBEDDING", MODEL_CONFIG.get("model_name", "all-MiniLM-L6-v2")),
            min_cluster_size=args.min_cluster_size or int(os.environ.get("TOPIC_MODEL_MIN_CLUSTER_SIZE", MODEL_CONFIG.get("min_topic_size", 60))),
            ngram_range=tuple(MODEL_CONFIG.get("n_gram_range", (1, 1))),
        )

        # ========== 4. 生成审稿人友好型manifest ==========
        self.review_manifest = {
            "input_file": str(self.input_file),
            "output_dir": str(self.output_dir),
            "method": method,
            "embedding_model": self.cfg.model_name,
            "min_cluster_size": self.cfg.min_cluster_size,
            "ngram_range": self.cfg.ngram_range,
            "custom_stopwords_count": len(self.custom_stopwords),
            "run_time": str(datetime.now()),
            "code_version": "_engine_bertopic.py@v3.0",  # 重命名后版本
        }

        # reduce_outliers 配置（供上游报告直接读取）
        self.review_manifest["reduce_outliers_cfg"] = {
            "enabled_when_noise_gt": 0.10,  # >10% 噪声即启用
            "target_noise_ratio": 0.15,
            "stages": [
                {"strategy": "c-tf-idf", "threshold": 0.1},
                {"strategy": "embeddings"},
                {"strategy": "distributions"},
            ],
        }

        with open(self.output_dir / "review_manifest.json", "w", encoding="utf-8") as f:
            json.dump(self.review_manifest, f, ensure_ascii=False, indent=2)

        self.topic_model: Optional[BERTopic] = None

        self.df: Optional[pd.DataFrame] = None

        self.stats = {
            "total_docs": 0,
            "valid_topics": 0,
            "noise_ratio": 0.0,
        }
        # per-run summaries (min_cluster_size runs)
        self.run_summaries: list[dict] = []
        self.project_prefix = PROJECT_PREFIX

    def run(self):
        print("=" * 60)
        print(f"第六步：BERTopic 主题建模 - {get_project_name()}")
        print("=" * 60)

        # 显示当前输入文件及其去噪方法类型
        input_file_str = str(self.input_file)
        # 自动识别方法类型
        import re
        match = re.search(r'topic_modeling(?:_part\w+)?_([A-Za-z_]+)\\?.csv', input_file_str)
        method = match.group(1) if match else "未知/默认"
        print(f"当前建模输入文件: {input_file_str}")
        print(f"自动识别去噪方法类型: {method}")

        self.df = self._load_data()
        if self.df is None:
            return

        self._train_model()
        print("\n" + "=" * 60)
        print("第六步完成！（三组min_cluster_size结果已保存）")
        print(f"  文档数: {self.stats['total_docs']}")
        print(f"  输入文件: {self.input_file}")
        print(f"  输出目录: {self.output_dir}")
        print("=" * 60)

    def _load_data(self) -> Optional[pd.DataFrame]:
        if not self.input_file.exists():
            print(f"错误：未找到输入文件 {self.input_file}")
            print("请先运行第四步/第五步")
            return None

        df = pd.read_csv(str(self.input_file), dtype={"PMID": str})
        print(f"加载数据: {len(df)} 条")

        # 预处理
        for col in ["Title", "Abstract"]:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str)

        if "Year" in df.columns:
            df["Year"] = pd.to_numeric(df["Year"], errors="coerce").fillna(0).astype(int)
        else:
            df["Year"] = 0

        if "Citation_Count" in df.columns:
            df["Citation_Count"] = pd.to_numeric(df["Citation_Count"], errors="coerce").fillna(0).astype(int)
        else:
            df["Citation_Count"] = 0

        # 优先使用已有 text_for_model，否则现拼
        if "text_for_model" not in df.columns:
            df["text_for_model"] = (df["Title"].str.strip() + ". " + df["Abstract"].str.strip()).str.strip()
        else:
            df["text_for_model"] = df["text_for_model"].fillna("").astype(str)

        # Step06 会写入 text_for_embedding（原文完整文本）。
        # 关键：embedding 用原文，vectorizer/c-tf-idf 用 text_for_model（可能被删词）。
        if "text_for_embedding" in df.columns:
            df["text_for_embedding"] = df["text_for_embedding"].fillna("").astype(str)
        else:
            df["text_for_embedding"] = df["text_for_model"]

        df = df[df["text_for_model"].str.len() > 0].copy()
        self.stats["total_docs"] = len(df)
        return df

    def _compute_adaptive_mc_sizes(self, docs: list, embeddings: np.ndarray) -> list:
        import math
        from collections import Counter
        
        N = len(docs)
        sqrt_N = math.sqrt(N)
        ln_N = math.log(N) if N > 0 else 1
        alpha = 0.15
        beta = 2.0
        mc_base = alpha * sqrt_N + beta * ln_N
        all_words = []
        for doc in docs[:min(5000, N)]:
            all_words.extend(doc.lower().split())
        word_counts = Counter(all_words)
        total_words = sum(word_counts.values())
        if total_words > 0:
            H_vocab = 0.0
            for count in word_counts.values():
                p = count / total_words
                if p > 0:
                    H_vocab -= p * math.log2(p)
            max_entropy = math.log2(len(word_counts)) if len(word_counts) > 1 else 1
            H_vocab_norm = H_vocab / max_entropy if max_entropy > 0 else 0
        else:
            H_vocab_norm = 0.5
        if embeddings is not None and len(embeddings) > 100:
            # 关键：固定随机性，保证 mc_sizes 可复现（审稿友好）
            rng = np.random.default_rng(42)
            sample_idx = rng.choice(len(embeddings), min(500, len(embeddings)), replace=False)
            sample_emb = embeddings[sample_idx]
            norms = np.linalg.norm(sample_emb, axis=1, keepdims=True)
            norms[norms == 0] = 1
            sample_emb_norm = sample_emb / norms
            sim_matrix = np.dot(sample_emb_norm, sample_emb_norm.T)
            np.fill_diagonal(sim_matrix, 0)
            avg_sim = sim_matrix.sum() / (len(sample_idx) * (len(sample_idx) - 1))
            density_factor = 1.0 - avg_sim
        else:
            density_factor = 0.5
        gamma = 0.1
        delta = 0.2
        mc_adjusted = mc_base * (1 + gamma * H_vocab_norm + delta * density_factor)
        mc_large = int(np.clip(mc_adjusted * 1.3, 15, sqrt_N * 0.5))
        mc_medium = int(np.clip(mc_adjusted * 1.0, 15, sqrt_N * 0.4))
        mc_small = int(np.clip(mc_adjusted * 0.7, 15, sqrt_N * 0.3))
        mc_min = int(np.clip(mc_adjusted * 0.4, 15, sqrt_N * 0.2))
        mc_sizes = sorted(list(set([mc_large, mc_medium, mc_small, mc_min])), reverse=True)
        print(f"  [公式] mc_base = {alpha} * sqrt({N}) + {beta} * ln({N}) = {mc_base:.1f}")
        print(f"  [公式] H_vocab_norm = {H_vocab_norm:.3f} (词汇香农熵归一化)")
        print(f"  [公式] density_factor = {density_factor:.3f} (向量空间密度)")
        print(f"  [公式] mc_adjusted = {mc_adjusted:.1f}")
        return mc_sizes

    def _train_model(self) -> None:
        assert self.df is not None
        docs_for_vectorizer = self.df["text_for_model"].tolist()
        docs_for_embedding = self.df["text_for_embedding"].tolist()
        embeddings: Optional[np.ndarray] = None
        if self.embedding_npz is not None and self.embedding_npz.exists():
            try:
                import numpy as _np
                d = _np.load(self.embedding_npz, allow_pickle=True)
                emb_all = _np.asarray(d["embeddings"], dtype=_np.float32)
                pmids_all = _np.asarray(d["pmids"]).astype(int)
                pmid_series = self.df["PMID"] if "PMID" in self.df.columns else pd.Series([0] * len(self.df))
                pmid_col = pd.to_numeric(pmid_series, errors="coerce").fillna(0).astype(int).to_numpy()
                pos = {int(p): i for i, p in enumerate(pmids_all.tolist())}
                idx = []
                missing_count = 0
                for p in pmid_col.tolist():
                    j = pos.get(int(p))
                    if j is not None:
                        idx.append(j)
                    else:
                        idx.append(-1)
                        missing_count += 1
                missing_ratio = missing_count / len(idx) if idx else 1.0
                if missing_ratio > 0.05:
                    print(f"[警告] embedding_npz 与当前数据 PMID 不匹配率 {missing_ratio:.1%}，回退为实时嵌入")
                    embeddings = None
                elif missing_count > 0:
                    print(f"[INFO] 预计算向量覆盖率 {1-missing_ratio:.1%}，缺失 {missing_count} 条将实时计算")
                    from sentence_transformers import SentenceTransformer
                    import torch
                    model = SentenceTransformer(self.cfg.model_name)
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    model = model.to(device)
                    dim = emb_all.shape[1]
                    embeddings = _np.zeros((len(idx), dim), dtype=_np.float32)
                    missing_docs = []
                    missing_positions = []
                    for i, j in enumerate(idx):
                        if j >= 0:
                            embeddings[i] = emb_all[j]
                        else:
                            missing_docs.append(docs_for_embedding[i])
                            missing_positions.append(i)
                    if missing_docs:
                        missing_embs = model.encode(missing_docs, show_progress_bar=False, device=device)
                        for pos_i, emb in zip(missing_positions, missing_embs):
                            embeddings[pos_i] = emb
                    print(f"[INFO] 成功加载 C 方案清洁向量 + 补充计算，共 {len(embeddings)} 条")
                else:
                    embeddings = emb_all[_np.asarray(idx, dtype=int)]
                    print(f"[INFO] 成功加载 C 方案清洁向量，共 {len(embeddings)} 条")
            except Exception as e:
                print(f"[警告] 读取 embedding_npz 失败，回退为实时嵌入: {e}")
                embeddings = None
        if embeddings is None:
            print("\n[进度] 正在生成文本嵌入（支持进度条，多核大批量加速）...")
            from sentence_transformers import SentenceTransformer
            import torch
            embedding_model = SentenceTransformer(self.cfg.model_name)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            embedding_model = embedding_model.to(device)
            embeddings = embedding_model.encode(
                docs_for_embedding,
                show_progress_bar=True,
                batch_size=256,
                device=device
            )
        mc_sizes = self._compute_adaptive_mc_sizes(docs_for_vectorizer, embeddings)
        print(f"\n[算法] 自适应 min_cluster_size 计算完成")
        print(f"  基于 McInnes et al. (2017) sqrt(N) 上界")
        print(f"  基于 Shannon 熵调整文本多样性")
        print(f"  计算得到 mc_sizes: {mc_sizes}")
        N = len(docs_for_vectorizer)
        print(f"\n开始遍历 {len(mc_sizes)} 个自适应 min_cluster_size (文献数={N})")
        for min_cluster_size in mc_sizes:
            print(f"\n[RUN] min_cluster_size={min_cluster_size}")
            try:
                print("[进度] BERTopic 主题建模中...")
                from bertopic import BERTopic
                from hdbscan import HDBSCAN
                from sklearn.feature_extraction.text import CountVectorizer
                from umap import UMAP
                umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42)
                hdbscan_model = HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    metric="euclidean",
                    cluster_selection_method="eom",
                    prediction_data=True,
                    core_dist_n_jobs=-1
                )
                # 关键修复：使用自定义停用词（A/B方案）而非仅英文停用词
                # 这是 BERTopic 的正确用法：停用词用于 TF-IDF 关键词提取，不影响 embedding
                vectorizer_model = CountVectorizer(
                    stop_words=self.custom_stopwords if self.custom_stopwords else "english",
                    ngram_range=self.cfg.ngram_range
                )
                print(f"  [CountVectorizer] 停用词数量: {len(self.custom_stopwords) if self.custom_stopwords else '英文默认'}")
                topic_model = BERTopic(
                    embedding_model=None,
                    umap_model=umap_model,
                    hdbscan_model=hdbscan_model,
                    vectorizer_model=vectorizer_model,
                    nr_topics=None,
                    calculate_probabilities=False,
                    verbose=False,
                )
                with tqdm(total=1, desc="BERTopic.fit_transform", unit="step") as pbar:
                    topics, probs = topic_model.fit_transform(docs_for_vectorizer, embeddings=embeddings)
                    pbar.update(1)
                initial_noise_ratio = float(np.mean(np.array(topics) == -1))
                initial_noise_count = int(np.sum(np.array(topics) == -1))
                print(f"[STAT] HDBSCAN原始噪声: {initial_noise_count} 篇 ({initial_noise_ratio:.2%})")
                
                # ========== 顶刊级降噪策略 (参考 Grootendorst 2022, BERTopic 最佳实践) ==========
                # 策略：多阶段降噪，先用宽松阈值，再用严格阈值，最后用分布策略
                reduce_outliers_stats = {"initial": initial_noise_ratio, "reduced_by_bertopic": 0.0, "applied": False}
                NOISE_THRESHOLD_TO_ENABLE = 0.10  # >10% 噪声就启用降噪
                TARGET_NOISE_RATIO = 0.15  # 目标噪声比例
                
                if initial_noise_ratio > NOISE_THRESHOLD_TO_ENABLE:
                    print(f"[进度] 启用多阶段 reduce_outliers (目标噪声 ≤ {TARGET_NOISE_RATIO:.0%})...")
                    
                    current_topics = topics
                    current_noise = initial_noise_ratio
                    
                    # 阶段 1: c-tf-idf 策略 (threshold=0.1, 较宽松)
                    try:
                        if current_noise > TARGET_NOISE_RATIO:
                            topics_stage1 = topic_model.reduce_outliers(
                                docs_for_vectorizer, current_topics, strategy="c-tf-idf", threshold=0.1
                            )
                            noise_stage1 = float(np.mean(np.array(topics_stage1) == -1))
                            print(f"  [阶段1] c-tf-idf(0.1): {current_noise:.2%} → {noise_stage1:.2%}")
                            if noise_stage1 < current_noise:
                                current_topics = topics_stage1
                                current_noise = noise_stage1
                    except Exception as e:
                        print(f"  [阶段1] 跳过: {e}")
                    
                    # 阶段 2: embeddings 策略 (基于向量相似度重分配)
                    try:
                        if current_noise > TARGET_NOISE_RATIO:
                            topics_stage2 = topic_model.reduce_outliers(
                                docs_for_vectorizer, current_topics, strategy="embeddings", embeddings=embeddings
                            )
                            noise_stage2 = float(np.mean(np.array(topics_stage2) == -1))
                            print(f"  [阶段2] embeddings: {current_noise:.2%} → {noise_stage2:.2%}")
                            if noise_stage2 < current_noise:
                                current_topics = topics_stage2
                                current_noise = noise_stage2
                    except Exception as e:
                        print(f"  [阶段2] 跳过: {e}")
                    
                    # 阶段 3: distributions 策略 (最激进，基于概率分布)
                    try:
                        if current_noise > TARGET_NOISE_RATIO:
                            topics_stage3 = topic_model.reduce_outliers(
                                docs_for_vectorizer, current_topics, strategy="distributions"
                            )
                            noise_stage3 = float(np.mean(np.array(topics_stage3) == -1))
                            print(f"  [阶段3] distributions: {current_noise:.2%} → {noise_stage3:.2%}")
                            if noise_stage3 < current_noise:
                                current_topics = topics_stage3
                                current_noise = noise_stage3
                    except Exception as e:
                        print(f"  [阶段3] 跳过: {e}")
                    
                    # 应用最终结果
                    final_noise = current_noise
                    print(f"  [汇总] 噪声降低: {initial_noise_ratio:.2%} → {final_noise:.2%} (降低 {initial_noise_ratio - final_noise:.2%})")
                    topics = current_topics
                    reduce_outliers_stats["reduced_by_bertopic"] = initial_noise_ratio - final_noise
                    reduce_outliers_stats["applied"] = True
                    reduce_outliers_stats["stages_applied"] = ["c-tf-idf", "embeddings", "distributions"]
                    
                    # 更新主题模型
                    topic_model.update_topics(docs_for_vectorizer, topics=topics)
                else:
                    print(f"[INFO] 噪声已较低 ({initial_noise_ratio:.2%} <= {NOISE_THRESHOLD_TO_ENABLE:.0%})，跳过内置降噪")
                
                final_noise_ratio = float(np.mean(np.array(topics) == -1))
                final_noise_count = int(np.sum(np.array(topics) == -1))
                print(f"[STAT] ===== 噪声统计汇总 =====")
                print(f"  HDBSCAN原始噪声: {initial_noise_count} 篇 ({initial_noise_ratio:.2%})")
                print(f"  BERTopic内置降噪: -{reduce_outliers_stats['reduced_by_bertopic']:.2%} (applied={reduce_outliers_stats.get('applied',False)})")
                print(f"  最终噪声: {final_noise_count} 篇 ({final_noise_ratio:.2%})")
                print(f"  [对比] ABC文本降噪效果需与baseline对比查看")
                topic_info = topic_model.get_topic_info()
                valid_topics = int((topic_info["Topic"] != -1).sum())
                noise_ratio = float(np.mean(np.array(topics) == -1))
                print(f"[STAT] 最终有效主题: {valid_topics}, 最终噪声比例: {noise_ratio:.2%}")
                self.topic_model = topic_model
                doc_topic = self._build_doc_topic(topics, probs)
                frontier, weights, thresholds = self._compute_frontier_indicators(doc_topic)
                print(f"\n聚类质量指标：")
                print(f"  min_cluster_size={min_cluster_size}")
                print(f"  有效主题数={valid_topics}")
                print(f"  噪声比例={noise_ratio:.2%}")
                print(f"  文献总数={N}")
                self.stats["valid_topics"] = int(valid_topics)
                self.stats["noise_ratio"] = float(noise_ratio)
                run_summary = {
                    "min_cluster_size": int(min_cluster_size),
                    "initial_noise_ratio": float(initial_noise_ratio),
                    "final_noise_ratio": float(final_noise_ratio),
                    "reduced_by_bertopic": float(reduce_outliers_stats.get("reduced_by_bertopic", 0.0)),
                    "reduce_outliers_applied": bool(reduce_outliers_stats.get("applied", False)),
                    "reduce_outliers_cfg": self.review_manifest.get("reduce_outliers_cfg", {}),
                }
                self.run_summaries.append(run_summary)
                try:
                    with open(self.output_dir / "run_summaries.json", "w", encoding="utf-8") as _wf:
                        json.dump(self.run_summaries, _wf, ensure_ascii=False, indent=2)
                except Exception:
                    pass
                self._save_results_multi(topic_model, doc_topic, frontier, weights, thresholds, min_cluster_size)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                import traceback
                print(f"[ERROR] min_cluster_size={min_cluster_size} 失败: {e}")
                traceback.print_exc()
                continue
    def _save_comparison_report(self, results_summary: list[dict]) -> None:
        import os
        import matplotlib.pyplot as plt
        md_lines = [
            "# BERTopic min_cluster_size 对比报告\n",
            "| min_cluster_size | 有效主题數 | 噪声比例 |",
            "|------------------|------------|----------|",
        ]
        for r in results_summary:
            md_lines.append(f"| {r['min_cluster_size']} | {r['valid_topics']} | {r['noise_ratio']:.2%} |")
        sizes = [int(r["min_cluster_size"]) for r in results_summary]
        topics = [int(r["valid_topics"]) for r in results_summary]
        noises = [float(r["noise_ratio"]) for r in results_summary]
        fig, ax1 = plt.subplots()
        ax1.set_xlabel("min_cluster_size")
        ax1.set_ylabel("有效主题数")
        ax1.plot(sizes, topics, "o-", label="有效主题数")
        ax2 = ax1.twinx()
        ax2.set_ylabel("噪声比例")
        ax2.plot(sizes, noises, "s--", label="噪声比例")
        fig.tight_layout()
        img_path = os.path.join(self.output_dir, "bertopic_mc_comparison.png")
        plt.title("min_cluster_size 对比")
        plt.savefig(img_path, dpi=150)
        plt.close(fig)
        md_lines.append("\n![](bertopic_mc_comparison.png)")
        md_path = os.path.join(self.output_dir, "bertopic_mc_comparison.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
        print(f"[保存] {md_path}")
    def _save_results_multi(self, topic_model, doc_topic, frontier, weights, thresholds, min_cluster_size):
        assert topic_model is not None
        prefix = f"{self.project_prefix}_mc{int(min_cluster_size)}"
        prev_model = self.topic_model
        self.topic_model = topic_model
        topic_info = topic_model.get_topic_info().copy()
        topic_info["TopWords"] = topic_info["Topic"].apply(lambda t: self._get_topic_words(int(t)))

        # 生成可审稿、可区分的 Topic_Label（避免不同 Topic 生成同一“主题名”）
        try:
            topic_info["Topic_Label"] = self._build_unique_topic_labels(topic_info)
        except Exception:
            # 不阻断主流程
            pass

        topic_info_path = self.output_dir / f"{prefix}_topic_info.csv"
        topic_info.to_csv(topic_info_path, index=False, encoding="utf-8-sig")
        print(f"[保存] {topic_info_path}")
        doc_topic_path = self.output_dir / f"{prefix}_doc_topic_mapping.csv"
        doc_topic.to_csv(doc_topic_path, index=False, encoding="utf-8-sig")
        print(f"[保存] {doc_topic_path}")
        frontier_path = self.output_dir / f"{prefix}_frontier_indicators.csv"
        frontier.to_csv(frontier_path, index=False, encoding="utf-8-sig")
        print(f"[保存] {frontier_path}")
        weights_path = self.output_dir / f"{prefix}_critic_weights.csv"
        weights.to_csv(weights_path, index=False, encoding="utf-8-sig")
        print(f"[保存] {weights_path}")
        thresholds_path = self.output_dir / f"{prefix}_thresholds.csv"
        thresholds.to_csv(thresholds_path, index=False, encoding="utf-8-sig")
        print(f"[保存] {thresholds_path}")
        excel_file = self.output_dir / f"{prefix}_summary.xlsx"
        with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
            topic_info.to_excel(writer, sheet_name="TopicInfo", index=False)
            frontier.to_excel(writer, sheet_name="Frontier", index=False)
            weights.to_excel(writer, sheet_name="CRITIC_Weights", index=False)
            thresholds.to_excel(writer, sheet_name="Thresholds", index=False)
        print(f"[保存] {excel_file}")
        model_path = self.output_dir / f"{prefix}_bertopic_model"
        topic_model.save(model_path, save_embedding_model=False)
        print(f"[保存] {model_path}")
        self.topic_model = prev_model
        print(f"  结果已保存到: {self.output_dir} (min_cluster_size={min_cluster_size})")
    def _get_topic_words(self, topic_id: int, top_n: int = 10) -> str:
        assert self.topic_model is not None
        topic = self.topic_model.get_topic(topic_id)
        if not topic:
            return ""
        if not isinstance(topic, list):
            return ""
        pairs = cast(list[tuple[str, float]], topic)
        words = pairs[:top_n]
        return "; ".join([w for w, _ in words])

    def _tokenize_topwords(self, topwords: str) -> list[str]:
        import re
        s = str(topwords or "")
        s = s.replace(";", ",")
        raw = [w.strip() for w in s.split(",") if w.strip()]
        out: list[str] = []
        for w in raw:
            w = w.strip().lower().replace("_", " ")
            w = re.sub(r"\s+", " ", w)
            if len(w) < 3:
                continue
            out.append(w)
        return out

    def _make_label_from_words(self, words: list[str], k: int) -> str:
        # 选择前 k 个词作为标签；用分隔符提高可读性
        if not words:
            return "General topic"
        sel = words[: max(1, k)]
        # 标题化：保留缩写样式的同时做轻量美化
        pretty = []
        for w in sel:
            if w.isupper():
                pretty.append(w)
            else:
                pretty.append(w)
        label = " · ".join(pretty)
        return label

    def _build_unique_topic_labels(self, topic_info: pd.DataFrame) -> pd.Series:
        """为 topic_info 生成稳定、可区分的 Topic_Label。

        目标：同一 (method, mc) 内不同 Topic 不共享同一 label；
        策略：
        - 基于 TopWords 的前 k 个关键词生成 label
        - 若重复，则逐步增加 k，并为冲突组追加“差异关键词”
        - 仍无法消歧时，最后才追加 (Topic <id>)
        """
        if topic_info is None or topic_info.empty or "Topic" not in topic_info.columns:
            return pd.Series(dtype=str)

        # 预先分词
        words_by_tid: dict[int, list[str]] = {}
        for _, row in topic_info.iterrows():
            try:
                tid = int(row.get("Topic"))
            except Exception:
                continue
            if tid < 0:
                continue
            words_by_tid[tid] = self._tokenize_topwords(row.get("TopWords", ""))

        # 初始 label（k=3）
        k = 3
        labels: dict[int, str] = {tid: self._make_label_from_words(words, k) for tid, words in words_by_tid.items()}

        # 迭代消歧
        for _ in range(6):
            groups: dict[str, list[int]] = {}
            for tid, lab in labels.items():
                groups.setdefault(lab, []).append(tid)
            dup_groups = {lab: tids for lab, tids in groups.items() if len(tids) > 1}
            if not dup_groups:
                break

            # 对每个冲突组，尝试增加 k 或追加差异词
            k = min(10, k + 1)
            for lab, tids in dup_groups.items():
                # 先尝试统一增加 k
                for tid in tids:
                    labels[tid] = self._make_label_from_words(words_by_tid.get(tid, []), k)

            # 再检查仍冲突的组，用“差异词”强行区分
            groups2: dict[str, list[int]] = {}
            for tid, lab in labels.items():
                groups2.setdefault(lab, []).append(tid)
            dup2 = {lab: tids for lab, tids in groups2.items() if len(tids) > 1}
            if not dup2:
                break

            for lab, tids in dup2.items():
                # 计算组内词汇差异
                union = []
                for tid in tids:
                    union.extend(words_by_tid.get(tid, []))
                union_set = set(union)

                for tid in tids:
                    words = words_by_tid.get(tid, [])
                    extra = None
                    # 找一个更靠后的词作为差异词
                    for w in words[k:]:
                        if w in union_set:
                            extra = w
                            break
                    if extra:
                        labels[tid] = f"{labels[tid]} · {extra}"

        # 最终兜底：若仍重复，追加 Topic id
        groups_final: dict[str, list[int]] = {}
        for tid, lab in labels.items():
            groups_final.setdefault(lab, []).append(tid)
        for lab, tids in groups_final.items():
            if len(tids) <= 1:
                continue
            for tid in tids:
                labels[tid] = f"{labels[tid]} (Topic {tid})"

        # 对齐到原 DataFrame 顺序
        out = []
        for _, row in topic_info.iterrows():
            try:
                tid = int(row.get("Topic"))
            except Exception:
                out.append("")
                continue
            if tid < 0:
                out.append("Outliers")
            else:
                out.append(labels.get(tid, "General topic"))
        return pd.Series(out)
    def _build_doc_topic(self, topics, probs) -> pd.DataFrame:
        assert self.df is not None
        df = self.df.copy()
        df["Topic"] = topics
        try:
            df["Topic_Probability"] = np.max(probs, axis=1)
        except Exception:
            df["Topic_Probability"] = np.nan
        for col in ["PMID", "Title", "Year", "Journal", "Citation_Count"]:
            if col not in df.columns:
                df[col] = "" if col in ["PMID", "Title", "Journal"] else 0
        return df[["PMID", "Title", "Year", "Journal", "Citation_Count", "Topic", "Topic_Probability"]]
    def _compute_frontier_indicators(self, doc_topic: pd.DataFrame):
        print("\n计算研究前沿指标...")
        topic_ids = sorted([t for t in pd.unique(doc_topic["Topic"]) if t != -1])
        total_docs = int((doc_topic["Topic"] != -1).sum())
        if total_docs <= 0:
            print("警告：没有有效主题文档")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        valid_years = doc_topic.loc[doc_topic["Year"] > 0, "Year"]
        max_year = int(valid_years.max()) if not valid_years.empty else datetime.now().year
        recent_start = max_year - (self.cfg.recent_years - 1)
        high_cited_threshold = max(1, int(doc_topic["Citation_Count"].quantile(self.cfg.high_cited_quantile)))
        rows = []
        for tid in topic_ids:
            subset = doc_topic[doc_topic["Topic"] == tid]
            n = len(subset)
            if n == 0:
                continue
            strength = n / total_docs
            avg_year = float(subset.loc[subset["Year"] > 0, "Year"].mean()) if (subset["Year"] > 0).any() else 0.0
            recent_ratio = float((subset["Year"] >= recent_start).sum() / n)
            avg_citations = float(subset["Citation_Count"].sum() / n)
            high_cited_count = int((subset["Citation_Count"] >= high_cited_threshold).sum())
            rows.append({"Topic": tid, "TopWords": self._get_topic_words(tid), "Document_Count": n, "Strength": strength, "Novelty_AvgYear": avg_year, "Heat_RecentRatio": recent_ratio, "Avg_Citations": avg_citations, "HighCited_Count": high_cited_count})
        indicators = pd.DataFrame(rows)
        metric_cols = ["Strength", "Novelty_AvgYear", "Heat_RecentRatio", "Avg_Citations", "HighCited_Count"]
        metrics = indicators[metric_cols].astype(float)
        minv, maxv = metrics.min(axis=0), metrics.max(axis=0)
        denom = (maxv - minv).replace(0, np.nan)
        norm = ((metrics - minv) / denom).fillna(0.0)
        norm.columns = [c + "_Norm" for c in metric_cols]
        std = norm.std(axis=0, ddof=0)
        corr = norm.corr().fillna(0.0)
        conflict = (1.0 - corr).sum(axis=0)
        c_j = std * conflict
        weights = c_j / c_j.sum() if c_j.sum() > 0 else c_j * 0 + 1 / len(c_j)
        weights_df = pd.DataFrame({"Indicator": norm.columns, "Weight": weights.values})
        composite = (norm * weights.values).sum(axis=1)
        result = pd.concat([indicators, norm], axis=1)
        result["Composite_Index"] = composite
        comp_50 = float(result["Composite_Index"].quantile(0.50))
        comp_75 = float(result["Composite_Index"].quantile(0.75))
        nov_60 = float(result["Novelty_AvgYear_Norm"].quantile(0.60))
        nov_75 = float(result["Novelty_AvgYear_Norm"].quantile(0.75))
        heat_50 = float(result["Heat_RecentRatio_Norm"].quantile(0.50))
        heat_75 = float(result["Heat_RecentRatio_Norm"].quantile(0.75))
        growth_median = float(result["Heat_RecentRatio"].median())
        def classify(row):
            comp_very_high = row["Composite_Index"] >= comp_75
            comp_high = row["Composite_Index"] >= comp_50
            nov_very_high = row["Novelty_AvgYear_Norm"] >= nov_75
            nov_high = row["Novelty_AvgYear_Norm"] >= nov_60
            nov_low = row["Novelty_AvgYear_Norm"] < nov_60
            heat_high = row["Heat_RecentRatio_Norm"] >= heat_50
            heat_low = row["Heat_RecentRatio_Norm"] < heat_50
            has_growth = row["Heat_RecentRatio"] >= growth_median
            if heat_high and comp_high:
                return "热点"
            if nov_high and has_growth and not comp_very_high:
                return "新兴"
            if nov_very_high and not comp_high:
                return "潜在"
            if nov_high and heat_low and not comp_high:
                return "潜在"
            if nov_low and heat_low:
                return "衰退"
            return "一般"
        result["Frontier_Type"] = result.apply(classify, axis=1)
        result = result.sort_values("Composite_Index", ascending=False)
        type_counts = result["Frontier_Type"].value_counts()
        print(f"  前沿分类: 热点={type_counts.get('热点',0)}, 新兴={type_counts.get('新兴',0)}, 潜在={type_counts.get('潜在',0)}, 衰退={type_counts.get('衰退',0)}, 一般={type_counts.get('一般',0)}")
        print(f"  [参考] Small et al. (2014) 新兴主题识别; Chen (2006) CiteSpace 热点检测")
        thresholds_df = pd.DataFrame([
            {"Name": "Composite_50%", "Value": comp_50},
            {"Name": "Composite_75%", "Value": comp_75},
            {"Name": "Novelty_60%", "Value": nov_60},
            {"Name": "Novelty_75%", "Value": nov_75},
            {"Name": "Heat_50%", "Value": heat_50},
            {"Name": "Heat_75%", "Value": heat_75},
            {"Name": "Growth_Median", "Value": growth_median},
            {"Name": "HighCited_Threshold", "Value": high_cited_threshold},
            {"Name": "Recent_Start_Year", "Value": recent_start},
        ])
        print(f"  计算完成，共 {len(result)} 个主题")
        return result, weights_df, thresholds_df


if __name__ == "__main__":
    modeler = TopicModeler()
    modeler.run()
