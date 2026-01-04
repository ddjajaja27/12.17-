#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step06_denoise.py
Step 06：文本去噪（baseline/A/B/C/AB/ABC）

严格目标：
- 对 Step04 的 topic_modeling 数据生成 6 份一致格式的输入 CSV
- 记录：输入输出行数、token 数、删除 token 数、使用停用词数、保护词数
- 同时输出：终端摘要 + 06_denoised_data/denoise_manifest.json

说明（非常重要）：
- C 方法为“向量投影去噪”（在 Step05 生成向量产物），本步不做删词；
  因此 C/ABC 在文本层面与 baseline/AB 的差异可能很小，这属于设计结果，会在 report 中明确。

输入：04_filtered_data/{PROJECT_PREFIX}_topic_modeling.csv
输出：06_denoised_data/{PROJECT_PREFIX}_topic_modeling_{method}.csv

用法：
- python step06_denoise.py
- python step06_denoise.py --only ABC
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple

import pandas as pd

try:
    from config import PATHS, PROJECT_PREFIX, SEARCH_KEYWORD, get_project_name
except ImportError:
    print("请确保 config.py 存在且配置正确")
    raise


ALL_METHODS = ["baseline", "A", "B", "C", "AB", "ABC"]


@dataclass
class MethodStats:
    method: str
    rows_in: int
    rows_out: int
    total_tokens_before: int
    total_tokens_after: int
    tokens_removed: int
    stopwords_loaded: int
    protected_words: int
    stopwords_effective: int
    text_column: str
    output_file: str
    removed_top: List[Tuple[str, int]]
    stopwords_sample: List[str]


_token_re = re.compile(r"[A-Za-z][A-Za-z0-9_\-]+")


def _extract_tokens(text: str) -> List[str]:
    return _token_re.findall(text.lower())


def read_wordlist(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    out: Set[str] = set()
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            w = line.strip().lower()
            if w:
                out.add(w)
    return out


def load_stopwords_for_method(base_dir: Path, method: str) -> Set[str]:
    if method in ("baseline", "C"):
        return set()

    stopwords: Set[str] = set()

    a_file = base_dir / "05_stopwords" / "Experiment_A_Statistical" / "output" / "combined_stopwords.txt"
    b_file = base_dir / "05_stopwords" / "Experiment_B_Semantic" / "output" / "stopwords_semantic_extended.txt"

    if "A" in method and a_file.exists():
        stopwords |= read_wordlist(a_file)
    if "B" in method and b_file.exists():
        stopwords |= read_wordlist(b_file)

    # 通用停用词（可选）
    general = base_dir / "05_stopwords" / "common_data" / "general_stopwords.txt"
    if general.exists():
        stopwords |= read_wordlist(general)

    return stopwords


def load_whitelist(df: pd.DataFrame, base_dir: Path) -> Set[str]:
    whitelist: Set[str] = set()

    wl_file = base_dir / "05_stopwords" / "common_data" / "whitelist.txt"
    if wl_file.exists():
        whitelist |= read_wordlist(wl_file)

    # 自动保护：检索词、前缀、关键词字段
    for s in [SEARCH_KEYWORD, PROJECT_PREFIX, get_project_name()]:
        if s:
            whitelist |= set(_extract_tokens(str(s)))

    for col in ["Keywords", "MeSH_Terms", "Title"]:
        if col in df.columns:
            sample = " ".join(df[col].dropna().astype(str).head(200).tolist())
            whitelist |= set(_extract_tokens(sample))

    # 避免 whitelist 过大
    if len(whitelist) > 8000:
        whitelist = set(list(sorted(whitelist))[:8000])

    # 保护词最短长度
    whitelist = {w for w in whitelist if len(w) >= 3}
    return whitelist


def ensure_text_for_model(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    # old topic model engine expects text_for_model
    if "text_for_model" in df.columns:
        df["text_for_model"] = df["text_for_model"].fillna("").astype(str)
        return df, "text_for_model"

    title_s = df["Title"].fillna("").astype(str) if "Title" in df.columns else pd.Series([""] * len(df))
    abstract_s = df["Abstract"].fillna("").astype(str) if "Abstract" in df.columns else pd.Series([""] * len(df))
    df["text_for_model"] = (title_s.str.strip() + ". " + abstract_s.str.strip()).str.strip()

    df["text_for_model"] = df["text_for_model"].fillna("").astype(str)
    return df, "text_for_model"


def apply_stopwords_to_text(text: str, stopwords: Set[str], whitelist: Set[str]) -> Tuple[str, List[str]]:
    """计算停用词统计，但不改动文本（embedding 需要原文）。"""
    if not text:
        return text, []
    toks = _extract_tokens(text)
    removed: List[str] = []
    for t in toks:
        if t in whitelist:
            continue
        if t in stopwords:
            removed.append(t)
    # 返回原文 + 已移除词列表（仅用于统计）
    return text, removed


def denoise_method(df_in: pd.DataFrame, base_dir: Path, method: str, out_dir: Path) -> MethodStats:
    df = df_in.copy()
    df, text_col = ensure_text_for_model(df)

    # 保留一份“用于 embedding 的原文”，避免 A/B 类方法删词导致向量语义稀疏、噪声升高。
    # Step07 建模引擎会优先使用该列进行 SentenceTransformer.encode。
    if "text_for_embedding" not in df.columns:
        df["text_for_embedding"] = df[text_col].fillna("").astype(str)

    # 关键设计：embedding 必须用“原始完整文本”，否则删词会让向量更稀疏，HDBSCAN 更容易判为噪声。
    # 因此无论哪种方法，都保存一份原文给建模引擎使用。
    if "text_for_embedding" not in df.columns:
        df["text_for_embedding"] = df[text_col].fillna("").astype(str)

    rows_in = len(df)

    stopwords = load_stopwords_for_method(base_dir, method)
    whitelist = load_whitelist(df, base_dir)

    protected = stopwords & whitelist
    effective = stopwords - whitelist

    total_before = 0
    total_after = 0
    removed_counter: Dict[str, int] = {}

    if method not in ("baseline", "C"):
        new_texts: List[str] = []
        for t in df[text_col].fillna("").astype(str).tolist():
            toks_before = _extract_tokens(t)
            total_before += len(toks_before)
            new_t, removed = apply_stopwords_to_text(t, effective, whitelist)
            total_after += len(new_t.split()) if new_t else 0
            for r in removed:
                removed_counter[r] = removed_counter.get(r, 0) + 1
            new_texts.append(new_t)

        # 重要：不覆盖 text_for_embedding；text_for_model 用于 vectorizer/c-tf-idf
        df[text_col] = new_texts
    else:
        # baseline / C：不做删词，但统计 token 数
        for t in df[text_col].fillna("").astype(str).tolist():
            toks_before = _extract_tokens(t)
            total_before += len(toks_before)
            total_after += len(toks_before)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{PROJECT_PREFIX}_topic_modeling_{method}.csv"
    df.to_csv(out_file, index=False, encoding="utf-8-sig")

    removed_top = sorted(removed_counter.items(), key=lambda x: x[1], reverse=True)[:30]
    stopwords_sample = list(sorted(list(effective)))[:50]

    return MethodStats(
        method=method,
        rows_in=rows_in,
        rows_out=len(df),
        total_tokens_before=total_before,
        total_tokens_after=total_after,
        tokens_removed=max(0, total_before - total_after),
        stopwords_loaded=len(stopwords),
        protected_words=len(protected),
        stopwords_effective=len(effective),
        text_column=text_col,
        output_file=str(out_file),
        removed_top=removed_top,
        stopwords_sample=stopwords_sample,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Step06: 文本去噪")
    parser.add_argument("--only", choices=ALL_METHODS, help="只运行指定方法")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    input_file = Path(PATHS["file_04_topic"])
    if not input_file.exists():
        print(f"输入不存在: {input_file}")
        return 2

    df_in = pd.read_csv(input_file)

    methods = [args.only] if args.only else ALL_METHODS
    out_dir = base_dir / "06_denoised_data"

    print("=" * 80)
    print(f"Step 06 文本去噪 - {get_project_name()} ({PROJECT_PREFIX})")
    print(f"输入: {input_file.name}  行数={len(df_in)}")
    print("方法: " + ", ".join(methods))
    print("=" * 80)

    started = time.time()

    results: Dict[str, Any] = {
        "project_prefix": PROJECT_PREFIX,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_file": str(input_file),
        "methods": {},
    }

    ok_any = False
    for m in methods:
        print(f"\n→ 处理方法 {m} ...", end="", flush=True)
        stats = denoise_method(df_in, base_dir, m, out_dir)
        results["methods"][m] = asdict(stats)
        ok_any = True

        print("✓")
        print(
            f"  行数: {stats.rows_in} -> {stats.rows_out} | token: {stats.total_tokens_before} -> {stats.total_tokens_after} | 删除: {stats.tokens_removed}"
        )
        if m not in ("baseline", "C"):
            print(f"  停用词: loaded={stats.stopwords_loaded} protected={stats.protected_words} effective={stats.stopwords_effective}")
            if stats.stopwords_sample:
                print("  停用词示例(前25): " + ", ".join(stats.stopwords_sample[:25]))
            if stats.removed_top:
                print("  删除token Top10: " + ", ".join([f"{w}({c})" for w, c in stats.removed_top[:10]]))
        else:
            print("  说明: 本方法不做删词（baseline 对照 / C 向量投影在 Step05 体现）")

    results["seconds_total"] = round(time.time() - started, 3)

    manifest_path = out_dir / "denoise_manifest.json"
    manifest_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("Step 06 完成")
    print(f"输出目录: {out_dir}")
    print(f"manifest: {manifest_path}")
    print("=" * 80)

    return 0 if ok_any else 1


if __name__ == "__main__":
    raise SystemExit(main())
