#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step07_topic_model.py
Step 07：主题建模（对 baseline/A/B/C/AB/ABC 逐方法建模）

关键约束（为避免你提到的“时间太长”）：
- 默认如果 07_topic_models/<METHOD>/ 下已经存在 *_mc*_topic_info.csv，则视为已完成并跳过
- 使用 --force 才会重新跑建模

同时输出：
- 终端：每个方法的可用 mc 列表、topic 数、噪声比例（从 doc_topic_mapping 推断）
- 07_topic_models/<METHOD>/{PROJECT_PREFIX}_topic_model_manifest.json

输入：06_denoised_data/{PROJECT_PREFIX}_topic_modeling_{method}.csv
输出：07_topic_models/<METHOD>/*

说明：
- 建模引擎：_engine_bertopic.py（BERTopic 主题建模核心）

用法：
- python step07_topic_model.py
- python step07_topic_model.py --only ABC
- python step07_topic_model.py --only ABC --force
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd

try:
    from config import PROJECT_PREFIX, get_project_name
except ImportError:
    print("请确保 config.py 存在且配置正确")
    raise


ALL_METHODS = ["baseline", "A", "B", "C", "AB", "ABC"]
_mc_re = re.compile(r"_mc(\d+)_")


def discover_mcs(method_dir: Path) -> List[int]:
    mcs: List[int] = []
    for f in method_dir.glob(f"{PROJECT_PREFIX}_mc*_topic_info.csv"):
        m = _mc_re.search(f.name)
        if m:
            try:
                mcs.append(int(m.group(1)))
            except ValueError:
                pass
    return sorted(list(set(mcs)))


def summarize_mc(method_dir: Path, mc: int) -> Dict[str, Any]:
    info_file = method_dir / f"{PROJECT_PREFIX}_mc{mc}_topic_info.csv"
    map_file = method_dir / f"{PROJECT_PREFIX}_mc{mc}_doc_topic_mapping.csv"

    out: Dict[str, Any] = {"mc": mc, "files": {"topic_info": str(info_file), "doc_topic_mapping": str(map_file)}}

    if info_file.exists():
        df = pd.read_csv(info_file)
        if "Topic" in df.columns:
            out["num_topics"] = int((df["Topic"] >= 0).sum())
        out["topic_info_rows"] = int(len(df))

    if map_file.exists():
        dfm = pd.read_csv(map_file)
        out["num_docs"] = int(len(dfm))
        if "Topic" in dfm.columns:
            noise = int((dfm["Topic"] == -1).sum())
            out["noise_docs"] = noise
            out["noise_ratio"] = float(noise / max(1, len(dfm)))

    return out


def run_model_engine(base_dir: Path, input_file: Path, output_dir: Path, method: str) -> int:
    """
    调用主题建模引擎。
    
    关键修复：对于包含 C 方案的方法（C/ABC），传递预计算的清洁向量文件。
    这是 C 方案的核心价值——使用向量投影去噪后的 embedding，而不是重新 encode。
    """
    script = base_dir / "_engine_bertopic.py"
    if not script.exists():
        print("建模引擎不存在: _engine_bertopic.py")
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)

    # 构建命令行参数
    cmd = [
        sys.executable,
        str(script),
        "--input",
        str(input_file),
        "--output_dir",
        str(output_dir),
    ]
    
    # 关键修复：对于包含 C 的方法，传递预计算的清洁向量
    if "C" in method.upper():
        c_vector_file = base_dir / "05_stopwords" / "Experiment_C_Vector" / "output" / "c_final_clean_vectors.npz"
        if c_vector_file.exists():
            cmd.extend(["--embedding_npz", str(c_vector_file)])
            print(f"  [C方案] 使用预计算清洁向量: {c_vector_file.name}")
        else:
            print(f"  [警告] C方案向量文件不存在: {c_vector_file}，将回退为实时嵌入")

    # 不捕获输出，保留实时日志
    return subprocess.call(cmd, cwd=str(base_dir))


def main() -> int:
    parser = argparse.ArgumentParser(description="Step07: 主题建模")
    parser.add_argument("--only", choices=ALL_METHODS, help="只运行指定方法")
    parser.add_argument("--force", action="store_true", help="强制重跑，即使已有输出")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    methods = [args.only] if args.only else ALL_METHODS

    print("=" * 80)
    print(f"Step 07 主题建模 - {get_project_name()} ({PROJECT_PREFIX})")
    print("=" * 80)

    overall: Dict[str, Any] = {
        "project_prefix": PROJECT_PREFIX,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "methods": {},
    }

    any_ok = False

    for m in methods:
        input_file = base_dir / "06_denoised_data" / f"{PROJECT_PREFIX}_topic_modeling_{m}.csv"
        out_dir = base_dir / "07_topic_models" / m.upper()
        out_dir.mkdir(parents=True, exist_ok=True)

        if not input_file.exists():
            print(f"\n→ {m}: ✗ 输入不存在 {input_file.name}")
            overall["methods"][m] = {"status": "missing_input", "input": str(input_file)}
            continue

        mcs_existing = discover_mcs(out_dir)
        if mcs_existing and not args.force:
            print(f"\n→ {m}: 已检测到输出（mc={mcs_existing}），跳过（用 --force 重跑）")
        else:
            print(f"\n→ {m}: 开始建模...（这一步可能较慢）")
            code = run_model_engine(base_dir, input_file, out_dir, method=m)
            if code != 0:
                print(f"  ✗ 建模失败：returncode={code}")
                overall["methods"][m] = {"status": "failed", "returncode": code}
                continue

        # 汇总现有输出
        mcs = discover_mcs(out_dir)
        mc_summaries = [summarize_mc(out_dir, mc) for mc in mcs]

        print(f"  [OK] 输出 mc 列表: {mcs}")
        if mc_summaries:
            # 打印最前面的 1-2 个摘要，避免刷屏
            for s in mc_summaries[:2]:
                nr = s.get("noise_ratio")
                nr_str = f"{nr:.2%}" if isinstance(nr, float) else "-"
                print(f"    mc={s['mc']} topics={s.get('num_topics','-')} docs={s.get('num_docs','-')} noise={nr_str}")

        manifest = {
            "method": m,
            "input_file": str(input_file),
            "output_dir": str(out_dir),
            "mcs": mcs,
            "mc_summaries": mc_summaries,
            "status": "ok" if mcs else "no_outputs",
        }
        (out_dir / f"{PROJECT_PREFIX}_topic_model_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        overall["methods"][m] = manifest
        any_ok = any_ok or bool(mcs)

    # overall manifest
    (base_dir / "07_topic_models" / "topic_models_manifest.json").write_text(
        json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 80)
    print("Step 07 完成")
    print("输出目录: 07_topic_models/")
    print("manifest: 07_topic_models/topic_models_manifest.json")
    print("=" * 80)

    return 0 if any_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
