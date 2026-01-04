#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
step05_stopwords.py
Step 05：停用词/向量去噪产物生成（A/B/C 三方案）

目标（严格可复现）：
- 运行 A/B/C 方案各自的子步骤脚本
- 统计每个子步骤产物（文件、数量、示例）
- 同时输出：终端摘要 + 05_stopwords/stopwords_manifest.json

输入：
- 04_filtered_data/{PROJECT_PREFIX}_topic_modeling.csv（用于上游一致性；本步内部脚本会自行读取）

输出：
- 05_stopwords/Experiment_A_Statistical/output/combined_stopwords.txt 等
- 05_stopwords/Experiment_B_Semantic/output/stopwords_semantic_extended.txt 等
- 05_stopwords/Experiment_C_Vector/output/c_final_clean_vectors.npz
- 05_stopwords/stopwords_manifest.json

用法：
- python step05_stopwords.py
- python step05_stopwords.py --only A

参考：
- Zou et al. (2023). Representation Engineering. arXiv:2310.01405.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np

try:
    from config import PROJECT_PREFIX, get_project_name
except ImportError:
    print("请确保 config.py 存在且配置正确")
    raise


@dataclass
class StepRun:
    name: str
    script: str
    ok: bool
    returncode: int
    seconds: float
    stdout_tail: str


def _tail(text: str, max_chars: int = 2500) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def run_script(script_path: Path, cwd: Optional[Path] = None, timeout_sec: int = 1800) -> StepRun:
    start = time.time()
    if not script_path.exists():
        return StepRun(
            name=script_path.name,
            script=str(script_path),
            ok=False,
            returncode=2,
            seconds=0.0,
            stdout_tail=f"脚本不存在: {script_path}",
        )

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(cwd or script_path.parent),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec,
        )
        out = (result.stdout or "") + (result.stderr or "")
        return StepRun(
            name=script_path.stem,
            script=str(script_path),
            ok=result.returncode == 0,
            returncode=result.returncode,
            seconds=time.time() - start,
            stdout_tail=_tail(out),
        )
    except subprocess.TimeoutExpired:
        return StepRun(
            name=script_path.stem,
            script=str(script_path),
            ok=False,
            returncode=124,
            seconds=time.time() - start,
            stdout_tail=f"超时: {timeout_sec}s",
        )
    except Exception as e:
        return StepRun(
            name=script_path.stem,
            script=str(script_path),
            ok=False,
            returncode=1,
            seconds=time.time() - start,
            stdout_tail=f"异常: {e}",
        )


def read_wordlist(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    words: List[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            w = line.strip()
            if not w:
                continue
            words.append(w)
    sample = words[:50]
    return {
        "path": str(path),
        "exists": True,
        "count": len(words),
        "sample": sample,
    }


def read_npz_meta(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    meta: Dict[str, Any] = {"path": str(path), "exists": True, "bytes": path.stat().st_size}
    try:
        with np.load(path) as z:
            keys = list(z.keys())
            meta["keys"] = keys
            shapes = {}
            for k in keys[:10]:
                try:
                    shapes[k] = list(z[k].shape)
                except Exception:
                    shapes[k] = None
            meta["shapes"] = shapes
    except Exception as e:
        meta["load_error"] = str(e)
    return meta


def scheme_a(base_dir: Path, no_run: bool) -> Dict[str, Any]:
    algo_dir = base_dir / "05_stopwords" / "Experiment_A_Statistical" / "algorithms"
    output_dir = base_dir / "05_stopwords" / "Experiment_A_Statistical" / "output"
    steps = [
        ("SID智能权重", algo_dir / "sid_algorithm.py"),
        ("EVT极值理论", algo_dir / "evt_bert_topic.py"),
        ("Dynamic IDF", algo_dir / "dynamic_idf_algorithm.py"),
        ("合并产物", algo_dir / "evt_idf_merger.py"),
    ]

    print("\n" + "─" * 70)
    print("[Step 05.A] 统计去噪停用词生成（SID→EVT→Dynamic IDF→Merger）")
    print("─" * 70)

    runs: List[StepRun] = []
    if not no_run:
        for title, script in steps:
            print(f"  → {title}: {script.name} ... ", end="", flush=True)
            r = run_script(script, cwd=algo_dir)
            runs.append(r)
            print("✓" if r.ok else f"✗ (code={r.returncode})")
    else:
        print("  [no-run] 跳过脚本执行，仅汇总既有产物")

    artifacts = {
        "evt_stopwords": read_wordlist(output_dir / "evt_stopwords.txt"),
        "idf_stopwords": read_wordlist(output_dir / "idf_stopwords.txt"),
        "combined_stopwords": read_wordlist(output_dir / "combined_stopwords.txt"),
    }

    combined_cnt = artifacts.get("combined_stopwords", {}).get("count", 0)
    print(f"  产物: combined_stopwords.txt = {combined_cnt} 词")
    if artifacts.get("combined_stopwords", {}).get("sample"):
        preview = artifacts["combined_stopwords"]["sample"][:25]
        print("  示例(前25): " + ", ".join(preview))

    steps_ok = True if no_run else all(r.ok for r in runs[-2:])
    return {
        "scheme": "A",
        "steps": [r.__dict__ for r in runs],
        "artifacts": artifacts,
        "ok": steps_ok and combined_cnt > 0,
    }


def scheme_b(base_dir: Path, no_run: bool) -> Dict[str, Any]:
    b_dir = base_dir / "05_stopwords" / "Experiment_B_Semantic"
    output_dir = b_dir / "output"
    steps = [
        ("SPA结构化标记", b_dir / "01_spa_preprocessing.py"),
        ("CNI语境推理", b_dir / "02_cni_inference.py"),
        ("SEC语义扩展", b_dir / "03_sec_expansion.py"),
    ]

    print("\n" + "─" * 70)
    print("[Step 05.B] 语义扩展停用词生成（SPA→CNI→SEC）")
    print("─" * 70)

    runs: List[StepRun] = []
    if not no_run:
        for title, script in steps:
            print(f"  → {title}: {script.name} ... ", end="", flush=True)
            r = run_script(script, cwd=b_dir)
            runs.append(r)
            print("✓" if r.ok else f"✗ (code={r.returncode})")
    else:
        print("  [no-run] 跳过脚本执行，仅汇总既有产物")

    artifacts = {
        "stopwords_semantic_extended": read_wordlist(output_dir / "stopwords_semantic_extended.txt"),
        "final_semantic_stopwords": read_wordlist(output_dir / "final_semantic_stopwords.txt"),
    }

    cnt = artifacts.get("stopwords_semantic_extended", {}).get("count", 0)
    print(f"  产物: stopwords_semantic_extended.txt = {cnt} 词")
    if artifacts.get("stopwords_semantic_extended", {}).get("sample"):
        preview = artifacts["stopwords_semantic_extended"]["sample"][:25]
        print("  示例(前25): " + ", ".join(preview))

    steps_ok = True if no_run else bool(runs) and runs[-1].ok
    return {
        "scheme": "B",
        "steps": [r.__dict__ for r in runs],
        "artifacts": artifacts,
        "ok": steps_ok and cnt > 0,
    }


def scheme_c(base_dir: Path, no_run: bool) -> Dict[str, Any]:
    c_dir = base_dir / "05_stopwords" / "Experiment_C_Vector"
    output_dir = c_dir / "output"
    steps = [
        ("V-Fusion向量融合", c_dir / "01_vfusion_embedding.py"),
        ("RepE正交投影", c_dir / "02_repe_projection.py"),
        ("输出清洁向量", c_dir / "03_output_vectors.py"),
    ]

    print("\n" + "─" * 70)
    print("[Step 05.C] 向量投影去噪（V-Fusion→RepE→Output Vectors）")
    print("─" * 70)

    runs: List[StepRun] = []
    if not no_run:
        for title, script in steps:
            print(f"  → {title}: {script.name} ... ", end="", flush=True)
            r = run_script(script, cwd=c_dir, timeout_sec=3600)
            runs.append(r)
            print("✓" if r.ok else f"✗ (code={r.returncode})")
    else:
        print("  [no-run] 跳过脚本执行，仅汇总既有产物")

    vec_meta = read_npz_meta(output_dir / "c_final_clean_vectors.npz")
    if vec_meta.get("exists"):
        mb = vec_meta.get("bytes", 0) / (1024 * 1024)
        print(f"  产物: c_final_clean_vectors.npz = {mb:.1f} MB")
        if vec_meta.get("keys"):
            print("  keys: " + ", ".join(vec_meta["keys"][:8]))

    steps_ok = True if no_run else bool(runs) and runs[-1].ok
    return {
        "scheme": "C",
        "steps": [r.__dict__ for r in runs],
        "artifacts": {"c_final_clean_vectors": vec_meta},
        "ok": steps_ok and bool(vec_meta.get("exists")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Step05: 停用词/向量产物生成")
    # 不再支持按方案单独运行，始终执行 A/B/C 三个方案（可用 --no_run 跳过子脚本执行仅汇总产物）
    parser.add_argument("--no_run", action="store_true", help="不执行子脚本，仅汇总既有产物并生成 manifest")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent

    print("=" * 80)
    print(f"Step 05 停用词生成 - {get_project_name()} ({PROJECT_PREFIX})")
    print("=" * 80)

    # 始终运行全部方案 A, B, C（不再响应 --only）
    schemes = ["A", "B", "C"]
    manifest: Dict[str, Any] = {
        "project_prefix": PROJECT_PREFIX,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "schemes": {},
    }

    if "A" in schemes:
        manifest["schemes"]["A"] = scheme_a(base_dir, no_run=args.no_run)
    if "B" in schemes:
        manifest["schemes"]["B"] = scheme_b(base_dir, no_run=args.no_run)
    if "C" in schemes:
        manifest["schemes"]["C"] = scheme_c(base_dir, no_run=args.no_run)

    out_path = base_dir / "05_stopwords" / "stopwords_manifest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    ok_all = all(v.get("ok") for v in manifest["schemes"].values())

    print("\n" + "=" * 80)
    print("Step 05 完成汇总")
    for k, v in manifest["schemes"].items():
        status = "✓" if v.get("ok") else "✗"
        print(f"  方案{k}: {status}")
    print(f"  manifest: {out_path}")
    print("=" * 80)

    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
