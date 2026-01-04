#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step08_cv_select.py
Step 08：C_v 一致性评估 + 选择每个方法最优 min_cluster_size（mc）

你问“第八步有什么意义？”——意义在于：
- Step07 会产出多个 mc（min_cluster_size）版本；
- Step08 用一致性指标（C_v）在这些候选里选一个“语义一致性更好”的最佳 mc；
- Step09/10 只使用最佳 mc，保证后续可视化与报告一致（不会乱选第一个文件）。

输入：
- 07_topic_models/<METHOD>/ROJECT_PREFIX}_mc*_topic_info{P.csv
- 06_denoised_data/{PROJECT_PREFIX}_topic_modeling_<method>.csv（用于 coherence 的 texts）

输出：
- 08_model_selection/cv_select_best_mc_report.txt
- 08_model_selection/best_mc_by_method.json

用法：
- python step08_cv_select.py
- python step08_cv_select.py --only ABC
- python step08_cv_select.py --max_docs 8000

参考：
- Röder et al. (2015). Exploring the space of topic coherence measures. WSDM.
- Newman et al. (2010). Automatic evaluation of topic coherence. NAACL.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel

try:
    from config import PROJECT_PREFIX, get_project_name
except ImportError:
    print("请确保 config.py 存在且配置正确")
    raise


ALL_METHODS = ["baseline", "VPD", "A", "B", "C", "AB", "ABC"]
_mc_re = re.compile(r"_mc(\d+)_")


def _as_float_noise(v: Any) -> Optional[float]:
    return float(v) if isinstance(v, float) else None


def _dominates(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """Pareto dominance for objectives: maximize C_v, minimize noise.

    a dominates b if:
    - cv_a >= cv_b and noise_a <= noise_b
    - and at least one inequality is strict
    """
    cv_a = a.get("cv")
    cv_b = b.get("cv")
    if not (isinstance(cv_a, float) and isinstance(cv_b, float)):
        return False
    na = _as_float_noise(a.get("noise_ratio"))
    nb = _as_float_noise(b.get("noise_ratio"))
    if na is None or nb is None:
        return False
    return (cv_a >= cv_b and na <= nb) and (cv_a > cv_b or na < nb)


def pareto_front(cands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return non-dominated candidates (noise must be present)."""
    filtered = [c for c in cands if isinstance(c.get("cv"), float) and isinstance(c.get("noise_ratio"), float)]
    front: List[Dict[str, Any]] = []
    for i, c in enumerate(filtered):
        dominated = False
        for j, other in enumerate(filtered):
            if i == j:
                continue
            if _dominates(other, c):
                dominated = True
                break
        if not dominated:
            front.append(c)

    # Stable ordering for reporting: C_v desc, noise asc, mc asc
    return sorted(front, key=lambda x: (-float(x["cv"]), float(x["noise_ratio"]), int(x["mc"])))


def _noise_ratio_from_doc_map(method_dir: Path, mc: int) -> Optional[float]:
    """从 Step07 的 doc_topic_mapping 计算噪音比例（Topic=-1 的占比）。"""
    p = method_dir / f"{PROJECT_PREFIX}_mc{mc}_doc_topic_mapping.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        if "Topic" not in df.columns:
            return None
        noise = int((df["Topic"] == -1).sum())
        return float(noise / max(1, len(df)))
    except Exception:
        return None


def parse_top_words(df: pd.DataFrame, topn: int = 10) -> List[List[str]]:
    col = None
    for c in ["TopWords", "Representation", "Words"]:
        if c in df.columns:
            col = c
            break
    if col is None:
        return []

    topics: List[List[str]] = []
    for v in df[col].dropna().tolist():
        if not isinstance(v, str):
            continue

        # 兼容：
        # - "['a', 'b', ...]" (Representation)
        # - "a, b, c"
        # - "a; b; c" (BERTopic 的 TopWords 导出常见格式)
        s = v.strip()
        s = s.strip().strip("[]")
        s = s.replace("\"", "").replace("'", "")

        if ";" in s and "," not in s:
            raw_parts = s.split(";")
        else:
            raw_parts = s.split(",")

        parts = [p.strip() for p in raw_parts if p.strip()]
        if parts:
            topics.append(parts[:topn])

    return topics


def load_texts_for_method(base_dir: Path, method: str, max_docs: int, seed: int) -> List[List[str]]:
    csv_path = base_dir / "06_denoised_data" / f"{PROJECT_PREFIX}_topic_modeling_{method}.csv"
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path)
    if "text_for_model" not in df.columns:
        # 兼容：拼接
        title_s = df["Title"].fillna("").astype(str) if "Title" in df.columns else pd.Series([""] * len(df))
        abstract_s = df["Abstract"].fillna("").astype(str) if "Abstract" in df.columns else pd.Series([""] * len(df))
        df["text_for_model"] = (title_s.str.strip() + ". " + abstract_s.str.strip()).str.strip()

    texts = df["text_for_model"].fillna("").astype(str).tolist()

    # 采样加速
    if max_docs > 0 and len(texts) > max_docs:
        rnd = random.Random(seed)
        idx = list(range(len(texts)))
        rnd.shuffle(idx)
        idx = idx[:max_docs]
        texts = [texts[i] for i in idx]

    # tokenize: Step06 已用空格拼接 token；这里再 split 足够
    tokenized: List[List[str]] = []
    for t in texts:
        toks = [w for w in t.split() if w]
        if toks:
            tokenized.append(toks)
    return tokenized


def eval_method(
    base_dir: Path,
    method: str,
    max_docs: int,
    seed: int,
    noise_ref: Optional[float],
    noise_ref_label: str,
) -> Dict[str, Any]:
    method_dir = base_dir / "07_topic_models" / method.upper()
    if not method_dir.exists():
        return {"method": method, "status": "missing_dir"}

    topic_info_files = sorted(method_dir.glob(f"{PROJECT_PREFIX}_mc*_topic_info.csv"))
    if not topic_info_files:
        return {"method": method, "status": "no_topic_info"}

    texts = load_texts_for_method(base_dir, method, max_docs=max_docs, seed=seed)
    if not texts:
        return {"method": method, "status": "missing_texts"}

    # 内存优化：只在使用时构建 Dictionary
    dictionary = Dictionary(texts)
    print(f"  [内存] {method} 方法：{len(texts)} 篇文档，Dictionary={len(dictionary)}")

    scores: List[Dict[str, Any]] = []
    for f in topic_info_files:
        m = _mc_re.search(f.name)
        if not m:
            continue
        mc = int(m.group(1))
        df = pd.read_csv(f)
        topics = parse_top_words(df, topn=10)
        if not topics:
            continue

        noise_ratio = _noise_ratio_from_doc_map(method_dir, mc)

        try:
            # Windows 死机修复：必须使用 processes=1（否则 gensim 多进程会导致死机）
            cm = CoherenceModel(
                topics=topics, 
                texts=texts, 
                dictionary=dictionary, 
                coherence="c_v",
                processes=1  # 关键：Windows 上必须单进程，否则死机
            )
            cv = float(cm.get_coherence())
        except Exception as e:
            print(f"    [警告] mc={mc} C_v 计算失败：{str(e)[:100]}")
            scores.append({"mc": mc, "cv": None, "error": str(e), "file": f.name, "noise_ratio": noise_ratio})
            continue

        scores.append({"mc": mc, "cv": cv, "file": f.name, "num_topics": len(topics), "noise_ratio": noise_ratio})

    valid = [s for s in scores if isinstance(s.get("cv"), float)]
    if not valid:
        return {"method": method, "status": "no_valid_scores", "scores": scores}

    def _noise_key(item: Dict[str, Any]) -> float:
        nr = item.get("noise_ratio")
        return float(nr) if isinstance(nr, float) else 1.0

    scores_by_cv = sorted(valid, key=lambda x: (-float(x["cv"]), _noise_key(x), int(x["mc"])))
    scores_by_noise_cv = sorted(valid, key=lambda x: (_noise_key(x), -float(x["cv"]), int(x["mc"])))
    evaluated_mcs = sorted({int(s["mc"]) for s in valid if isinstance(s.get("mc"), int)})

    # 噪音约束（层级递减）：方法越多，噪音阈值越低。
    # A/B/C <= baseline；AB <= min(A,B,baseline)；ABC <= min(AB,C,A,B,baseline)
    ref = noise_ref if isinstance(noise_ref, float) else None

    selection_details: Dict[str, Any] = {
        "noise_ref": ref,
        "noise_ref_label": noise_ref_label,
        "feasible_mcs": [],
        "pareto_mcs": [],
        "rule": "",
    }

    # 标准化多目标规则：
    # 1) 若存在噪音阈值 ref：先约束 noise<=ref 得到可行集 F
    # 2) 在 F 上计算 Pareto 前沿（maximize C_v, minimize noise），从前沿选 C_v 最大者（平手选 noise 更小）
    # 3) 若 F 为空：在全体上计算 Pareto 前沿，选 noise 最小者（平手选 C_v 更大）
    # 4) 若 noise 全缺失：退化为按 C_v 最大选择

    if method != "baseline" and ref is not None:
        feasible = [s for s in valid if isinstance(s.get("noise_ratio"), float) and float(s["noise_ratio"]) <= ref + 1e-12]
        selection_details["feasible_mcs"] = sorted([int(s["mc"]) for s in feasible if isinstance(s.get("mc"), int)])
        if feasible:
            front = pareto_front(feasible)
            selection_details["pareto_mcs"] = [int(s["mc"]) for s in front if isinstance(s.get("mc"), int)]
            if front:
                best = sorted(front, key=lambda x: (-float(x["cv"]), float(x["noise_ratio"]), int(x["mc"])))[0]
                selection_note = f"constraint+pareto: noise<={noise_ref_label}({ref:.2%}); pick max C_v on Pareto front"
                selection_details["rule"] = "feasible(F): pareto(F) -> max C_v (tie: min noise)"
            else:
                # 可行集存在，但噪音缺失导致无法算 Pareto：退化
                best = max(feasible, key=lambda x: float(x["cv"]))
                selection_note = f"constraint: noise<={noise_ref_label}({ref:.2%}); pick max C_v (noise_missing_for_pareto)"
                selection_details["rule"] = "feasible(F): max C_v (noise missing)"
        else:
            # 无可行解：全体 Pareto，优先最小噪音
            front = pareto_front(valid)
            selection_details["pareto_mcs"] = [int(s["mc"]) for s in front if isinstance(s.get("mc"), int)]
            if front:
                best = sorted(front, key=lambda x: (float(x["noise_ratio"]), -float(x["cv"]), int(x["mc"])))[0]
                selection_note = f"fallback+pareto: no_mc_beats_noise_ref({noise_ref_label}={ref:.2%}); pick min noise on Pareto front"
                selection_details["rule"] = "infeasible: pareto(All) -> min noise (tie: max C_v)"
            else:
                best = scores_by_cv[0]
                selection_note = f"fallback: no_noise_available; pick max C_v"
                selection_details["rule"] = "infeasible: max C_v (noise missing)"
    else:
        # baseline：无噪音约束，仍可用 Pareto（但通常按 C_v 即可）
        front = pareto_front(valid)
        selection_details["pareto_mcs"] = [int(s["mc"]) for s in front if isinstance(s.get("mc"), int)]
        if front:
            best = sorted(front, key=lambda x: (-float(x["cv"]), float(x["noise_ratio"]), int(x["mc"])))[0]
            selection_note = "pareto: pick max C_v on Pareto front (tie: min noise)"
            selection_details["rule"] = "baseline: pareto(All) -> max C_v (tie: min noise)"
        else:
            best = scores_by_cv[0]
            selection_note = "cv_only"
            selection_details["rule"] = "baseline: max C_v"

    return {
        "method": method,
        "status": "ok",
        "max_docs": max_docs,
        "seed": seed,
        "best": best,
        "selection_note": selection_note,
        "noise_ref": noise_ref,
        "noise_ref_label": noise_ref_label,
        "selection_details": selection_details,
        "evaluated_mcs": evaluated_mcs,
        "scores_by_cv": scores_by_cv,
        "scores_by_noise_cv": scores_by_noise_cv,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Step08: C_v 最优 mc 选择")
    parser.add_argument(
        "--base_dir",
        default=None,
        help="工作目录（包含 06_denoised_data/07_topic_models/08_model_selection）。默认使用脚本所在目录。",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="输出目录（默认写入 <base_dir>/08_model_selection）。",
    )
    parser.add_argument("--only", choices=ALL_METHODS, help="只跑指定方法")
    parser.add_argument("--max_docs", type=int, default=3000, help="最多用于一致性计算的文档数（默认3000，0=全量，防止内存爆炸）")
    parser.add_argument("--seed", type=int, default=20251220, help="采样随机种子")
    args = parser.parse_args()

    def _resolve_base_dir() -> Path:
        if args.base_dir:
            return Path(args.base_dir).resolve()
        ptr = Path(__file__).resolve().parent / "reproducible_pipeline" / "MAIN_WORKDIR.txt"
        if ptr.exists():
            try:
                p = ptr.read_text(encoding="utf-8", errors="replace").strip().strip('"')
                if p:
                    cand = Path(p).expanduser()
                    if cand.exists():
                        return cand.resolve()
            except Exception:
                pass
        return Path(__file__).resolve().parent

    base_dir = _resolve_base_dir()
    methods = [args.only] if args.only else ALL_METHODS

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (base_dir / "08_model_selection")

    def run_batch(base_dir: Path, methods: List[str], max_docs: int, seed: int, out_dir: Path) -> int:
        out_dir.mkdir(parents=True, exist_ok=True)
        all_results = {"project_prefix": PROJECT_PREFIX, "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "max_docs": max_docs, "seed": seed, "methods": {}}
        best_map = {}
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append(f"C_v 最优 mc 选择报告 - {get_project_name()} ({PROJECT_PREFIX})")
        report_lines.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"max_docs={max_docs} seed={seed}")
        report_lines.append("参考：Röder et al. (2015) WSDM; Newman et al. (2010) NAACL")
        report_lines.append("".rstrip())
        report_lines.append("【一、问题定义（超参数选择）】")
        report_lines.append("- 目标：对每个去噪方法（baseline/A/B/C/AB/ABC），在 Step07 产出的候选 min_cluster_size (mc) 中选择一个最优值，用于 Step09/Step10 的统一可视化与报告。")
        report_lines.append("- 评价指标：主题一致性 C_v（越大越好）+ 噪音比例 noise（Topic=-1 文档占比，越小越好）。")
        report_lines.append("".rstrip())
        report_lines.append("【二、标准化多目标选择准则（审稿友好版）】")
        report_lines.append("对每个候选 mc_i，定义：")
        report_lines.append("- c_i = C_v(mc_i)  （maximize）")
        report_lines.append("- n_i = noise(mc_i)（minimize）")
        report_lines.append("噪音约束：给定参考阈值 r（来自父方法的最小噪音；baseline 无此约束），可行集：")
        report_lines.append("- F(r) = { i | n_i <= r }")
        report_lines.append("Pareto（帕累托）前沿定义（双目标）：候选 a 支配 b 当且仅当：")
        report_lines.append("- c_a >= c_b 且 n_a <= n_b，并且至少一项严格更优")
        report_lines.append("- Pareto(F) = F 中所有不被其它候选支配的点")
        report_lines.append("最终决策规则：")
        report_lines.append("- 若 F(r) 非空：从 Pareto(F) 中选 c 最大者；若并列，选 n 更小者（再并列选 mc 更小者，保证确定性）。")
        report_lines.append("- 若 F(r) 为空：说明无候选能满足噪音阈值；在全体候选上计算 Pareto(All)，先选 n 最小者；若并列选 c 更大者（再并列选 mc 更小者）。")
        report_lines.append("说明：这是一种常见的“硬约束 + 多目标折中”策略，避免把 C_v 提升建立在明显更差的噪音上，同时在不可行时给出最保守、可解释的退化规则。")
        report_lines.append("".rstrip())
        report_lines.append("【三、候选 mc 从何而来（Step07 自适应生成的内在原因）】")
        report_lines.append("Step08 并不‘固定只评估 22/39/56’，它只评估 Step07 实际写到磁盘的 *_mc*_topic_info.csv。")
        report_lines.append("Step07 的候选 mc 由 _engine_bertopic.py 自适应计算得到，核心形式为：")
        report_lines.append("- mc_base = α·sqrt(N) + β·ln(N)")
        report_lines.append("- mc_adjusted = mc_base · (1 + γ·H_vocab_norm + δ·density_factor)")
        report_lines.append("其中：N 为文档数；H_vocab_norm 为词汇香农熵归一化（反映文本多样性）；density_factor 为向量空间密度的反向量（越稠密越小）。")
        report_lines.append("候选集由若干缩放与裁剪得到（并去重排序）：")
        report_lines.append("- mc ∈ { 1.3·mc_adjusted, 1.0·mc_adjusted, 0.7·mc_adjusted, 0.4·mc_adjusted } 经过 clip 到 [15, 0.5·sqrt(N)] 等上界")
        report_lines.append("为何你经常看到‘重复的候选值’：如果不同去噪方法的 N 与文本统计特征接近，代入同一公式会得到相近的 mc_adjusted，离散化（取整+clip）后自然落在同一组整数上（例如 22/39/56/73）。")
        report_lines.append("可复现性：向量密度估计需要抽样；为审稿可复现，我们固定了抽样随机种子，因此同一数据下候选 mc 列表稳定。")
        report_lines.append("引用（背景依据）：McInnes et al., 2017 (HDBSCAN/JOSS, sqrt(N) 经验上界); Campello et al., 2013 (HDBSCAN 密度聚类思想); Shannon (信息熵用于多样性刻画)。")
        report_lines.append("=" * 80)

        any_ok = False

        # 先评估 baseline，拿到 baseline 的“参考噪音”
        baseline_noise_ref: Optional[float] = None
        noise_best: Dict[str, float] = {}
        if "baseline" in methods:
            report_lines.append("\n→ 评估 baseline (用于噪音参考) ...")
            r0 = eval_method(base_dir, "baseline", max_docs=max_docs, seed=seed, noise_ref=None, noise_ref_label="none")
            all_results["methods"]["baseline"] = r0
            if r0.get("status") == "ok":
                b0 = r0["best"]
                baseline_noise_ref = b0.get("noise_ratio") if isinstance(b0.get("noise_ratio"), float) else None
                best_map["baseline"] = {"mc": b0["mc"], "cv": b0["cv"], "file": b0.get("file"), "noise_ratio": baseline_noise_ref}
                if isinstance(baseline_noise_ref, float):
                    noise_best["baseline"] = float(baseline_noise_ref)
                any_ok = True
                nr_str = f"{baseline_noise_ref:.2%}" if isinstance(baseline_noise_ref, float) else "-"
                report_lines.append(f"[baseline] best_mc={b0['mc']} C_v={b0['cv']:.4f} noise={nr_str} (file={b0.get('file')})")
            else:
                report_lines.append(f"[baseline] 失败：{r0.get('status')}")

        # 评估顺序必须符合“方法层级”，这样父方法的噪音可作为子方法阈值。
        ordered = [m for m in ["baseline", "VPD", "A", "B", "C", "AB", "ABC"] if m in methods]
        parents: Dict[str, List[str]] = {
            "VPD": ["baseline"],
            "A": ["baseline"],
            "B": ["baseline"],
            "C": ["baseline"],
            "AB": ["A", "B", "baseline"],
            "ABC": ["AB", "C", "A", "B", "baseline"],
        }

        for m in ordered:
            if m == "baseline":
                continue

            cand_refs = [noise_best[p] for p in parents.get(m, []) if p in noise_best]
            if not cand_refs and isinstance(baseline_noise_ref, float):
                cand_refs = [float(baseline_noise_ref)]

            noise_ref = min(cand_refs) if cand_refs else None
            label = "min(" + ",".join([p for p in parents.get(m, []) if p in noise_best] or (["baseline"] if isinstance(baseline_noise_ref, float) else [])) + ")" if noise_ref is not None else "none"

            report_lines.append(f"\n→ 评估 {m} ...")
            r = eval_method(base_dir, m, max_docs=max_docs, seed=seed, noise_ref=noise_ref, noise_ref_label=label)
            all_results["methods"][m] = r
            if r.get("status") != "ok":
                report_lines.append(f"[{m}] 失败：{r.get('status')}")
                continue
            best = r["best"]
            best_map[m] = {"mc": best["mc"], "cv": best["cv"], "file": best.get("file"), "noise_ratio": best.get("noise_ratio"), "selection_note": r.get("selection_note")}
            if isinstance(best.get("noise_ratio"), float):
                noise_best[m] = float(best["noise_ratio"])
            any_ok = True
            nr = best.get("noise_ratio")
            nr_str = f"{nr:.2%}" if isinstance(nr, float) else "-"
            report_lines.append(f"[{m}] best_mc={best['mc']} C_v={best['cv']:.4f} noise={nr_str} ({r.get('selection_note')}) (file={best.get('file')})")
            if r.get("evaluated_mcs"):
                report_lines.append(f"    evaluated_mcs={r['evaluated_mcs']} (n={len(r['evaluated_mcs'])})")

            sel = r.get("selection_details") or {}
            if isinstance(sel, dict):
                if sel.get("noise_ref") is not None:
                    try:
                        report_lines.append(f"    noise_ref={float(sel['noise_ref']):.2%}  label={sel.get('noise_ref_label','-')}")
                    except Exception:
                        report_lines.append(f"    noise_ref={sel.get('noise_ref')}  label={sel.get('noise_ref_label','-')}")
                if sel.get("feasible_mcs"):
                    report_lines.append(f"    feasible_mcs={sel.get('feasible_mcs')}")
                if sel.get("pareto_mcs"):
                    report_lines.append(f"    pareto_mcs={sel.get('pareto_mcs')}")
                if sel.get("rule"):
                    report_lines.append(f"    decision_rule={sel.get('rule')}")

            # 展示与“实际选择准则”一致的 Top 候选：
            # - 正常情况：按 C_v 排序
            # - fallback 情况：按 (noise asc, C_v desc) 排序
            note = str(r.get("selection_note") or "")
            if note.startswith("fallback:"):
                cand = r.get("scores_by_noise_cv") or []
                report_lines.append("    候选排序：noise ↑，其次 C_v ↓")
            else:
                cand = r.get("scores_by_cv") or []
                report_lines.append("    候选排序：C_v ↓")

            topk = cand[:3]
            printed_mcs = set()
            for s in topk:
                nr2 = s.get("noise_ratio")
                nr2_str = f"{nr2:.2%}" if isinstance(nr2, float) else "-"
                report_lines.append(
                    f"    mc={s['mc']}  C_v={s['cv']:.4f}  noise={nr2_str}  topics={s.get('num_topics','-')}  ({s.get('file')})"
                )
                printed_mcs.add(int(s["mc"]))

            # 如果被选中的 mc 没出现在 Top3（常见于：按噪音优先的 fallback 场景），单独补一行避免误解。
            if isinstance(best.get("mc"), int) and int(best["mc"]) not in printed_mcs:
                nr2 = best.get("noise_ratio")
                nr2_str = f"{nr2:.2%}" if isinstance(nr2, float) else "-"
                report_lines.append(
                    f"    [selected] mc={best['mc']}  C_v={best['cv']:.4f}  noise={nr2_str}  topics={best.get('num_topics','-')}  ({best.get('file')})"
                )

        # 支持分批运行（--only）：尽量合并已有 best_mc_by_method.json，避免覆盖之前方法结果。
        best_path = out_dir / "best_mc_by_method.json"
        if best_path.exists():
            try:
                prev = json.loads(best_path.read_text(encoding="utf-8"))
                if isinstance(prev, dict):
                    prev.update(best_map)
                    best_map = prev
            except Exception:
                pass

        best_path.write_text(json.dumps(best_map, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "cv_select_best_mc_report.txt").write_text("\n".join(report_lines), encoding="utf-8")
        (out_dir / "cv_scores_full.json").write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")

        return 0 if any_ok else 1

    code = run_batch(base_dir, methods, args.max_docs, args.seed, out_dir)
    if code == 0:
        print("Step 08 完成")
        print(f"best_mc: {out_dir / 'best_mc_by_method.json'}")
        print(f"report:  {out_dir / 'cv_select_best_mc_report.txt'}")
    else:
        print("Step 08 未找到有效结果，请检查日志与输入文件")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
