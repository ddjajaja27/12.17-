#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step10_report.py
Step 10：生成“严格可复现”的研究报告（按方法分别输出）

你提的关键要求，本脚本实现：
- 每个方法（baseline/A/B/C/AB/ABC）报告内容不同（因为 stopwords/删词/模型结果不同）
- 报告必须体现 Step05/06/07：做了什么、跑了哪些子步骤、生成了多少、输出文件在哪里、示例有哪些
- 同时把关键统计在终端也打印一遍（不会只写在 report 里）
- Step08 的 best mc 一致性选择，会被报告引用，并用于定位 Step07 的“最终采用版本”

输入：
- 04_filtered_data/*_filter_log.txt（可选）
- 05_stopwords/stopwords_manifest.json
- 06_denoised_data/denoise_manifest.json
- 07_topic_models/topic_models_manifest.json（可选）
- 08_model_selection/best_mc_by_method.json
- 07_topic_models/<METHOD>/{PROJECT_PREFIX}_mc{best}_*.csv
- 09_visualization/<METHOD>/*.png（可选）

输出：
- 10_report/<METHOD>/{PROJECT_PREFIX}_{method}_report.md

用法：
- python step10_report.py
- python step10_report.py --only ABC
"""

from __future__ import annotations

import argparse
import json
import time
import re
import subprocess
import sys
import platform
import hashlib
import locale
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd

# HTML 转换支持
try:
    import markdown
    HAS_MARKDOWN = True
except ImportError:
    HAS_MARKDOWN = False

# DOCX 转换支持（可选）
try:
    import pypandoc  # type: ignore
    HAS_PYPANDOC = True
except ImportError:
    HAS_PYPANDOC = False

try:
    from config import PROJECT_PREFIX, SEARCH_KEYWORD, get_project_name
except ImportError:
    print("请确保 config.py 存在且配置正确")
    raise


ALL_METHODS = ["baseline", "VPD"]  # 已停用 A/B/AB/ABC，归档于 07_topic_models/_archived_AB_methods


def _safe_load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_filter_log(base_dir: Path) -> Optional[str]:
    logs = list((base_dir / "04_filtered_data").glob(f"{PROJECT_PREFIX}_filter_log.txt"))
    if not logs:
        return None
    try:
        return logs[0].read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None


def _pick_best_files(base_dir: Path, method: str, best_mc: int) -> Dict[str, Path]:
    d = base_dir / "07_topic_models" / method.upper()
    return {
        "topic_info": d / f"{PROJECT_PREFIX}_mc{best_mc}_topic_info.csv",
        "frontier": d / f"{PROJECT_PREFIX}_mc{best_mc}_frontier_indicators.csv",
        "doc_map": d / f"{PROJECT_PREFIX}_mc{best_mc}_doc_topic_mapping.csv",
        "summary": d / f"{PROJECT_PREFIX}_mc{best_mc}_summary.xlsx",
    }


def _load_step08_context(base_dir: Path) -> Dict[str, Any]:
    """Load Step08 selection artifacts for reporting."""
    out: Dict[str, Any] = {"best": {}, "scores_full": {}}
    best_path = base_dir / "08_model_selection" / "best_mc_by_method.json"
    scores_path = base_dir / "08_model_selection" / "cv_scores_full.json"
    best = _safe_load_json(best_path) or {}
    scores = _safe_load_json(scores_path) or {}
    out["best"] = best
    out["scores_full"] = scores
    out["best_mtime"] = best_path.stat().st_mtime if best_path.exists() else 0.0
    out["scores_mtime"] = scores_path.stat().st_mtime if scores_path.exists() else 0.0
    return out


def _get_pkg_version(name: str) -> str:
    try:
        from importlib.metadata import version  # type: ignore

        return version(name)
    except Exception:
        return "-"


def _get_git_commit_hash(base_dir: Path) -> str:
    """Best-effort git commit hash for reproducibility."""
    try:
        out = subprocess.check_output(["git", "-C", str(base_dir), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        s = out.decode("utf-8", errors="replace").strip()
        return s if s else "-"
    except Exception:
        return "-"


def _sha256_file(path: Path) -> str:
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return "-"


def _fmt_mtime(ts: float) -> str:
    try:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "-"


def _count_csv_rows_fast(path: Path) -> Optional[int]:
    """Count CSV rows quickly without pandas; returns number of data rows (excluding header)."""
    try:
        with path.open("rb") as f:
            n = 0
            for _ in f:
                n += 1
        return max(0, n - 1)
    except Exception:
        return None


def _step07_review_manifest(base_dir: Path, method: str) -> Optional[Dict[str, Any]]:
    p = base_dir / "07_topic_models" / method.upper() / "review_manifest.json"
    return _safe_load_json(p)


def _fingerprint_row(path: Path, *, want_rows: bool = True) -> str:
    if not path.exists():
        return f"| {path.as_posix()} | - | - | - | - |"
    st = path.stat()
    size_kb = st.st_size / 1024.0
    mtime = _fmt_mtime(st.st_mtime)
    rows = "-"
    if want_rows and path.suffix.lower() == ".csv":
        r = _count_csv_rows_fast(path)
        rows = str(r) if isinstance(r, int) else "-"
    sha = _sha256_file(path)
    sha12 = sha[:12] if isinstance(sha, str) and len(sha) >= 12 else sha
    return f"| {path.as_posix()} | {rows} | {mtime} | {size_kb:.1f} KB | {sha12} |"


def _reproducibility_section(base_dir: Path, method: str, *, best_mc: int) -> List[str]:
    """A compact reproducibility checklist for top-journal style reporting."""
    lines: List[str] = []
    lines.append("### 复现性清单（Reproducibility Checklist）\n")
    lines.append("| 项目 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| OS / Platform | {platform.platform()} |")
    lines.append(f"| Python | {platform.python_version()} |")

    # Code identity
    git_hash = _get_git_commit_hash(base_dir)
    lines.append(f"| Git commit hash | {git_hash} |")
    rm = _step07_review_manifest(base_dir, method) or {}
    code_version = rm.get("code_version") if isinstance(rm, dict) else None
    lines.append(f"| Step07 code_version | {code_version or '-'} |")

    # Seeds & sampling
    lines.append("| Step07 adaptive mc density sampling seed | 42 (固定；见 _engine_bertopic.py) |")
    lines.append("| Step08 text sampling seed | 20251220（默认，可用 step08_cv_select.py --seed 修改） |")
    lines.append("| Step08 max_docs | 3000（默认，可用 step08_cv_select.py --max_docs 修改；0=全量） |")

    # Key packages (best-effort)
    lines.append(f"| pandas | {_get_pkg_version('pandas')} |")
    lines.append(f"| numpy | {_get_pkg_version('numpy')} |")
    lines.append(f"| gensim | {_get_pkg_version('gensim')} |")
    lines.append(f"| bertopic | {_get_pkg_version('bertopic')} |")
    lines.append(f"| hdbscan | {_get_pkg_version('hdbscan')} |")
    lines.append(f"| umap-learn | {_get_pkg_version('umap-learn')} |")
    lines.append(f"| matplotlib | {_get_pkg_version('matplotlib')} |")
    lines.append(f"| seaborn | {_get_pkg_version('seaborn')} |")
    lines.append(f"| pypandoc (optional) | {_get_pkg_version('pypandoc')} |")
    lines.append("")

    # Data fingerprints
    lines.append("### 输入数据指纹（Input Fingerprints）\n")
    lines.append("说明：用于审稿复现对齐。sha256 取前 12 位；rows 为 CSV 数据行数（不含表头）。\n")
    lines.append("| 文件 | rows | mtime | size | sha256[:12] |")
    lines.append("|---|---:|---|---:|---|")

    # Step08 artifacts
    lines.append(_fingerprint_row(base_dir / "08_model_selection" / "best_mc_by_method.json", want_rows=False))
    lines.append(_fingerprint_row(base_dir / "08_model_selection" / "cv_scores_full.json", want_rows=False))

    # Step07 chosen outputs
    d = base_dir / "07_topic_models" / method.upper()
    lines.append(_fingerprint_row(d / f"{PROJECT_PREFIX}_mc{best_mc}_topic_info.csv"))
    lines.append(_fingerprint_row(d / f"{PROJECT_PREFIX}_mc{best_mc}_frontier_indicators.csv"))
    lines.append(_fingerprint_row(d / f"{PROJECT_PREFIX}_mc{best_mc}_doc_topic_mapping.csv"))
    lines.append(_fingerprint_row(d / "review_manifest.json", want_rows=False))

    # Step06 input to Step07
    lines.append(_fingerprint_row(base_dir / "06_denoised_data" / f"{PROJECT_PREFIX}_topic_modeling_{method}.csv"))

    lines.append("")
    lines.append("备注：若复现时图中文字出现方块，请确认系统存在可用中文字体（如 Microsoft YaHei/SimHei），或在 Step09 中检查字体自动选择日志。\n")
    return lines


def _maybe_refresh_step09(base_dir: Path, method: str, *, reference_mtime: float, force: bool = False) -> None:
    """Ensure Step09 outputs are present and not older than Step08 selection.

    This prevents Step10 embedding stale figures after Step08 changes.
    """
    out_dir = base_dir / "09_visualization" / method.upper()
    needs = force or (not out_dir.exists())

    key_files = [
        out_dir / "fig02_frontier_evolution.png",
        out_dir / "fig06_frontier_bubble.png",
        out_dir / "fig07_temporal_evolution.png",
        out_dir / "viz_report.html",
    ]

    if not needs:
        for p in key_files:
            if not p.exists():
                needs = True
                break
            try:
                if p.stat().st_mtime + 1e-6 < reference_mtime:
                    needs = True
                    break
            except Exception:
                needs = True
                break

    if not needs:
        return

    try:
        print(f"  [Step09] 检测到可视化缺失/过期，自动重跑: {method}")
        script = base_dir / "step09_visualization.py"
        subprocess.check_call([sys.executable, str(script), "--only", method], cwd=str(base_dir))
    except Exception as e:
        print(f"  [Step09] 自动重跑失败，将继续生成报告（可能嵌入旧图/缺图）：{str(e)[:160]}")


def _method_stopword_section(method: str, sw_manifest: Optional[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    lines.append("## Step 05 停用词/向量产物（A/B/C）\n")

    if sw_manifest is None:
        lines.append("- 未找到 stopwords_manifest.json（请先运行 step05_stopwords.py）\n")
        return lines

    schemes = sw_manifest.get("schemes", {})

    def add_scheme(letter: str, title: str):
        info = schemes.get(letter)
        if not info:
            lines.append(f"- 方案{letter}：未生成/缺失\n")
            return
        ok = "✅" if info.get("ok") else "❌"
        lines.append(f"- 方案{letter}（{title}）：{ok}")
        # steps
        steps = info.get("steps", [])
        if steps:
            lines.append(f"  - 子步骤数: {len(steps)}")
            for s in steps:
                lines.append(f"    - {s.get('name')} | rc={s.get('returncode')} | {s.get('seconds',0):.1f}s")
        # artifacts preview
        artifacts = info.get("artifacts", {})
        for k, v in artifacts.items():
            if isinstance(v, dict) and v.get("exists"):
                if "count" in v:
                    lines.append(f"  - {k}: {v.get('count')} 词 | {Path(v.get('path','')).as_posix()}")
                    sample = v.get("sample") or []
                    if sample:
                        lines.append("    - 示例(前25): " + ", ".join(sample[:25]))
                else:
                    # npz
                    lines.append(f"  - {k}: {v.get('bytes',0)/(1024*1024):.1f} MB | {Path(v.get('path','')).as_posix()}")
            elif isinstance(v, dict):
                lines.append(f"  - {k}: 缺失 | {Path(v.get('path','')).as_posix()}")

        lines.append("")

    # baseline：仍然汇报 Step05，但说明不使用
    if method == "baseline":
        lines.append("- baseline：本方法不使用停用词删词；仅用于对照。\n")
        return lines

    # A / AB / ABC
    if "A" in method:
        add_scheme("A", "统计去噪（SID→EVT→Dynamic IDF→Merger）")
    # B / AB / ABC
    if "B" in method:
        add_scheme("B", "语义扩展（SPA→CNI→SEC）")
    # C / ABC
    if "C" in method:
        add_scheme("C", "向量投影（V-Fusion→RepE→Output Vectors）")

    return lines


def _method_denoise_section(method: str, denoise_manifest: Optional[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    lines.append("## Step 06 文本去噪统计（按方法）\n")

    if denoise_manifest is None:
        lines.append("- 未找到 denoise_manifest.json（请先运行 step06_denoise.py）\n")
        return lines

    m = denoise_manifest.get("methods", {}).get(method)
    if not m:
        lines.append(f"- 未找到方法 {method} 的去噪统计\n")
        return lines

    lines.append(f"- 输出: {m.get('output_file')}")
    lines.append(f"- 行数: {m.get('rows_in')} → {m.get('rows_out')}")
    lines.append(f"- token: {m.get('total_tokens_before')} → {m.get('total_tokens_after')}（删除 {m.get('tokens_removed')}）")

    if method in ("baseline", "C"):
        lines.append("- 说明: 本方法不做删词（baseline 对照 / C 的贡献在 Step05 向量投影体现）\n")
        return lines

    lines.append(f"- 停用词: loaded={m.get('stopwords_loaded')} protected={m.get('protected_words')} effective={m.get('stopwords_effective')}")
    sw_sample = m.get("stopwords_sample") or []
    if sw_sample:
        lines.append("- 停用词示例(前25): " + ", ".join(sw_sample[:25]))

    removed_top = m.get("removed_top") or []
    if removed_top:
        top10 = removed_top[:10]
        lines.append("- 删除token Top10: " + ", ".join([f"{w}({c})" for w, c in top10]))

    lines.append("")
    return lines


def _method_topic_model_section(
    base_dir: Path,
    method: str,
    best_mc: int,
    topic_phrase_map: Optional[Dict[int, str]] = None,
) -> List[str]:
    lines: List[str] = []
    lines.append("## Step 07 主题建模与前沿识别（最终采用 best mc 版本）\n")

    files = _pick_best_files(base_dir, method, best_mc)
    ti = files["topic_info"]
    dm = files["doc_map"]
    fi = files["frontier"]

    lines.append(f"- best_mc = {best_mc}（来自 Step08 C_v 选择）")
    lines.append(f"- topic_info: {ti.as_posix()}")
    lines.append(f"- frontier_indicators: {fi.as_posix()}")
    lines.append(f"- doc_topic_mapping: {dm.as_posix()}\n")

    if ti.exists():
        dft = pd.read_csv(ti)
        if "Topic" in dft.columns:
            num_topics = int((dft["Topic"] >= 0).sum())
            lines.append(f"- 有效主题数: {num_topics}")
        if "Count" in dft.columns:
            if "Topic" in dft.columns:
                dft2 = dft[dft["Topic"] >= 0].copy()
            else:
                dft2 = dft.copy()
            top = dft2.sort_values("Count", ascending=False).head(5)
            lines.append("- Top5主题(按Count):")
            col = "TopWords" if "TopWords" in dft.columns else ("Representation" if "Representation" in dft.columns else None)
            for _, r in top.iterrows():
                try:
                    tid = int(r.get("Topic"))
                except Exception:
                    tid = None

                words = str(r.get(col, "")) if col else ""
                words = words.replace("\n", " ")

                if tid is not None and topic_phrase_map and tid in topic_phrase_map:
                    label = topic_phrase_map[tid]
                else:
                    if col == "TopWords":
                        kws = _extract_keywords_from_topwords(words, limit=10)
                    elif col == "Representation":
                        kws = _extract_keywords_from_representation(words, limit=10)
                    else:
                        kws = []
                    label = _generate_topic_phrase(kws)

                lines.append(f"  - {label}: Count={r.get('Count')} | {words[:120]}")

    if dm.exists():
        dfm = pd.read_csv(dm)
        if "Topic" in dfm.columns:
            noise = int((dfm["Topic"] == -1).sum())
            lines.append(f"- 文献数: {len(dfm)} | 噪声文献: {noise} | 噪声比例: {noise/max(1,len(dfm)):.2%}")

    lines.append("")
    return lines


def _method_viz_section(base_dir: Path, method: str) -> List[str]:
    lines: List[str] = []
    lines.append("## Step 09 可视化产物\n")
    d = base_dir / "09_visualization" / method.upper()
    if not d.exists():
        lines.append("- 未找到可视化目录（可先运行 step09_visualization.py）\n")
        return lines
    imgs = sorted(d.glob("*.png"))
    lines.append(f"- 图表数量: {len(imgs)}")

    viz_report = d / "viz_report.html"
    if viz_report.exists():
        rel_report = Path("../../") / "09_visualization" / method.upper() / viz_report.name
        lines.append(f"- 解释页: <a href=\"{rel_report.as_posix()}\">viz_report.html</a>")
    for img in imgs[:20]:
        # 相对路径：HTML 在 10_report/<METHOD>/ 下，需要回到根目录再进 09_visualization
        rel = Path("../../") / "09_visualization" / method.upper() / img.name
        lines.append(f"  - ![{img.stem}]({rel.as_posix()})")
    lines.append("")
    return lines


def _citations_section() -> List[str]:
    return [
        "## 参考文献\n",
        "- Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. arXiv:2203.05794.",
        "- McInnes, L., Healy, J., & Astels, S. (2017). hdbscan: Hierarchical density based clustering. JOSS, 2(11), 205.",
        "- McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection. arXiv:1802.03426.",
        "- Campello, R. J., Moulavi, D., & Sander, J. (2013). Density-based clustering based on hierarchical density estimates. PAKDD.",
        "- Röder, M., Both, A., & Hinneburg, A. (2015). Exploring the space of topic coherence measures. WSDM.",
        "- Newman, D., Lau, J. H., Grieser, K., & Baldwin, T. (2010). Automatic evaluation of topic coherence. NAACL.",
        "- Shannon, C. E. (1948). A Mathematical Theory of Communication. Bell System Technical Journal.",
        "- Small, H., Boyack, K. W., & Klavans, R. (2014). Identifying emerging topics in science and technology. Research Policy, 43(8), 1450-1467.",
        "- Chen, C. (2006). CiteSpace II: Detecting and visualizing emerging trends and transient patterns in scientific literature. JASIST, 57(3), 359-377.",
        "- Diakoulaki, D., Mavrotas, G., & Papayannakis, L. (1995). Determining objective weights in multiple criteria problems: The CRITIC method. Computers & Operations Research, 22(7), 763-770.",
        "- Zou, A., et al. (2023). Representation Engineering. arXiv:2310.01405.",
        "",
    ]


def _methodology_section(*, base_dir: Path, method: str, best_mc: int) -> List[str]:
    """生成研究方法论章节，包含公式和理论依据"""
    return [
        "## 研究方法\n",
        "### 技术路线\n",
        "本研究采用以下流程进行主题建模与研究前沿识别：\n",
        "```",
        "Step 1: 数据采集 (PubMed API) → Step 2: 引用数据补充 (OpenCitations)",
        "    ↓",
        "Step 3: 数据清洗与预处理 → Step 4: 文献类型过滤",
        "    ↓",
        "Step 5: 文本去噪 (停用词 + 保护词) → Step 6: 去噪数据应用",
        "    ↓",
        "Step 7: BERTopic 主题建模 → Step 8: C_v 一致性评估",
        "    ↓",
        "Step 9: 可视化分析 → Step 10: 报告生成",
        "```\n",
        "### 核心算法\n",
        "#### BERTopic 主题建模\n",
        "BERTopic 是一种基于 Transformer 的神经主题建模方法 (Grootendorst, 2022)，核心流程如下：\n",
        "**Step 1: 文档嵌入** - 使用预训练语言模型将文档转换为语义向量：",
        "$$\\mathbf{e}_i = \\text{SentenceTransformer}(d_i), \\quad \\mathbf{e}_i \\in \\mathbb{R}^{384}$$\n",
        "**Step 2: 降维 (UMAP)** - 使用 UMAP 进行非线性降维 (McInnes et al., 2018)：",
        "$$\\mathbf{u}_i = \\text{UMAP}(\\mathbf{e}_i; n_{\\text{neighbors}}=15, n_{\\text{components}}=5)$$\n",
        "**Step 3: 聚类 (HDBSCAN)** - 使用 HDBSCAN 进行密度聚类 (McInnes et al., 2017)：",
        "$$c_i = \\text{HDBSCAN}(\\mathbf{u}_i; \\text{min\\_cluster\\_size})$$",
        "其中 $c_i \\in \\{-1, 0, 1, ..., K\\}$，$c_i = -1$ 表示噪声点。\n",
        "**Step 4: 主题表示 (c-TF-IDF)** - 基于类别的 TF-IDF 提取主题关键词：",
        "$$\\text{c-TF-IDF}_{t,c} = \\frac{f_{t,c}}{\\sum_{t'} f_{t',c}} \\cdot \\log\\left(1 + \\frac{A}{f_t}\\right)$$",
        "其中 $f_{t,c}$ 为词 $t$ 在主题 $c$ 中的频率，$A$ 为文档总数。\n",
        "#### CRITIC 客观赋权法\n",
        "CRITIC (Diakoulaki et al., 1995) 是一种基于指标对比强度和冲突性的客观赋权方法：\n",
        "**Step 1: 数据标准化**",
        "$$x_{ij}^* = \\frac{x_{ij} - \\min_j(x_{ij})}{\\max_j(x_{ij}) - \\min_j(x_{ij})}$$\n",
        "**Step 2: 计算信息量**",
        "$$C_j = \\sigma_j \\cdot \\sum_{k \\neq j} (1 - r_{jk})$$",
        "其中 $\\sigma_j$ 为指标标准差，$r_{jk}$ 为指标间相关系数。\n",
        "**Step 3: 计算权重**",
        "$$w_j = \\frac{C_j}{\\sum_{k=1}^m C_k}$$\n",
        "#### 研究前沿综合指数\n",
        "$$\\text{Composite Index}_i = \\sum_{j=1}^{5} w_j \\cdot x_{ij}^*$$",
        "其中五个评价指标为：Strength（规模）、Novelty（新颖性）、Heat（热度）、Avg_Citations（引用度）、HighCited_Count（高被引）。\n",

        "#### Step08：min_cluster_size (mc) 的多目标选择（硬约束 + Pareto）\n",
        "为保证可复现与审稿可解释性，我们将 mc 选择表述为标准多目标问题。对候选 $mc_i$ 定义：",
        "$$c_i = C_v(mc_i) \\quad (\\text{maximize}), \\qquad n_i = noise(mc_i) \\quad (\\text{minimize})$$",
        "其中 $noise$ 为 doc_topic_mapping 中 $Topic=-1$ 的文档占比。对非 baseline 方法，引入来自父方法的噪音约束阈值 $r$：",
        "$$F(r)=\\{i\\mid n_i \\le r\\}$$",
        "Pareto（帕累托）支配：候选 $a$ 支配 $b$ 当且仅当 $(c_a\\ge c_b) \\wedge (n_a\\le n_b)$ 且至少一项严格更优；帕累托前沿为所有不被支配的候选集合。",
        "决策规则：",
        "- 若 $F(r)$ 非空：在 $Pareto(F)$ 上选择 $c$ 最大者；若并列选 $n$ 更小者（再并列选更小的 $mc$，确保确定性）。",
        "- 若 $F(r)$ 为空：说明没有候选满足噪音阈值；在 $Pareto(All)$ 上优先选择 $n$ 最小者；若并列选择 $c$ 更大者（再并列选更小的 $mc$）。\n",
        "该策略避免用更差的噪音换取表观更高的一致性分数，并在不可行时给出保守、可解释的退化路径（Röder et al., 2015; Newman et al., 2010）。\n",

        "#### Step07：自适应候选 mc 的生成（公式 + 内在原因）\n",
        "候选 mc 不是人工固定列表，而由文档规模与语料统计自适应生成。设文档数为 $N$：",
        "$$mc_{base} = \\alpha\\sqrt{N} + \\beta\\ln(N)$$",
        "进一步用文本多样性（词汇香农熵归一化）与向量空间密度做调整：",
        "$$mc_{adj} = mc_{base}\\cdot(1+\\gamma H_{vocab}^{norm}+\\delta\\,density_{factor})$$",
        "最后取多个缩放版本并裁剪到合理上界得到候选集（并去重）：",
        "$$mc\\in\\{1.3\\,mc_{adj},\\ 1.0\\,mc_{adj},\\ 0.7\\,mc_{adj},\\ 0.4\\,mc_{adj}\\}$$",
        "**为什么候选常常看起来‘重复’**：不同去噪方法若 $N$ 与统计特征接近，上式会得到相近的 $mc_{adj}$，离散化（取整+clip）后自然落在相同的整数集合中。为保证复现，我们固定了密度抽样的随机种子。\n",

        "#### 符号表（Notation）\n",
        "| 符号 | 含义 |",
        "|---|---|",
        "| $N$ | 文档数 |",
        "| $mc$ / $mc_i$ | HDBSCAN 的 min\\_cluster\\_size（候选超参数） |",
        "| $c_i$ | $C_v(mc_i)$，主题一致性（越大越好） |",
        "| $n_i$ | $noise(mc_i)$，噪音比例（Topic=-1 占比，越小越好） |",
        "| $r$ | 噪音约束阈值（来自父方法的最小噪音） |",
        "| $F(r)$ | 可行集 $\\{i\\mid n_i\\le r\\}$ |",
        "| $Pareto(\\cdot)$ | 不被其它候选支配的集合（双目标：max $c$ / min $n$） |\n",

        *_reproducibility_section(base_dir, method, best_mc=best_mc),

        "### 前沿分类规则\n",
        "基于 Small et al. (2014) 和 Chen (2006) 的研究前沿理论：\n",
        "| 分类 | 中文 | 判定条件 | 理论依据 |",
        "|-----|-----|---------|---------|",
        "| 🔥 **Hotspot** | 热点 | Composite ≥ 50% 且 Heat ≥ 50% | Chen (2006) citation burst |",
        "| 🌱 **Emerging** | 新兴 | Novelty ≥ 60% 且 Growth ≥ 30% | Small et al. (2014) |",
        "| 💎 **Potential** | 潜在 | Novelty ≥ 75% 且 Composite < 50% | Shibata et al. (2008) |",
        "| 📉 **Declining** | 衰退 | Novelty < 60% 且 Heat < 50% | 逆向推断 |",
        "| ➖ **General** | 一般 | 其他情况 | 稳定常规领域 |\n",
        "",
    ]


def _key_findings_section(base_dir: Path, method: str, best_mc: int, topic_phrase_map: Optional[Dict[int, str]] = None) -> List[str]:
    """生成关键发现章节"""
    lines: List[str] = []
    lines.append("## 关键发现与研究建议\n")
    
    files = _pick_best_files(base_dir, method, best_mc)
    frontier_file = files["frontier"]
    
    if not frontier_file.exists():
        lines.append("- 未找到前沿指标文件\n")
        return lines
    
    try:
        df = pd.read_csv(frontier_file)
    except:
        return lines
    
    # 统计前沿类型分布
    if "Frontier_Type" in df.columns:
        type_counts = df["Frontier_Type"].value_counts().to_dict()
        total = len(df)
        
        lines.append("### 前沿主题分类统计\n")
        lines.append("| 类型 | 数量 | 占比 | 说明 |")
        lines.append("|-----|-----|-----|------|")
        
        type_info = {
            "热点": ("🔥", "当前学术界高度关注的研究方向"),
            "新兴": ("🌱", "近年快速发展的新兴领域"),
            "潜在": ("💎", "具有发展潜力的前沿方向"),
            "衰退": ("📉", "研究热度下降的传统领域"),
            "一般": ("➖", "常规稳定的研究领域"),
        }
        
        for ft, (emoji, desc) in type_info.items():
            count = type_counts.get(ft, 0)
            pct = count / total * 100 if total > 0 else 0
            lines.append(f"| {emoji} {ft} | {count} | {pct:.1f}% | {desc} |")
        
        lines.append("")
    
    # 辅助函数：从TopWords生成短语
    def _get_phrase_from_row(row):
        try:
            tid = int(row.get("Topic"))
        except Exception:
            tid = None
        if tid is not None and topic_phrase_map and tid in topic_phrase_map:
            return topic_phrase_map[tid]
        keywords = _extract_keywords_from_topwords(row.get("TopWords", ""), limit=8)
        return _generate_topic_phrase(keywords)
    
    # 识别Top热点
    if "Heat_RecentRatio" in df.columns and "TopWords" in df.columns:
        lines.append("### Current Research Hotspots (Heat Top 5)\n")
        hot_topics = df.nlargest(5, "Heat_RecentRatio")
        for i, (_, row) in enumerate(hot_topics.iterrows(), 1):
            phrase = _get_phrase_from_row(row)
            heat = row["Heat_RecentRatio"]
            lines.append(f"{i}. **{phrase}** (Recent ratio: {heat:.1%})")
        lines.append("")
    
    # 识别新兴方向
    if "Novelty_AvgYear" in df.columns:
        lines.append("### Emerging Research Directions (Novelty Top 5)\n")
        novel_topics = df.nlargest(5, "Novelty_AvgYear")
        for i, (_, row) in enumerate(novel_topics.iterrows(), 1):
            phrase = _get_phrase_from_row(row)
            year = row["Novelty_AvgYear"]
            lines.append(f"{i}. **{phrase}** (Avg. year: {year:.1f})")
        lines.append("")
    
    # 高影响力主题
    if "Avg_Citations" in df.columns:
        lines.append("### High-Impact Research (Citations Top 5)\n")
        cited_topics = df.nlargest(5, "Avg_Citations")
        for i, (_, row) in enumerate(cited_topics.iterrows(), 1):
            phrase = _get_phrase_from_row(row)
            cit = row["Avg_Citations"]
            lines.append(f"{i}. **{phrase}** (Avg. citations: {cit:.1f})")
        lines.append("")
    
    lines.append("### Research Recommendations\n")
    lines.append("1. **Focus on Hotspots**: Track Heat Top 10 topics - these are currently highly active research areas")
    lines.append("2. **Identify Emerging Trends**: High Novelty topics represent the latest research trends")
    lines.append("3. **Study High-Impact Work**: Topics with high citations contain seminal papers worth reading")
    lines.append("4. **Explore Potential Topics**: Potential-type topics may become future hotspots")
    lines.append("")
    
    return lines


# 主题短语模板（基于关键词组合生成专业描述）
TOPIC_PHRASE_TEMPLATES = {
    # 治疗相关
    ("therapy", "eradication"): "H. pylori eradication therapy",
    ("eradication", "triple"): "Triple therapy for H. pylori eradication",
    ("eradication", "quadruple"): "Quadruple therapy regimens",
    ("treatment", "resistance"): "Antibiotic resistance in treatment",
    ("vonoprazan", "eradication"): "Vonoprazan-based eradication therapy",
    ("probiotics", "eradication"): "Probiotic-supplemented eradication therapy",
    ("medicine", "chinese"): "Traditional Chinese medicine therapy",
    ("decoction", "chinese"): "Chinese herbal decoction treatment",
    ("guidelines", "consensus"): "Clinical guidelines and consensus",
    ("management", "guidelines"): "Clinical management guidelines",
    
    # 癌症相关
    ("cancer", "gastric"): "Gastric cancer pathogenesis",
    ("cancer", "risk"): "Cancer risk factors",
    ("cancer", "incidence"): "Cancer incidence and trends",
    ("cancer", "mortality"): "Cancer incidence and mortality trends",
    ("adenocarcinoma", "gastric"): "Gastric adenocarcinoma",
    ("carcinogenesis", "gastric"): "Gastric carcinogenesis mechanisms",
    ("tumor", "immune"): "Tumor immune microenvironment",
    ("immunotherapy", "cancer"): "Cancer immunotherapy",
    ("immunotherapy", "pdl1"): "PD-L1/PD-1 immunotherapy",
    ("lncrnas", "cancer"): "LncRNA in cancer progression",
    ("lncrna", "expression"): "LncRNA expression profiling",
    
    # 溃疡相关
    ("ulcer", "peptic"): "Peptic ulcer disease",
    ("ulcer", "bleeding"): "Peptic ulcer bleeding",
    ("ulcer", "duodenal"): "Duodenal ulcer",
    ("nsaid", "ulcer"): "NSAID-induced ulcer",
    ("aspirin", "bleeding"): "Aspirin-associated GI bleeding",
    
    # 淋巴瘤相关
    ("lymphoma", "malt"): "MALT lymphoma",
    ("lymphoma", "gastric"): "Gastric lymphoma",
    
    # 毒力因子
    ("caga", "vaca"): "CagA/VacA virulence factors",
    ("caga", "positive"): "CagA-positive strains",
    ("virulence", "factors"): "Virulence factor analysis",
    ("genotypes", "virulence"): "Virulence genotyping",
    
    # 微生物组
    ("microbiota", "gut"): "Gut microbiota interaction",
    ("microbiome", "gastric"): "Gastric microbiome",
    ("dysbiosis", "microbiota"): "Microbiota dysbiosis",
    
    # 诊断与检测
    ("diagnosis", "endoscopy"): "Endoscopic diagnosis",
    ("detection", "molecular"): "Molecular detection methods",
    ("detection", "electrochemical"): "Electrochemical biosensor detection",
    ("detection", "lamp"): "LAMP-based rapid detection",
    ("detection", "dna"): "DNA-based detection methods",
    ("test", "urea"): "Urea breath test",
    ("biopsy", "histology"): "Histological biopsy analysis",
    ("ai", "detection"): "AI-assisted diagnosis",
    ("ai", "images"): "AI-based image analysis",
    ("images", "learning"): "Machine learning image analysis",
    ("artificial", "intelligence"): "Artificial intelligence applications",
    
    # 炎症与免疫
    ("gastritis", "chronic"): "Chronic gastritis",
    ("gastritis", "atrophic"): "Atrophic gastritis",
    ("inflammation", "gastric"): "Gastric inflammation",
    ("immune", "response"): "Immune response mechanisms",
    ("cytokines", "inflammation"): "Cytokine-mediated inflammation",
    ("il", "expression"): "Interleukin expression",
    ("il", "cells"): "Interleukin and immune cells",
    
    # 细胞机制
    ("cells", "epithelial"): "Epithelial cell responses",
    ("cells", "expression"): "Cell gene expression",
    ("apoptosis", "cells"): "Cell apoptosis mechanisms",
    ("signaling", "pathway"): "Signaling pathway analysis",
    ("expression", "genes"): "Gene expression analysis",
    
    # 流行病学
    ("prevalence", "infection"): "Infection prevalence",
    ("prevalence", "children"): "Prevalence in children",
    ("epidemiology", "global"): "Global epidemiology",
    ("transmission", "infection"): "Transmission patterns",
    
    # 耐药性
    ("resistance", "antibiotic"): "Antibiotic resistance",
    ("resistance", "clarithromycin"): "Clarithromycin resistance",
    ("resistance", "metronidazole"): "Metronidazole resistance",
    ("resistance", "mutations"): "Antibiotic resistance mutations",
    ("mutations", "resistance"): "Resistance mutations",
    
    # 其他疾病关联
    ("diabetes", "mellitus"): "Diabetes mellitus association",
    ("cardiovascular", "disease"): "Cardiovascular disease link",
    ("liver", "nafld"): "NAFLD and liver disease",
    ("nafld", "fatty"): "Non-alcoholic fatty liver disease",
    ("iron", "deficiency"): "Iron deficiency anemia",
    ("pancreatic", "cancer"): "Pancreatic cancer association",
    ("pancreatitis", "pancreatic"): "Pancreatitis studies",
    
    # 天然产物与抗菌
    ("extract", "activity"): "Natural extract activity",
    ("compounds", "antibacterial"): "Antibacterial compounds",
    ("probiotics", "effects"): "Probiotic effects",
    ("probiotics", "diarrhea"): "Probiotics for diarrhea prevention",
    
    # 内镜相关
    ("endoscopic", "resection"): "Endoscopic resection",
    ("esd", "submucosal"): "Endoscopic submucosal dissection",
    ("metaplasia", "intestinal"): "Intestinal metaplasia",
    ("lesion", "endoscopic"): "Endoscopic lesion management",
    
    # 病毒与感染
    ("virus", "cancers"): "Infection-attributable cancers",
    ("hepatitis", "virus"): "Hepatitis virus co-infection",
    ("hpv", "cancer"): "HPV-related cancers",
    
    # 酶与蛋白
    ("urease", "activity"): "Urease enzyme activity",
    ("urease", "inhibitors"): "Urease inhibitors",
    ("proteins", "protein"): "Protein structure analysis",
    ("adhesion", "binding"): "Bacterial adhesion mechanisms",
    
    # 外膜囊泡与分泌
    ("omvs", "vesicles"): "Outer membrane vesicles",
    ("secretion", "system"): "Type IV secretion system",
    
    # 动物模型
    ("mice", "model"): "Mouse model studies",
    ("mouse", "infection"): "Mouse infection model",
    ("animal", "model"): "Animal model studies",
    
    # 药物与临床
    ("drug", "delivery"): "Drug delivery systems",
    ("drug", "efficacy"): "Drug efficacy studies",
    ("ppi", "inhibitor"): "Proton pump inhibitor therapy",
    ("amoxicillin", "clarithromycin"): "Amoxicillin-clarithromycin regimen",
    
    # 基因与遗传
    ("gene", "expression"): "Gene expression profiling",
    ("polymorphisms", "risk"): "Genetic polymorphism risk",
    ("snp", "association"): "SNP association studies",
    
    # 血清学与诊断
    ("serology", "antibody"): "Serological antibody testing",
    ("igg", "antibody"): "IgG antibody detection",
    ("stool", "antigen"): "Stool antigen test",
    
    # 地区研究
    ("chinese", "population"): "Chinese population study",
    ("asian", "population"): "Asian population study",
    ("pediatric", "children"): "Pediatric infection",
}


def _generate_topic_phrase(keywords: List[str]) -> str:
    """根据关键词生成专业的英文主题短语描述"""
    if not keywords:
        return "General topic"
    
    # 标准化关键词
    kw_lower = [k.lower().strip() for k in keywords[:10]]
    
    # 尝试匹配预定义模板（按两个关键词组合查找）
    for (k1, k2), phrase in TOPIC_PHRASE_TEMPLATES.items():
        if k1 in kw_lower and k2 in kw_lower:
            return phrase
    
    # 单关键词专业术语映射
    single_term_map = {
        "urease": "Urease enzyme studies",
        "omvs": "Outer membrane vesicles (OMVs)",
        "biofilm": "Biofilm formation",
        "adhesin": "Adhesin-mediated colonization",
        "flagella": "Flagellar motility",
        "chemotaxis": "Chemotaxis mechanisms",
        "autophagy": "Autophagy pathway",
        "apoptosis": "Apoptosis regulation",
        "vaccine": "Vaccine development",
        "nanoparticles": "Nanoparticle-based therapy",
        "curcumin": "Curcumin anti-H. pylori activity",
        "garlic": "Garlic extract antimicrobial effects",
        "honey": "Honey antibacterial properties",
        "propolis": "Propolis antimicrobial activity",
        "lactoferrin": "Lactoferrin antimicrobial effects",
    }
    
    # 检查单关键词匹配
    for kw in kw_lower[:3]:
        if kw in single_term_map:
            return single_term_map[kw]
    
    # 如果没有匹配，尝试智能组合
    # 策略：核心名词 + 修饰词 + 研究类型
    core_nouns = ["therapy", "treatment", "cancer", "ulcer", "gastritis", "lymphoma", 
                  "infection", "resistance", "microbiota", "diagnosis", "eradication",
                  "adenocarcinoma", "carcinoma", "metaplasia", "dysplasia", "inflammation",
                  "colonization", "pathogenesis", "virulence"]
    
    modifiers = ["gastric", "peptic", "chronic", "atrophic", "intestinal", "duodenal",
                 "antibiotic", "triple", "quadruple", "molecular", "endoscopic",
                 "bacterial", "mucosal", "epithelial", "systemic"]
    
    study_types = ["analysis", "mechanisms", "factors", "patterns", "effects", 
                   "response", "expression", "pathogenesis", "association", "studies",
                   "activity", "regulation", "interaction"]
    
    found_noun = None
    found_modifier = None
    found_study = None
    
    for kw in kw_lower:
        if not found_noun:
            for noun in core_nouns:
                if noun in kw:
                    found_noun = kw
                    break
        if not found_modifier:
            for mod in modifiers:
                if mod in kw:
                    found_modifier = kw
                    break
        if not found_study:
            for st in study_types:
                if st in kw:
                    found_study = kw
                    break
    
    # 组合短语
    parts = []
    if found_modifier:
        parts.append(found_modifier.capitalize())
    if found_noun:
        parts.append(found_noun)
    elif keywords:
        parts.append(keywords[0])
    if found_study and len(parts) < 3:
        parts.append(found_study)
    
    if len(parts) >= 2:
        # 形成短语：如 "Gastric cancer analysis"
        phrase = " ".join(parts[:3])
        return phrase.capitalize() if phrase else keywords[0].capitalize()
    
    # 备选：使用前2-3个关键词组合成短语，添加 "studies" 后缀使其更专业
    if len(keywords) >= 2:
        base = f"{keywords[0].capitalize()} {keywords[1]}"
        if len(keywords) >= 3 and len(keywords[2]) > 3:
            base = f"{base} {keywords[2]}"
        return base + " studies" if len(base) < 30 else base
    
    # 最后备选：单个关键词 + "research"
    return f"{keywords[0].capitalize()} research"


def _extract_keywords_from_topwords(words_str: str, limit: int = 8) -> List[str]:
    s = str(words_str or "").replace(";", ", ").strip()
    kws = [w.strip() for w in s.split(",") if w.strip()]
    return kws[:limit]


def _extract_keywords_from_representation(rep: str, limit: int = 8) -> List[str]:
    s = str(rep or "").strip().strip("[]").replace("'", "").replace('"', "")
    kws = [w.strip() for w in s.split(",") if w.strip()]
    return kws[:limit]


def _contains_cjk(s: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", str(s or "")))


def _translate_en_phrase_to_cn(en_phrase: str) -> str:
    """将英文短语粗翻为中文短语（用于 Topic Description 的中文部分）。

    目标：可读、领域相关、宁可保留少量英文也不胡译。
    """
    CN = {
        # general
        "study": "研究",
        "studies": "研究",
        "analysis": "分析",
        "mechanism": "机制",
        "mechanisms": "机制",
        "prevention": "预防",
        "rate": "率",
        "rates": "率",
        "factor": "因素",
        "factors": "因素",
        "gut": "肠道",
        "effects": "作用",
        "effect": "作用",
        "association": "关联",
        "associations": "关联",
        "risk": "风险",
        "management": "管理",
        "guidelines": "指南",
        "consensus": "共识",
        "response": "反应",
        "expression": "表达",
        "interaction": "相互作用",
        "regulation": "调控",
        "triple": "三联",
        "quadruple": "四联",
        "patterns": "模式",
        "pathogenesis": "发病机制",

        # domain
        "microbial": "微生物",
        "gastric": "胃",
        "cancer": "癌",
        "carcinoma": "癌",
        "adenocarcinoma": "腺癌",
        "ulcer": "溃疡",
        "peptic": "消化性",
        "gastritis": "胃炎",
        "lymphoma": "淋巴瘤",
        "malt": "MALT",
        "therapy": "治疗",
        "treatment": "治疗",
        "eradication": "根除",
        "antibiotic": "抗生素",
        "resistance": "耐药",
        "microbiota": "微生物群",
        "microbiome": "微生物组",
        "diagnosis": "诊断",
        "detection": "检测",

        # epidemiology / natural products
        "infection": "感染",
        "prevalence": "患病率",
        "natural": "天然",
        "extract": "提取物",
        "extracts": "提取物",
        "activity": "活性",
        "endoscopic": "内镜",
        "endoscopy": "内镜",
        "immune": "免疫",
        "immunotherapy": "免疫治疗",
        "tumor": "肿瘤",
        "biofilm": "生物膜",
        "probiotic": "益生菌",
        "probiotics": "益生菌",
        "urease": "尿素酶",
        "virulence": "毒力",
        "vaccine": "疫苗",
        "nanoparticle": "纳米颗粒",
        "nanoparticles": "纳米颗粒",
        "drug": "药物",
        "delivery": "递送",
        "ppi": "PPI",

        # organism
        "helicobacter": "幽门螺杆菌",
        "pylori": "幽门螺杆菌",
        "h": "幽门螺杆菌",
    }

    phrase = str(en_phrase or "").strip()
    # 预规范化：将 H. pylori 视为一个整体概念，避免 h + pylori 重复翻译
    phrase = re.sub(r"\bH\.?\s*pylori\b", "HPYLORI", phrase, flags=re.IGNORECASE)
    # 处理分隔符
    phrase = phrase.replace("—", "-")
    parts = [p.strip() for p in phrase.split("-") if p.strip()]

    def _render_part(part: str) -> str:
        tokens = re.findall(r"[A-Za-z0-9]+", part)
        mapped = []
        for tok in tokens:
            low = tok.lower()
            if low == "hpylori":
                mapped.append("幽门螺杆菌")
                continue
            if low in ("caga", "vaca"):
                mapped.append(tok)
                continue
            mapped.append(CN.get(low, tok))

        # 连写中文，保留必要空格
        out = ""
        for t in mapped:
            if not out:
                out = t
                continue
            if _contains_cjk(out[-1]) and _contains_cjk(t[:1]):
                out += t
            else:
                out += " " + t
        # 特例："胃"+"癌" -> "胃癌"
        out = out.replace("胃 癌", "胃癌")
        # 美化：中文分隔符
        out = out.replace(",", "、")
        out = re.sub(r"\s+", " ", out).strip()
        return out

    cn_parts = [_render_part(p) for p in parts]
    cn = "—".join([p for p in cn_parts if p])
    cn = cn.replace("幽门螺杆菌 幽门螺杆菌", "幽门螺杆菌")
    cn = cn.replace("幽门螺杆菌幽门螺杆菌", "幽门螺杆菌")
    cn = re.sub(r"(幽门螺杆菌){2,}", "幽门螺杆菌", cn)
    return cn.strip()


def _make_bilingual_topic_desc(en_phrase: str, fallback_keywords: List[str]) -> str:
    cn = _translate_en_phrase_to_cn(en_phrase)
    if not _contains_cjk(cn):
        # 回退：从关键词里尽量拼出中文（保留少量英文也可以）
        kws = [k.strip() for k in (fallback_keywords or []) if k.strip()]
        cn_kws = _translate_en_phrase_to_cn(" ".join(kws[:4]))
        cn = cn_kws if _contains_cjk(cn_kws) else "研究主题"
    # 使用中文全角括号，避免被 step11 误判成再次替换
    return f"{cn}（{en_phrase}）"


def _build_unique_topic_phrase_map(topic_info_df: pd.DataFrame) -> Dict[int, str]:
    """为同一方法内的所有 Topic 生成稳定且唯一的主题短语。

    严谨原则：
    - 先按既有规则生成短语（保持可读性与一致性）
    - 若出现重复短语（不同 Topic 生成同一 phrase），则优先追加“差异关键词”消歧；仍冲突则追加 Variant 序号（不暴露 Topic 编号）
    - 同一 Topic 在整份报告的任何位置都使用同一个短语
    """
    phrase_by_topic: Dict[int, str] = {}
    topics_by_phrase: Dict[str, List[int]] = {}

    if topic_info_df is None or topic_info_df.empty or "Topic" not in topic_info_df.columns:
        return phrase_by_topic

    keywords_by_topic: Dict[int, List[str]] = {}
    keyword_freq: Dict[str, int] = {}

    for _, row in topic_info_df.iterrows():
        try:
            topic_id = int(row.get("Topic"))
        except Exception:
            continue
        if topic_id < 0:
            continue

        if "Representation" in topic_info_df.columns:
            keywords = _extract_keywords_from_representation(row.get("Representation", ""), limit=10)
        elif "TopWords" in topic_info_df.columns:
            keywords = _extract_keywords_from_topwords(row.get("TopWords", ""), limit=10)
        else:
            keywords = []

        keywords_by_topic[topic_id] = keywords
        for w in keywords:
            wl = str(w or "").strip().lower()
            if wl:
                keyword_freq[wl] = keyword_freq.get(wl, 0) + 1

        base_phrase_en = _generate_topic_phrase(keywords)
        phrase_by_topic[topic_id] = base_phrase_en
        topics_by_phrase.setdefault(base_phrase_en, []).append(topic_id)

    # 严格消歧（语义化优先）：同 phrase 多 topic 时，用“低频差异关键词”（近似 IDF）做区分，仍冲突则追加 Variant 序号
    for base_phrase, topic_ids in topics_by_phrase.items():
        if len(topic_ids) <= 1:
            continue

        # base_phrase 的词集合（用于排除“已经表达过”的词）
        base_tokens = set([t.lower() for t in re.findall(r"[A-Za-z0-9]+", base_phrase)])

        new_phrases: Dict[int, str] = {}
        for tid in topic_ids:
            words = [w.strip() for w in (keywords_by_topic.get(tid) or []) if str(w).strip()]
            # 选择低频词优先（更能区分主题），并排除 base_phrase 已含词
            scored = []
            for w in words:
                wl = w.lower()
                if wl in base_tokens:
                    continue
                scored.append((keyword_freq.get(wl, 999), words.index(w), w))
            scored.sort(key=lambda x: (x[0], x[1]))
            extra = [x[2] for x in scored[:2] if x[2]]

            if extra:
                new_phrases[tid] = f"{base_phrase} — {', '.join(extra)}"
            else:
                new_phrases[tid] = base_phrase

        # 检查是否已消歧
        inv: Dict[str, List[int]] = {}
        for tid, phr in new_phrases.items():
            inv.setdefault(phr, []).append(tid)
        still_dup = {phr: tids for phr, tids in inv.items() if len(tids) > 1}
        if not still_dup:
            for tid, phr in new_phrases.items():
                phrase_by_topic[tid] = phr
            continue

        # 兜底：仍重复则追加 Variant 序号（不显示 Topic ID）
        for i, tid in enumerate(sorted(topic_ids)):
            phrase_by_topic[tid] = f"{new_phrases.get(tid, base_phrase)} — v{i+1}"

    # 最终输出：中文（英文）双语（直接写入报告，避免后续翻译不全/英文(英文)）
    bilingual: Dict[int, str] = {}
    for tid, en in phrase_by_topic.items():
        bilingual[tid] = _make_bilingual_topic_desc(en, fallback_keywords=keywords_by_topic.get(tid, []))

    return bilingual


def _method_frontier_metrics_section(base_dir: Path, method: str, best_mc: int, topic_phrase_map: Optional[Dict[int, str]] = None) -> List[str]:
    """生成该方法的研究前沿五个指标 Top 10 排名"""
    lines: List[str] = []
    lines.append("## 研究前沿指标分析\n")
    lines.append("按五个关键指标排序，每个指标展示得分最高的 10 个主题。\n")
    lines.append("**指标说明**：")
    lines.append("- **Strength（强度）**: 主题规模占比，反映研究领域的重要程度")
    lines.append("- **Novelty（新颖性）**: 平均发表年份，年份越新表示研究越前沿")
    lines.append("- **Heat（热点）**: 近3年文献占比，反映当前学术关注度")
    lines.append("- **Avg_Citations（引用度）**: 平均被引用次数，反映学术影响力")
    lines.append("- **HighCited_Count（高被引）**: 高被引文献数（≥30次），反映经典程度\n")
    
    files = _pick_best_files(base_dir, method, best_mc)
    frontier_file = files["frontier"]
    
    if not frontier_file.exists():
        lines.append("- 未找到研究前沿指标文件（请先运行 step07_topic_model.py）\n")
        return lines
    
    try:
        df = pd.read_csv(frontier_file)
    except Exception as e:
        lines.append(f"- 无法读取前沿指标文件：{str(e)}\n")
        return lines
    
    # 五个指标的中英文标签
    metrics = {
        "Strength": {"cn": "强度（文献数量占比）", "format": ".4f"},
        "Novelty_AvgYear": {"cn": "新颖性（平均发表年份）", "format": ".1f"},
        "Heat_RecentRatio": {"cn": "热点（近期文献占比）", "format": ".2%"},
        "Avg_Citations": {"cn": "引用度（平均被引用次数）", "format": ".2f"},
        "HighCited_Count": {"cn": "高被引（高被引文献数）", "format": ".0f"},
    }
    
    for metric_col, metric_info in metrics.items():
        lines.append(f"\n### {metric_col} - {metric_info['cn']}\n")
        
        # 排序并取 Top 10
        if metric_col not in df.columns:
            lines.append(f"- 列 {metric_col} 不存在\n")
            continue
        
        df_sorted = df[["Topic", "TopWords", metric_col]].dropna().sort_values(metric_col, ascending=False).head(10)
        
        lines.append("| Rank | Topic（label） | Score | Keywords |")
        lines.append("|------|---------------|-------|----------|")
        
        for rank, (_, row) in enumerate(df_sorted.iterrows(), 1):
            topic_id = int(row["Topic"])
            score_val = row[metric_col]
            
            # 格式化得分
            fmt = metric_info['format']
            if fmt == ".4f":
                score_str = f"{score_val:.4f}"
            elif fmt == ".1f":
                score_str = f"{score_val:.1f}"
            elif fmt == ".2%":
                score_str = f"{score_val:.2%}"
            elif fmt == ".2f":
                score_str = f"{score_val:.2f}"
            else:
                score_str = f"{score_val:.0f}"
            
            # 获取关键词并生成专业短语描述（优先使用全局唯一映射）
            if topic_phrase_map and topic_id in topic_phrase_map:
                topic_phrase = topic_phrase_map[topic_id]
            else:
                keywords = _extract_keywords_from_topwords(row.get("TopWords", ""), limit=12)
                topic_phrase = _generate_topic_phrase(keywords)
            
            kw = ", ".join(_extract_keywords_from_topwords(row.get("TopWords", ""), limit=8))
            lines.append(f"| {rank} | {topic_phrase} | {score_str} | {kw} |")
        
        lines.append("")
    
    return lines


def _method_top_topics_section(base_dir: Path, method: str, best_mc: int, topic_phrase_map: Optional[Dict[int, str]] = None) -> List[str]:
    """生成该方法的主要主题概览"""
    lines: List[str] = []
    lines.append("## 主要研究主题（该方法特有发现）\n")
    
    files = _pick_best_files(base_dir, method, best_mc)
    ti = files["topic_info"]
    
    if not ti.exists():
        lines.append("- 主题数据暂无\n")
        return lines
    
    try:
        dft = pd.read_csv(ti)
        
        # 过滤有效主题（Topic >= 0）
        valid_topics = dft[dft["Topic"] >= 0].copy()
        
        if valid_topics.empty:
            lines.append("- 未找到有效主题\n")
            return lines
        
        # 按 Count 排序，取 Top 10
        top_topics = valid_topics.sort_values("Count", ascending=False).head(10)
        
        lines.append(f"- Total **{len(valid_topics)}** valid topics identified")
        lines.append(f"- Top 10 topics by document count:\n")
        
        for idx, (_, row) in enumerate(top_topics.iterrows(), 1):
            topic_id = int(row["Topic"])
            count = int(row["Count"])
            
            # 提取关键词
            keywords = _extract_keywords_from_representation(row.get("Representation", ""), limit=8)

            # 生成专业短语描述（优先使用全局唯一映射）
            if topic_phrase_map and topic_id in topic_phrase_map:
                topic_phrase = topic_phrase_map[topic_id]
            else:
                topic_phrase = _generate_topic_phrase(keywords)
            
            # 关键词显示（英文，前6个）
            en_keywords = ", ".join(keywords[:6])
            
            lines.append(f"{idx}. **{topic_phrase}**")
            lines.append(f"   - Documents: {count}")
            lines.append(f"   - Keywords: {en_keywords}\n")
        
        lines.append("")
    
    except Exception as e:
        lines.append(f"- 解析主题时出错: {str(e)[:100]}\n")
    
    return lines


def _convert_md_to_html(md_content: str, title: str) -> str:
    """将 Markdown 转换为漂亮的 HTML 页面"""
    # 基础 HTML 转换（不依赖 markdown 库）
    html_body = md_content
    
    # 标题转换
    html_body = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html_body, flags=re.MULTILINE)
    html_body = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html_body, flags=re.MULTILINE)
    html_body = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html_body, flags=re.MULTILINE)
    
    # 代码块
    html_body = re.sub(r'```(\w+)?\n(.*?)```', r'<pre><code>\2</code></pre>', html_body, flags=re.DOTALL)
    
    # 列表项
    html_body = re.sub(r'^- (.+)$', r'<li>\1</li>', html_body, flags=re.MULTILINE)
    
    # 图片
    html_body = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', r'<img src="\2" alt="\1" style="max-width:100%;">', html_body)
    
    # 粗体
    html_body = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html_body)
    
    # 换行
    html_body = html_body.replace('\n\n', '</p><p>')
    
    # 如果有 markdown 库，使用它做更好的转换
    if HAS_MARKDOWN:
        try:
            html_body = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])
        except:
            pass  # 使用基础转换
    
    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <!-- KaTeX for math rendering -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js" onload="renderMathInElement(document.body, {{delimiters: [{{left: '$$', right: '$$', display: true}}, {{left: '$', right: '$', display: false}}]}});"></script>
    <style>
        :root {{
            --primary-color: #2563eb;
            --bg-color: #f8fafc;
            --card-bg: #ffffff;
            --text-color: #1e293b;
            --border-color: #e2e8f0;
        }}
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            line-height: 1.8;
            color: var(--text-color);
            background: var(--bg-color);
            padding: 2rem;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: var(--card-bg);
            padding: 3rem;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
        }}
        h1 {{
            color: var(--primary-color);
            font-size: 2rem;
            margin-bottom: 1.5rem;
            padding-bottom: 1rem;
            border-bottom: 3px solid var(--primary-color);
        }}
        h2 {{
            color: var(--text-color);
            font-size: 1.4rem;
            margin: 2rem 0 1rem 0;
            padding: 0.5rem 0;
            border-left: 4px solid var(--primary-color);
            padding-left: 1rem;
            background: linear-gradient(90deg, #eff6ff 0%, transparent 100%);
        }}
        h3 {{
            font-size: 1.1rem;
            margin: 1.5rem 0 0.75rem 0;
            color: #475569;
        }}
        p {{
            margin: 0.75rem 0;
        }}
        ul, ol {{
            margin: 1rem 0;
            padding-left: 1.5rem;
        }}
        li {{
            margin: 0.5rem 0;
        }}
        pre {{
            background: #1e293b;
            color: #e2e8f0;
            padding: 1.5rem;
            border-radius: 8px;
            overflow-x: auto;
            margin: 1rem 0;
            font-size: 0.9rem;
        }}
        code {{
            font-family: "Fira Code", "Monaco", "Consolas", monospace;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1.5rem 0;
            font-size: 0.95rem;
        }}
        th, td {{
            border: 1px solid var(--border-color);
            padding: 0.75rem 1rem;
            text-align: left;
        }}
        th {{
            background: #f1f5f9;
            font-weight: 600;
        }}
        tr:nth-child(even) {{
            background: #f8fafc;
        }}
        tr:hover {{
            background: #eff6ff;
        }}
        img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            margin: 1rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        strong {{
            color: var(--primary-color);
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1.5rem 0;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
            padding: 1.25rem;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 1.75rem;
            font-weight: 700;
            color: var(--primary-color);
        }}
        .stat-label {{
            font-size: 0.85rem;
            color: #64748b;
            margin-top: 0.25rem;
        }}
        .footer {{
            margin-top: 3rem;
            padding-top: 1.5rem;
            border-top: 1px solid var(--border-color);
            text-align: center;
            color: #94a3b8;
            font-size: 0.85rem;
        }}
        @media print {{
            body {{
                background: white;
                padding: 0;
            }}
            .container {{
                box-shadow: none;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        {html_body}
        <div class="footer">
            Generated by Topic Modeling Pipeline | {time.strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>'''
    return html


def build_report_for_method(
    base_dir: Path,
    method: str,
    *,
    step08_ctx: Optional[Dict[str, Any]] = None,
    refresh_viz: bool = True,
    refresh_reference_mtime: float = 0.0,
) -> Dict[str, Any]:
    # load manifests
    sw_manifest = _safe_load_json(base_dir / "05_stopwords" / "stopwords_manifest.json")
    denoise_manifest = _safe_load_json(base_dir / "06_denoised_data" / "denoise_manifest.json")
    best_map = _safe_load_json(base_dir / "08_model_selection" / "best_mc_by_method.json")

    if best_map is None or method not in best_map:
        return {"method": method, "status": "missing_best_mc"}

    best_mc = int(best_map[method]["mc"])

    # 保证 Step09 不会嵌入老图：必要时先刷新可视化
    if refresh_viz:
        _maybe_refresh_step09(base_dir, method, reference_mtime=refresh_reference_mtime)

    # 为当前方法构建“主题短语唯一映射”（同一报告内不允许不同 Topic 显示相同短语）
    topic_phrase_map: Dict[int, str] = {}
    try:
        files = _pick_best_files(base_dir, method, best_mc)
        ti = files.get("topic_info")
        if ti and Path(ti).exists():
            dft = pd.read_csv(Path(ti))
            topic_phrase_map = _build_unique_topic_phrase_map(dft)
    except Exception:
        topic_phrase_map = {}

    out_dir = base_dir / "10_report" / method.upper()
    out_dir.mkdir(parents=True, exist_ok=True)

    md: List[str] = []
    md.append(f"# {get_project_name()} 主题建模研究报告（{method.upper()}）\n")
    md.append(f"- Project Prefix: {PROJECT_PREFIX}")
    md.append(f"- Search Keyword: {SEARCH_KEYWORD}")
    md.append(f"- 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 方法论章节（总览 + 公式/符号/复现性；不在这里放“本方法结果”，避免打乱步骤顺序）
    md.extend(_methodology_section(base_dir=base_dir, method=method, best_mc=best_mc))

    # Step04 log
    fl = _read_filter_log(base_dir)
    md.append("## Step 04 类型过滤摘要\n")
    if fl:
        md.append("```text")
        md.append(fl.strip()[:4000])
        md.append("```\n")
    else:
        md.append("- 未找到过滤日志（可忽略）\n")

    md.extend(_method_stopword_section(method, sw_manifest))
    md.extend(_method_denoise_section(method, denoise_manifest))
    md.extend(_method_topic_model_section(base_dir, method, best_mc, topic_phrase_map=topic_phrase_map))

    # Step08 section（放在 Step07 之后，保证顺序；仅此处给出本方法选择结果）
    md.append("## Step 08：mc 选择结果（本方法）\n")
    md.append(f"- selected best_mc = {best_mc}")
    md.append(f"- C_v = {best_map[method].get('cv')}")
    if best_map[method].get("noise_ratio") is not None:
        md.append(f"- noise = {best_map[method].get('noise_ratio')}")
    if best_map[method].get("selection_note"):
        md.append(f"- selection_note = {best_map[method].get('selection_note')}")

    # 如果存在 Step08 全量记录（cv_scores_full.json），补充可行集/帕累托集/规则
    if step08_ctx and isinstance(step08_ctx, dict):
        scores_full = step08_ctx.get("scores_full") or {}
        methods_blob = (scores_full.get("methods") or {}) if isinstance(scores_full, dict) else {}
        mrec = methods_blob.get(method) if isinstance(methods_blob, dict) else None
        if isinstance(mrec, dict):
            sel = mrec.get("selection_details")
            md.append(f"- evaluated_mcs = {mrec.get('evaluated_mcs','-')}")
            if isinstance(sel, dict) and sel.get("noise_ref") is not None:
                try:
                    md.append(f"- noise_ref r = {float(sel.get('noise_ref')):.2%} ({sel.get('noise_ref_label','-')})")
                except Exception:
                    md.append(f"- noise_ref r = {sel.get('noise_ref')} ({sel.get('noise_ref_label','-')})")
            if isinstance(sel, dict) and sel.get("feasible_mcs"):
                md.append(f"- feasible_mcs F(r) = {sel.get('feasible_mcs')}")
            if isinstance(sel, dict) and sel.get("pareto_mcs"):
                md.append(f"- pareto_mcs = {sel.get('pareto_mcs')}")
            if isinstance(sel, dict) and sel.get("rule"):
                md.append(f"- decision_rule = {sel.get('rule')}")
    md.append("")

    # Step09 可视化（放在 Step08 之后）
    md.extend(_method_viz_section(base_dir, method))

    # —— 结果/分析章节（放到步骤之后，符合“先方法后结果”的写作习惯）——
    md.extend(_method_top_topics_section(base_dir, method, best_mc, topic_phrase_map=topic_phrase_map))
    md.extend(_method_frontier_metrics_section(base_dir, method, best_mc, topic_phrase_map=topic_phrase_map))
    md.extend(_key_findings_section(base_dir, method, best_mc, topic_phrase_map=topic_phrase_map))

    md.extend(_citations_section())

    md_content = "\n".join(md)
    
    # 保存 Markdown
    out_file = out_dir / f"{PROJECT_PREFIX}_{method}_report.md"
    out_file.write_text(md_content, encoding="utf-8")
    
    # 保存 HTML
    html_file = out_dir / f"{PROJECT_PREFIX}_{method}_report.html"
    html_content = _convert_md_to_html(md_content, f"{get_project_name()} 主题建模研究报告（{method.upper()}）")
    html_file.write_text(html_content, encoding="utf-8")

    return {"method": method, "status": "ok", "report": str(out_file), "html": str(html_file), "best_mc": best_mc}


def _ensure_pandoc(auto_download: bool) -> None:
    if not HAS_PYPANDOC:
        raise RuntimeError("missing_pypandoc")
    try:
        _ = pypandoc.get_pandoc_version()
        return
    except OSError:
        if not auto_download:
            raise
    pypandoc.download_pandoc()
    _ = pypandoc.get_pandoc_version()


def _convert_md_file_to_docx(md_path: Path, docx_path: Path, *, auto_download_pandoc: bool) -> None:
    """Convert a Markdown file to DOCX using pandoc.

    Notes:
    - pandoc resolves relative image paths based on the input file location.
    - Uses common markdown extensions and tex_math_dollars for $...$ / $$...$$.
    """
    _ensure_pandoc(auto_download_pandoc)
    docx_path.parent.mkdir(parents=True, exist_ok=True)
    # Resource path is critical on Windows: pandoc resolves images relative to its resource path
    # (often the current working directory), not necessarily the markdown file directory.
    # We include both the markdown directory and the repo root to make paths like
    # ../../09_visualization/... resolvable.
    resource_path = ";".join([str(md_path.parent), str(Path(__file__).resolve().parent)])

    extra_args = [
        "--quiet",
        "--from",
        "markdown+pipe_tables+grid_tables+fenced_code_blocks+backtick_code_blocks+tex_math_dollars",
        "--resource-path",
        resource_path,
    ]

    # pandoc 在 Windows 上的 stdout/stderr 可能使用本地代码页（如 cp936），
    # pypandoc 默认按 utf-8 解码会抛出 "Pandoc output was not utf-8."。
    # 这里使用系统首选编码解码，仅影响日志解码，不影响生成的 docx。
    pandoc_encoding = locale.getpreferredencoding(False) or "utf-8"
    pypandoc.convert_file(
        str(md_path),
        to="docx",
        format="md",
        outputfile=str(docx_path),
        extra_args=extra_args,
        encoding=pandoc_encoding,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Step10: 报告生成")
    parser.add_argument(
        "--base_dir",
        default=None,
        help="工作目录（包含 07_topic_models/08_model_selection/10_report 等）。默认使用脚本所在目录。",
    )
    parser.add_argument("--only", help="只跑指定方法（不区分大小写：baseline/A/B/C/AB/ABC）")
    parser.add_argument("--docx", action="store_true", help="同时生成 Word 文档（.docx，需 pandoc）")
    parser.add_argument(
        "--no-refresh-viz",
        action="store_true",
        help="不自动刷新 Step09 可视化（默认若检测到缺失/过期会自动重跑 Step09，避免嵌入老图）",
    )
    parser.add_argument(
        "--no-download-pandoc",
        action="store_true",
        help="pandoc 缺失时不自动下载（默认会自动下载 pandoc）",
    )
    args = parser.parse_args()

    def _resolve_base_dir() -> Path:
        if args.base_dir:
            return Path(args.base_dir).resolve()
        # 若存在主流程指针，则默认使用它
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
    if args.only:
        only_raw = str(args.only).strip()
        # 兼容：用户习惯用 C 指代当前主流程的 VPD
        if only_raw.upper() in ("VPD", "C"):
            only_norm = "VPD"
        elif only_raw.lower() == "baseline":
            only_norm = "baseline"
        else:
            only_norm = only_raw.upper() if only_raw.upper() in ("A", "B", "AB", "ABC") else only_raw.lower()

        if only_norm not in ALL_METHODS:
            print(f"--only 参数无效: {only_raw}")
            print("可选: baseline/VPD（兼容别名 C）")
            return 2
        methods = [only_norm]
    else:
        methods = ALL_METHODS

    print("=" * 80)
    print(f"Step 10 报告生成 - {get_project_name()} ({PROJECT_PREFIX})")
    print("=" * 80)

    step08_ctx = _load_step08_context(base_dir)
    refresh_ref_mtime = float(max(step08_ctx.get("best_mtime", 0.0), step08_ctx.get("scores_mtime", 0.0)))

    ok_any = False
    for m in methods:
        print(f"\n→ 生成 {m} 报告...", end="", flush=True)
        r = build_report_for_method(
            base_dir,
            m,
            step08_ctx=step08_ctx,
            refresh_viz=not args.no_refresh_viz,
            refresh_reference_mtime=refresh_ref_mtime,
        )
        if r.get("status") != "ok":
            print(f"✗ ({r.get('status')})")
            continue
        print("✓")
        print(f"  best_mc={r.get('best_mc')}  MD: {r.get('report')}")
        print(f"               HTML: {r.get('html')}")

        if args.docx:
            md_path = Path(r.get("report"))
            docx_path = md_path.with_suffix(".docx")
            try:
                _convert_md_file_to_docx(md_path, docx_path, auto_download_pandoc=not args.no_download_pandoc)
                print(f"               DOCX: {docx_path.as_posix()}")
            except Exception as exc:
                # 不让 docx 失败影响主流程（MD/HTML 已完成）
                reason = str(exc)
                if isinstance(exc, RuntimeError) and str(exc) == "missing_pypandoc":
                    reason = "缺少依赖 pypandoc（pip install pypandoc）"
                print(f"               DOCX: ✗ ({reason})")

        ok_any = True

    print("\n" + "=" * 80)
    print("Step 10 完成")
    print("输出目录: 10_report/")
    print("=" * 80)

    return 0 if ok_any else 1


if __name__ == "__main__":
    raise SystemExit(main())
