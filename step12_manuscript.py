#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""step12_manuscript.py

生成“文章格式”的应用型论文 Word（DOCX）：只呈现一个最终方法在幽门螺杆菌领域的应用方法与结果。
注意：
- 文稿中不展示本机绝对路径（如 D:/...），避免出现“电脑内部地址”。
- 可选保留补充材料的相对链接（不暴露绝对路径）。

输出：
- word_报告/{PROJECT_PREFIX}_manuscript.md
- word_报告/{PROJECT_PREFIX}_manuscript.docx（需要 pypandoc + pandoc；可自动下载）

用法：
- python step12_manuscript.py
- python step12_manuscript.py --no-docx
- python step12_manuscript.py --only ABC
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import pypandoc  # type: ignore
    HAS_PYPANDOC = True
except Exception:
    HAS_PYPANDOC = False

try:
    from config import (
        PROJECT_PREFIX,
        SEARCH_KEYWORD,
        get_project_name,
        START_DATE,
        END_DATE,
        DATA_SOURCE,
        EXCLUDE_PUB_TYPES,
        MODEL_NAME,
        MIN_TOPIC_SIZE,
        EMBEDDING_BACKEND,
    )
except Exception:
    print("请确保 config.py 存在且配置正确")
    raise


ALL_METHODS = ["baseline", "VPD", "A", "B", "C", "AB", "ABC"]


def _resolve_base_dir(arg: str | None) -> Path:
    if arg:
        return Path(arg).resolve()
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


def _safe_load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _get_git_commit_hash(base_dir: Path) -> str:
    try:
        out = subprocess.check_output(["git", "-C", str(base_dir), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        s = out.decode("utf-8", errors="replace").strip()
        return s if s else "-"
    except Exception:
        return "-"


def _count_csv_rows_fast(path: Path) -> Optional[int]:
    try:
        with path.open("rb") as f:
            n = 0
            for _ in f:
                n += 1
        return max(0, n - 1)
    except Exception:
        return None


def _rel_from_word_dir(base_dir: Path, target: Path) -> str:
    word_dir = base_dir / "word_报告"
    try:
        return target.resolve().relative_to(word_dir.resolve()).as_posix()
    except Exception:
        # fallback: use a simple relative path from base_dir, still usually resolvable via resource-path
        try:
            return target.resolve().relative_to(base_dir.resolve()).as_posix()
        except Exception:
            return target.as_posix()


def _mklink(text: str, rel_path: str) -> str:
    return f"[{text}]({rel_path})"


def _fmt_pct(x: Any) -> str:
    try:
        return f"{float(x) * 100:.2f}%"
    except Exception:
        return "-"


def _fmt_float(x: Any, nd: int = 4) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "-"


def _load_step08_scores(base_dir: Path) -> Dict[str, Any]:
    p = base_dir / "08_model_selection" / "cv_scores_full.json"
    return _safe_load_json(p) or {}


def _load_best_mc_by_method(base_dir: Path) -> Dict[str, Any]:
    p = base_dir / "08_model_selection" / "best_mc_by_method.json"
    return _safe_load_json(p) or {}


def _load_any_review_manifest(base_dir: Path) -> Dict[str, Any]:
    """Load a representative Step07 review_manifest.json (best-effort).

    We use it to populate paper-style parameter descriptions without exposing file paths.
    """
    for m in ("VPD", "ABC", "AB", "C", "B", "A", "BASELINE"):
        p = base_dir / "07_topic_models" / m / "review_manifest.json"
        rec = _safe_load_json(p)
        if isinstance(rec, dict):
            return rec
    return {}


def _method_links_section(base_dir: Path, method: str) -> List[str]:
    lines: List[str] = []
    lines.append("## 补充材料（Supplementary Materials）\n")
    lines.append("说明：以下链接为相对路径，不展示本机绝对路径。\n")
    lines.append("| 内容 | 链接 |")
    lines.append("|---|---|")

    m_dir = method.upper() if method != "baseline" else "BASELINE"
    viz = base_dir / "09_visualization" / m_dir / "viz_report.html"
    rep_html = base_dir / "10_report" / m_dir / f"{PROJECT_PREFIX}_{method}_report.html"
    rep_docx = base_dir / "10_report" / m_dir / f"{PROJECT_PREFIX}_{method}_report.docx"

    if viz.exists():
        lines.append(f"| 图表解释页（HTML） | {_mklink('viz_report.html', '../' + _rel_from_word_dir(base_dir, viz))} |")
    if rep_html.exists():
        lines.append(f"| 详细报告（HTML） | {_mklink('report.html', '../' + _rel_from_word_dir(base_dir, rep_html))} |")
    if rep_docx.exists():
        lines.append(f"| 详细报告（DOCX） | {_mklink('report.docx', '../' + _rel_from_word_dir(base_dir, rep_docx))} |")

    lines.append("")
    return lines


def _dataset_overview(base_dir: Path) -> List[str]:
    lines: List[str] = []
    lines.append("## 数据概况\n")
    baseline_file = base_dir / "06_denoised_data" / f"{PROJECT_PREFIX}_topic_modeling_baseline.csv"
    n_docs = _count_csv_rows_fast(baseline_file)

    lines.append(f"- 数据源：{DATA_SOURCE}")
    lines.append(f"- 检索词：{SEARCH_KEYWORD}")
    lines.append(f"- 时间范围：{START_DATE} – {END_DATE}")
    lines.append(f"- 语料规模：{n_docs if isinstance(n_docs, int) else '-'} 篇（以本次导出为准）")
    lines.append("")
    return lines


def _methods_section(*, method: str, best_mc: int) -> List[str]:
    rm = _load_any_review_manifest(Path(__file__).resolve().parent)
    emb_model = rm.get("embedding_model") if isinstance(rm, dict) else None
    ngram_range = rm.get("ngram_range") if isinstance(rm, dict) else None
    reduce_outliers_cfg = rm.get("reduce_outliers_cfg") if isinstance(rm, dict) else None
    emb_model = emb_model or MODEL_NAME

    exclude_types = ", ".join(EXCLUDE_PUB_TYPES) if isinstance(EXCLUDE_PUB_TYPES, list) else str(EXCLUDE_PUB_TYPES)
    method_label = method.upper() if method != "baseline" else "BASELINE"

    return [
        "## 方法（Materials and Methods）\n",
        "### 文献检索与纳入排除\n",
        f"以 \"{SEARCH_KEYWORD}\" 为检索词，从 {DATA_SOURCE} 获取文献记录，时间范围为 {START_DATE}–{END_DATE}。\n",
        f"为降低非研究性文本对主题建模的干扰，排除文献类型包括：{exclude_types}。\n",
        "对标题与摘要等字段进行清洗，并构建统一的可建模文本字段用于主题建模。\n",
        "### 文本去噪与术语保护\n",
        "为提升主题可解释性并降低通用词对聚类的干扰，我们对文本进行领域适配的去噪处理：构造领域停用词并引入保护词机制，以避免核心医学术语被误删。\n",
        "### 主题建模与聚类\n",
        "使用 BERTopic 进行主题建模（Grootendorst, 2022）。首先将文献文本编码为语义向量，再通过 UMAP 进行降维（McInnes et al., 2018），并使用 HDBSCAN 进行密度聚类（McInnes et al., 2017）。\n",
        f"嵌入模型/后端：{emb_model}（backend={EMBEDDING_BACKEND}）。\n",
        f"主题最小规模阈值：MIN_TOPIC_SIZE={MIN_TOPIC_SIZE}。\n",
        f"n-gram 范围：{ngram_range if ngram_range is not None else '[1,1]'}。\n",
        "HDBSCAN 的关键超参数为 min_cluster_size（记为 mc）。聚类中被标记为 Topic=-1 的文献视为噪音点。\n",
        (f"离群点处理配置（如启用）：{json.dumps(reduce_outliers_cfg, ensure_ascii=False)}\n" if reduce_outliers_cfg else ""),
        "### 关键参数设定\n",
        f"本文采用单一最终模型进行分析，方法标识为 {method_label}；HDBSCAN 的 min_cluster_size 设定为 {best_mc}，用于控制簇的最小规模并获得稳定的主题划分。\n",
        "### 评价与可视化\n",
        "为表征主题可解释性与聚类可靠性，我们报告主题规模、近年热度与时间新颖性等指标，并将 Topic=-1 的文献占比作为噪音比例参考。\n",
        "基于最终模型输出主题统计与图表，并生成可视化解释页与详细报告（见补充材料），用于复核与扩展。\n",
        "",
    ]


def _reproducibility_section(base_dir: Path, *, method: str, best_mc: int) -> List[str]:
    lines: List[str] = []
    lines.append("## 复现性（Reproducibility）\n")
    lines.append("| 项目 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| 生成时间 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |")
    lines.append(f"| OS / Platform | {platform.platform()} |")
    lines.append(f"| Python | {platform.python_version()} |")
    lines.append(f"| Git commit hash | {_get_git_commit_hash(base_dir)} |")

    method_label = method.upper() if method != "baseline" else "BASELINE"
    lines.append(f"| 最终模型方法标识 | {method_label} |")
    lines.append(f"| min_cluster_size（mc） | {best_mc} |")
    lines.append("")
    return lines


def _references_section() -> List[str]:
    return [
        "## 参考文献\n",
        "- Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. arXiv:2203.05794.",
        "- McInnes, L., Healy, J., & Astels, S. (2017). hdbscan: Hierarchical density based clustering. JOSS, 2(11), 205.",
        "- McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection. arXiv:1802.03426.",
        "",
    ]


def _topic_phrase_from_topwords(topwords: str, *, limit: int = 6) -> str:
    try:
        parts = [p.strip() for p in str(topwords).split(";") if p.strip()]
        parts = parts[:limit]
        return "/".join(parts) if parts else "(topic)"
    except Exception:
        return "(topic)"


def build_application_md(base_dir: Path, *, method: str) -> str:
    best_map = _load_best_mc_by_method(base_dir)
    mrec = best_map.get(method) if isinstance(best_map, dict) else None
    best_mc = int(mrec.get("mc")) if isinstance(mrec, dict) and mrec.get("mc") is not None else None
    if best_mc is None:
        raise RuntimeError(f"missing_best_mc_for_{method}")

    title = f"{get_project_name()}（{SEARCH_KEYWORD}）文献主题建模与研究前沿识别：应用研究"

    md: List[str] = []
    md.append(f"# {title}\n")
    md.append("作者：（待补）\n")
    md.append("单位：（待补）\n")
    md.append("")

    # Abstract（保守：不做过度结论，只说明做了什么与产物在哪里）
    md.append("## 摘要\n")
    baseline_file = base_dir / "06_denoised_data" / f"{PROJECT_PREFIX}_topic_modeling_baseline.csv"
    n_docs = _count_csv_rows_fast(baseline_file)
    md.append(
        f"我们对 {get_project_name()}（检索词：{SEARCH_KEYWORD}）相关文献进行主题建模与研究前沿识别，"
        f"以揭示该领域的核心主题结构与近年研究热点。主题建模采用 BERTopic（UMAP 降维 + HDBSCAN 聚类）并结合领域适配的文本去噪与术语保护。"
        f"本次分析纳入文献约 {n_docs if isinstance(n_docs, int) else '-'} 篇；最终模型参数设定为 min_cluster_size={best_mc}。"
        f"我们报告主题规模、近年热度与时间新颖性等指标，并据此总结该领域近期研究前沿与热点方向。\n"
    )
    md.append("关键词：幽门螺杆菌；主题建模；BERTopic；研究前沿；研究热点\n")
    md.append("")

    md.append("## 引言\n")
    md.append(
        "幽门螺杆菌（Helicobacter pylori）相关研究跨越基础机制、诊疗策略与人群流行病学等多个方向，文献规模大、主题演化快。"
        "为获得可解释的研究全景，我们使用主题建模刻画主题结构，并基于主题规模、近年热度与时间新颖性等指标识别该领域的研究前沿与热点方向。\n"
    )
    md.append("")

    md.extend(_dataset_overview(base_dir))
    md.extend(_methods_section(method=method, best_mc=best_mc))

    # Results (single method)
    md.append("## 结果（Results）\n")
    md.append(f"最终模型：min_cluster_size={best_mc}。\n")

    m_dir = method.upper() if method != "baseline" else "BASELINE"
    topic_info = base_dir / "07_topic_models" / m_dir / f"{PROJECT_PREFIX}_mc{best_mc}_topic_info.csv"
    doc_map = base_dir / "07_topic_models" / m_dir / f"{PROJECT_PREFIX}_mc{best_mc}_doc_topic_mapping.csv"
    frontier = base_dir / "07_topic_models" / m_dir / f"{PROJECT_PREFIX}_mc{best_mc}_frontier_indicators.csv"
    viz_dir = base_dir / "09_visualization" / m_dir
    upgrade_dir = base_dir / "12_top_journal_upgrade" / m_dir

    try:
        import pandas as pd

        if doc_map.exists():
            dm = pd.read_csv(doc_map)
            noise_ratio = float((dm["Topic"] == -1).mean()) if "Topic" in dm.columns and len(dm) else None
            md.append(f"- 噪音比例（Topic=-1 文献占比）：{_fmt_pct(noise_ratio) if noise_ratio is not None else '-'}\n")

        if topic_info.exists():
            ti = pd.read_csv(topic_info)
            if "Topic" in ti.columns:
                num_topics = int((ti["Topic"] >= 0).sum())
                md.append(f"- 有效主题数：{num_topics}\n")

            # Top themes by size
            if "Count" in ti.columns:
                md.append("### 核心主题（按主题规模）\n")
                core = ti[ti.get("Topic", 0) >= 0].sort_values("Count", ascending=False).head(10)
                md.append("| 主题 | 文献数 | 代表词（TopWords） |")
                md.append("|---|---:|---|")
                for _, r in core.iterrows():
                    label = _topic_phrase_from_topwords(r.get("TopWords", ""))
                    md.append(f"| {label} | {int(r.get('Count',0))} | {str(r.get('TopWords',''))[:120]} |")
                md.append("")

        if frontier.exists():
            fr = pd.read_csv(frontier)

            def _table(title: str, df: 'pd.DataFrame') -> None:
                md.append(f"### {title}\n")
                md.append("| 热点方向 | 规模(Strength) | 新颖性(AvgYear) | 热度(RecentRatio) | 复合指数 |")
                md.append("|---|---:|---:|---:|---:|")
                for _, r in df.iterrows():
                    label = _topic_phrase_from_topwords(r.get("TopWords", ""))
                    md.append(
                        "| "
                        + label
                        + " | "
                        + _fmt_float(r.get("Strength"), 4)
                        + " | "
                        + _fmt_float(r.get("Novelty_AvgYear"), 2)
                        + " | "
                        + _fmt_float(r.get("Heat_RecentRatio"), 3)
                        + " | "
                        + _fmt_float(r.get("Composite_Index"), 4)
                        + " |"
                    )
                md.append("")

            if "Composite_Index" in fr.columns:
                _table("研究前沿与热点（按复合指数 Composite_Index）", fr.sort_values("Composite_Index", ascending=False).head(10))
            if "Novelty_AvgYear" in fr.columns:
                _table("近年新兴方向（按时间新颖性 Novelty_AvgYear）", fr.sort_values("Novelty_AvgYear", ascending=False).head(10))
            if "Heat_RecentRatio" in fr.columns:
                _table("近期热度方向（按近年占比 Heat_RecentRatio）", fr.sort_values("Heat_RecentRatio", ascending=False).head(10))

    except Exception:
        md.append("（结果表格生成失败：请检查 07_topic_models 产物是否齐全。）\n")

    # Figure 1 in Results: Macro overview (UMAP)
    p_fig01 = (upgrade_dir / "fig01_umap_overview.png") if (upgrade_dir / "fig01_umap_overview.png").exists() else (viz_dir / f"{PROJECT_PREFIX}_{method}_topic_distribution.png")
    if p_fig01.exists():
        md.append("### Figure 1. Overview of topic landscape（主题版图总览）\n")
        md.append(
            "该图展示所有有效主题在语义空间的整体分布，并将细粒度主题自动归并为 5–7 个宏观板块（如治疗、诊断、流行病学等）。"
            "宏观板块的颜色与包络用于强调研究“板块结构”，帮助读者从大量细主题快速把握领域格局。\n"
        )
        rel = "../" + _rel_from_word_dir(base_dir, p_fig01)
        md.append(f"![]({rel})")
        md.append("*Figure 1. UMAP overview of topics with macro classes.*\n")
        md.append("")

    # Figure 2 in Results: Macro treemap (synthesis)
    p_fig02 = upgrade_dir / "fig02_macro_treemap.png"
    if p_fig02.exists():
        md.append("### Figure 2. Macro synthesis treemap（宏观归并：矩形树图）\n")
        md.append(
            "为降低 100+ 细主题带来的认知负担，我们将主题归并为少数宏观板块，并以矩形树图呈现层级结构："
            "大方块为宏观板块（面积≈文献量），小方块为具体主题（颜色≈近年热度）。"
            "该图用于支持“研究分为若干大板块，且板块内部存在显著的新老交替/范式转移”的叙事。\n"
        )
        rel = "../" + _rel_from_word_dir(base_dir, p_fig02)
        md.append(f"![]({rel})")
        md.append("*Figure 2. Treemap of macro clusters (size=docs, color=heat).*\n")
        md.append("")

    # Figure 3 in Results: Research Frontiers Bubble
    md.append("### Figure 3. Research frontiers bubble plot（研究前沿气泡图）\n")
    md.append(
        "横轴表示时间新颖性（越右越新），纵轴表示近期热度（越上越热），气泡大小表示主题强度/规模。"
        "通常左下象限为较“经典/老生常谈”的方向，右上象限为“新且热”的潜在明日之星。"
        "在本研究中，Vonoprazan 相关双联治疗与 AI 影像/诊断等方向位于右上区域，提示其在近年呈现更高热度与更新近的研究关注。\n"
    )
    p_fig06 = (upgrade_dir / "fig03_frontier_bubble.png") if (upgrade_dir / "fig03_frontier_bubble.png").exists() else (viz_dir / "fig06_frontier_bubble.png")
    if p_fig06.exists():
        rel = "../" + _rel_from_word_dir(base_dir, p_fig06)
        md.append(f"![]({rel})")
        md.append("*Figure 3. Research frontiers bubble plot (Novelty vs Heat; bubble size = Strength).*\n")
    md.append("")

    # Figure 4 in Results: Temporal evolution (succession)
    p_fig07 = (upgrade_dir / "fig04_temporal_evolution.png") if (upgrade_dir / "fig04_temporal_evolution.png").exists() else (viz_dir / "fig07_temporal_evolution.png")
    if p_fig07.exists():
        md.append("### Figure 4. Temporal evolution（时间演化：新老交替）\n")
        md.append(
            "该图以“少即是多”的方式展示两组关键对比："
            "（1）治疗：传统三联疗法与伏诺拉生双联疗法；"
            "（2）诊断：传统检测路径（如呼气/活检）与 AI 图像识别。"
            "通过对比曲线斜率与拐点，可直观看到新技术/新疗法从边缘走向主流的过程。\n"
        )
        rel = "../" + _rel_from_word_dir(base_dir, p_fig07)
        md.append(f"![]({rel})")
        md.append("*Figure 4. Temporal evolution with two succession contrasts (therapy & diagnosis).*\n")
        md.append("")

    # Figure 5 in Results: Thematic river (macro shift)
    p_river = upgrade_dir / "fig08_macro_river.png"
    if p_river.exists():
        md.append("### Figure 5. Thematic river（主题河流：宏观板块消长）\n")
        md.append(
            "为刻画领域重心在不同时间段的迁移，我们将主题按宏观板块聚合并绘制堆叠面积图。"
            "该图用于观察 2005–2015 与 2016–2025 两个时期，哪些板块扩张/收缩以及是否出现后期突增（如 AI 诊断相关板块）。\n"
        )
        rel = "../" + _rel_from_word_dir(base_dir, p_river)
        md.append(f"![]({rel})")
        md.append("*Figure 5. Thematic river by macro clusters (stacked area).*\n")
        md.append("")

    # Figure 6 in Results: Country comparison (inferred)
    p_country = upgrade_dir / "fig09_country_hotspots.png"
    if p_country.exists():
        md.append("### Figure 6. Geographic signal（地理线索：国家对比）\n")
        md.append(
            "医学顶刊常关注“谁在推动这些热点”。我们基于 PubMed 条目中的作者单位（Affiliation）进行国家信息的启发式推断，"
            "对 Vonoprazan 与 AI 两类热点主题进行国家层面的发文量对比。"
            "需要说明：该统计依赖于单位字段的可用性与解析规则，且通常仅覆盖抽样的 PMID（详见图注/产物参数），因此结果应作为趋势线索而非精确计量。\n"
        )
        rel = "../" + _rel_from_word_dir(base_dir, p_country)
        md.append(f"![]({rel})")
        md.append("*Figure 6. Country comparison for key hotspot topics (inferred from affiliations; sample-limited).*\n")
        md.append("")

    # 附两张示例图（默认 ABC，如果只跑其它方法则不插图）
    md.extend(_method_links_section(base_dir, method))
    md.extend(_reproducibility_section(base_dir, method=method, best_mc=best_mc))

    md.append("## 讨论\n")
    md.append(
        "(1) 主题建模属于无监督学习，结果受语料质量、文本预处理与嵌入表征影响。我们通过领域去噪与术语保护降低通用词干扰，以提升主题可解释性。\n"
        "(2) 本文聚焦幽门螺杆菌领域的主题结构与热点识别；所采用的 BERTopic（结合 UMAP 与 HDBSCAN）提供了可解释的主题表示与聚类框架，可用于对大规模文献进行结构化综述与趋势提炼。\n"
        "(3) 由于不同数据源/字段完整性差异，引用与期刊字段的缺失可能影响部分前沿指标；建议在最终投稿前补充缺失模式与敏感性分析。\n"
    )
    md.append("")

    md.append("## 结论\n")
    md.append(
        "本文对幽门螺杆菌领域文献进行主题结构刻画与研究前沿识别，并基于主题规模、时间新颖性与近年热度等指标总结近期研究热点与前沿方向。"
        "完整主题列表与图表解释已在附录链接的产物中给出，便于复核与扩展。\n"
    )
    md.append("")

    md.extend(_references_section())

    return "\n".join(md)


def _ensure_pandoc(auto_download: bool) -> str:
    if not HAS_PYPANDOC:
        raise RuntimeError("missing_pypandoc")
    try:
        _ = pypandoc.get_pandoc_version()
    except OSError:
        if not auto_download:
            raise
        pypandoc.download_pandoc()
        _ = pypandoc.get_pandoc_version()
    return str(pypandoc.get_pandoc_path())


def _convert_md_file_to_docx_cli(md_path: Path, docx_path: Path, *, auto_download_pandoc: bool) -> None:
    """Convert Markdown to DOCX by calling pandoc CLI directly.

    Reason: On Windows, pandoc stderr may be in local code page; pypandoc may fail with
    'Pandoc output was not utf-8.' when decoding. Using subprocess with stderr/stdout
    redirected avoids decode errors.
    """
    pandoc = _ensure_pandoc(auto_download_pandoc)
    docx_path.parent.mkdir(parents=True, exist_ok=True)

    resource_path = ";".join([str(md_path.parent), str(Path(__file__).resolve().parent)])
    args = [
        pandoc,
        str(md_path),
        "--quiet",
        "--from",
        "markdown+pipe_tables+grid_tables+fenced_code_blocks+backtick_code_blocks+tex_math_dollars",
        "--resource-path",
        resource_path,
        "-o",
        str(docx_path),
    ]
    subprocess.check_call(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _pick_writable_docx_path(preferred: Path) -> Path:
    """Return a writable output path.

    If the preferred path is locked by Word (Permission denied on Windows),
    fall back to a timestamped filename.
    """
    if not preferred.exists():
        return preferred
    # Try to remove existing file; if locked, fall back.
    try:
        preferred.unlink()
        return preferred
    except PermissionError:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return preferred.with_name(preferred.stem + f"_{ts}" + preferred.suffix)
    except Exception:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return preferred.with_name(preferred.stem + f"_{ts}" + preferred.suffix)


def main() -> int:
    parser = argparse.ArgumentParser(description="Step12: 生成整合版文章 Word")
    parser.add_argument(
        "--base_dir",
        default=None,
        help="工作目录（包含 07_topic_models/08_model_selection/09_visualization/10_report 等）。默认使用 MAIN_WORKDIR.txt。",
    )
    parser.add_argument(
        "--final-method",
        default="C",
        help="应用论文使用的最终方法（baseline/VPD；兼容别名 C，默认 C=VPD）",
    )
    parser.add_argument("--no-docx", action="store_true", help="只生成 Markdown，不生成 DOCX")
    parser.add_argument(
        "--no-download-pandoc",
        action="store_true",
        help="pandoc 缺失时不自动下载（默认会自动下载）",
    )
    args = parser.parse_args()

    base_dir = _resolve_base_dir(args.base_dir)

    m_raw = str(args.final_method).strip()
    if m_raw.upper() in ("VPD", "C"):
        method = "VPD"
    else:
        method = m_raw.upper() if m_raw.upper() in ("A", "B", "AB", "ABC") else m_raw.lower()
    if method not in ALL_METHODS:
        print(f"--final-method 参数无效: {m_raw}")
        print("可选: baseline/VPD（兼容别名 C）")
        return 2

    out_dir = base_dir / "word_报告"
    out_dir.mkdir(parents=True, exist_ok=True)

    md_path = out_dir / f"{PROJECT_PREFIX}_application_{method}.md"
    preferred_docx_path = out_dir / f"{PROJECT_PREFIX}_application_{method}.docx"

    md_content = build_application_md(base_dir, method=method)
    md_path.write_text(md_content, encoding="utf-8")
    print(f"MD: {md_path.as_posix()}")

    if args.no_docx:
        return 0

    try:
        docx_path = _pick_writable_docx_path(preferred_docx_path)
        _convert_md_file_to_docx_cli(md_path, docx_path, auto_download_pandoc=not args.no_download_pandoc)
        print(f"DOCX: {docx_path.as_posix()}")
    except Exception as exc:
        msg = str(exc)
        if "permission" in msg.lower() or "denied" in msg.lower():
            msg = "目标 DOCX 可能正在被 Word 占用：请先关闭该 DOCX 后重试，或使用脚本自动生成的带时间戳文件名。"
        print(f"DOCX: ✗ ({msg[:200]})")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
