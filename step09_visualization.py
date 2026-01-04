#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step09_visualization.py
Step 09：可视化（严格使用 Step08 选择的 best mc）

输入：
- 08_model_selection/best_mc_by_method.json
- 07_topic_models/<METHOD>/{PROJECT_PREFIX}_mc{best}_topic_info.csv
- 07_topic_models/<METHOD>/{PROJECT_PREFIX}_mc{best}_frontier_indicators.csv

输出：
- 09_visualization/<METHOD>/*.png
- 09_visualization/<METHOD>/viz_manifest.json

用法：
- python step09_visualization.py
- python step09_visualization.py --only ABC

依赖：matplotlib, seaborn（wordcloud 可选）
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import numpy as np

try:
    from config import PROJECT_PREFIX, get_project_name
except ImportError:
    print("请确保 config.py 存在且配置正确")
    raise


ALL_METHODS = ["baseline", "VPD"]


def _ensure_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Prefer a CJK-capable font on Windows to avoid tofu squares in figures.
    # We pick the first available font from a shortlist and then keep seaborn/matplotlib consistent.
    try:
        from matplotlib import font_manager

        available = {f.name for f in font_manager.fontManager.ttflist}
    except Exception:
        available = set()

    preferred = [
        "Microsoft YaHei",
        "Microsoft JhengHei",
        "SimHei",
        "SimSun",
        "NSimSun",
        "PingFang SC",
        "Noto Sans CJK SC",
        "WenQuanYi Zen Hei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    chosen = next((n for n in preferred if n in available), "DejaVu Sans")

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # Consistent publication-like style
    sns.set_theme(style="whitegrid", font=chosen)
    return plt, sns


def _safe_method_dir(m: str) -> str:
    return m.upper() if m != "baseline" else "BASELINE"


def _resolve_base_dir(arg: Optional[str]) -> Path:
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


def _clean_label(s: str) -> str:
    s = str(s or "").strip()
    s = s.replace("_", " ")
    s = " ".join(s.split())
    return s


def _make_topic_display(topic_info: pd.DataFrame, topic_id: int, *, max_len: int = 48) -> str:
    # 顶刊写作要求：展示语义标签，不暴露 Topic 编号。
    if topic_info is None or topic_info.empty or "Topic" not in topic_info.columns:
        return "Topic"
    row = topic_info.loc[topic_info["Topic"] == topic_id]
    if row.empty:
        return "Topic"
    r0 = row.iloc[0]
    if "Topic_Label" in topic_info.columns and pd.notna(r0.get("Topic_Label")):
        base = _clean_label(str(r0.get("Topic_Label")))
    elif "TopWords" in topic_info.columns and pd.notna(r0.get("TopWords")):
        words = [w.strip() for w in str(r0.get("TopWords")).split(";") if w.strip()]
        base = " · ".join([_clean_label(w) for w in words[:3]])
    elif "Representation" in topic_info.columns and pd.notna(r0.get("Representation")):
        words = [w.strip() for w in str(r0.get("Representation")).strip("[]").replace("'", "").split(",") if w.strip()]
        base = " · ".join([_clean_label(w) for w in words[:3]])
    else:
        base = "Topic"
    if len(base) > max_len:
        base = base[: max_len - 1] + "…"
    return base


def _build_topic_label_maps(topic_info: pd.DataFrame) -> Tuple[Dict[int, str], Dict[int, str]]:
    """Return (short_label, long_label) maps for plotting + tables."""
    short: Dict[int, str] = {}
    long: Dict[int, str] = {}
    if topic_info is None or topic_info.empty or "Topic" not in topic_info.columns:
        return short, long

    valid = topic_info[topic_info["Topic"] >= 0].copy()
    for _, r in valid.iterrows():
        try:
            tid = int(r.get("Topic"))
        except Exception:
            continue
        long_label = _make_topic_display(topic_info, tid, max_len=140)
        short_label = _make_topic_display(topic_info, tid, max_len=42)
        long[tid] = long_label
        short[tid] = short_label

    # Ensure uniqueness for short labels
    inv: Dict[str, List[int]] = {}
    for tid, lab in short.items():
        inv.setdefault(lab, []).append(tid)
    for lab, tids in inv.items():
        if len(tids) <= 1:
            continue
        # 不使用 Topic 编号做消歧：用版本号 v2/v3...（保持不暴露编号）
        for k, tid in enumerate(tids, start=1):
            if k == 1:
                continue
            short[tid] = f"{lab} (v{k})"
    return short, long


def _write_viz_report_html(
    out_dir: Path,
    method: str,
    best_mc: int,
    charts: List[str],
    topic_info: pd.DataFrame,
    frontier: pd.DataFrame,
    doc_map: pd.DataFrame,
) -> str:
    """Generate a lightweight HTML explainer to accompany figures."""

    short_map, long_map = _build_topic_label_maps(topic_info)

    total_docs = int(len(doc_map)) if doc_map is not None else 0
    noise_docs = int((doc_map["Topic"] == -1).sum()) if doc_map is not None and "Topic" in doc_map.columns else 0
    noise_ratio = noise_docs / max(1, total_docs)
    valid_topics = int((topic_info["Topic"] >= 0).sum()) if topic_info is not None and "Topic" in topic_info.columns else 0

    # Hotspot Top8 used in fig07
    top8_hot: List[int] = []
    if frontier is not None and not frontier.empty and {"Topic", "Composite_Index", "Frontier_Type"}.issubset(set(frontier.columns)):
        f = frontier[frontier["Topic"] >= 0].copy()
        hot = f[f["Frontier_Type"].astype(str) == "热点"].copy()
        if hot.empty:
            hot = f
        top8_hot = hot.sort_values("Composite_Index", ascending=False).head(8)["Topic"].astype(int).tolist()

    def img_block(filename: str, title: str, bullets: List[str]) -> str:
        b = "".join([f"<li>{_clean_label(x)}</li>" for x in bullets])
        return (
            f"<h3>{title}</h3>"
            f"<ul>{b}</ul>"
            f"<p><img src=\"{filename}\" style=\"max-width: 100%; height: auto; border:1px solid #eee;\"/></p>"
        )

    # Build concise tables
    hotspot_rows = "".join(
        [
            f"<tr><td>{i+1}</td><td>{_clean_label(short_map.get(t, ''))}</td></tr>"
            for i, t in enumerate(top8_hot)
        ]
    )

    html = f"""<!doctype html>
<html lang=\"zh-CN\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Visualization Report - {PROJECT_PREFIX} - {method.upper()}</title>
  <link rel=\"stylesheet\" href=\"https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.5.1/github-markdown.min.css\" />
  <style>
    body {{ margin: 0; background: #fff; }}
    .markdown-body {{ box-sizing: border-box; min-width: 200px; max-width: 980px; margin: 0 auto; padding: 24px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; }}
    th {{ background: #f6f8fa; }}
  </style>
</head>
<body>
  <article class=\"markdown-body\">
    <h1>可视化解读（{method.upper()}）</h1>
    <ul>
      <li>Project: {PROJECT_PREFIX}</li>
      <li>best_mc: {best_mc}</li>
      <li>有效主题数（Topic≥0）: {valid_topics}</li>
      <li>噪声文献（Topic=-1）: {noise_docs} / {total_docs}（{noise_ratio:.1%}）</li>
      <li>生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}</li>
    </ul>

    <h2>如何读这些图（顶刊写作习惯）</h2>
    <p>原则：先用“宏观结构图”解释整体格局，再用“机制/证据图”解释为什么，最后用“时间演化图”讲趋势与结论。</p>

    <h2>Top8 热点主题（用于 fig07）</h2>
    <p>说明：按 <b>Frontier_Type=热点</b> 且 <b>Composite_Index</b> 最高选出 Top8，用于时间演化折线图。</p>
        <table>
            <thead><tr><th>Rank</th><th>Topic（短标签）</th></tr></thead>
            <tbody>{hotspot_rows}</tbody>
        </table>

    <h2>图表</h2>
    {img_block('fig06_frontier_bubble.png','fig06 前沿气泡图（结构）',[
        '横轴：Novelty_AvgYear（越右越新）',
        '纵轴：Heat_RecentRatio（越上越热）',
        '气泡大小：Strength（越大代表主题规模/影响更强）',
        '颜色：Frontier_Type（热点/新兴/潜在/衰退/一般）',
        '读图要点：右上象限通常代表“新+热”的潜在明日之星；左下象限多为“较老且较冷”的经典或边缘方向。'
    ])}
    {img_block('fig07_temporal_evolution.png','fig07 时间演化（新老交替叙事）',[
        '图分为两栏：治疗（传统三联 vs 伏诺拉生双联）与诊断（传统检测 vs AI 影像）',
        '看斜率：上升更陡=方向加速；平台/回落=热度趋稳或衰退',
        '看拐点：新技术/新疗法从“出现”到“主流”的关键年份',
        '写作建议：正文只讲这两组对比，形成“范式迁移/新老交替”的强叙事。'
    ])}
    {img_block('fig02_frontier_evolution.png','fig02 前沿类型随时间变化（宏观）',[
        '各类 Frontier_Type 的年度文献量变化',
        '用于回答：领域是否正在从“衰退/一般”向“热点/新兴”迁移？',
        '写作建议：用“热点占比上升/新兴增加”来支撑“领域进入活跃期”的判断。'
    ])}
    {img_block('fig05_citation_heatmap.png','fig05 指标热力图（证据）',[
        '按 Composite_Index Top20 的主题展示五指标（归一化）',
        '用于解释：综合指数高的主题究竟来自“热度”还是“影响力/规模/新颖性”。'
    ])}
    {img_block('fig04_topic_size_violin.png','fig04 主题规模分布（鲁棒性）',[
        '展示 topic size 的分布形状（长尾/均匀/是否少数主题过大）',
        '用于说明模型是否被少数大主题主导，辅助讨论可解释性。'
    ])}
    {img_block(f'{PROJECT_PREFIX}_{method}_topic_distribution.png','UMAP 语义空间总览（宏观板块）',[
        '每个点代表一个主题（Topic≥0），点的颜色代表自动归并得到的宏观大类（Macro Classes）',
        '淡色包络用于提示“板块”边界，帮助读者从 100+ 细主题上升到 5–7 个宏观研究板块',
        '写作建议：先用这张图说“领域分为哪些板块”，再用 fig06/fig07 讲“板块内部的新老交替与热点崛起”'
    ])}
  </article>
</body>
</html>
"""

    out_path = out_dir / "viz_report.html"
    out_path.write_text(html, encoding="utf-8")
    return str(out_path)


def _filter_valid_topics(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if "Topic" in df.columns:
        return df[df["Topic"] >= 0].copy()
    return df.copy()


def _plot_fig01_topic_noise_comparison(doc_map: pd.DataFrame, out_dir: Path, method: str):
    plt, _ = _ensure_matplotlib()
    if doc_map is None or doc_map.empty or "Topic" not in doc_map.columns:
        return None

    total = len(doc_map)
    noise = int((doc_map["Topic"] == -1).sum())
    valid = total - noise
    noise_ratio = noise / max(1, total)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(["Valid", "Noise (-1)"], [valid, noise], color=["#4C78A8", "#F58518"])
    ax.set_title(f"Topic vs Noise ({method.upper()})  noise={noise_ratio:.1%}")
    ax.set_ylabel("Documents")
    for i, v in enumerate([valid, noise]):
        ax.text(i, v + max(1, total * 0.01), str(v), ha="center", va="bottom", fontsize=10)
    p = out_dir / "fig01_topic_noise_comparison.png"
    fig.tight_layout()
    fig.savefig(p, dpi=300)
    plt.close(fig)
    return str(p)


def _plot_fig02_frontier_evolution(doc_map: pd.DataFrame, frontier: pd.DataFrame, out_dir: Path, method: str):
    plt, sns = _ensure_matplotlib()
    if doc_map is None or doc_map.empty or frontier is None or frontier.empty:
        return None
    if not {"Topic", "Year"}.issubset(set(doc_map.columns)):
        return None
    if "Frontier_Type" not in frontier.columns or "Topic" not in frontier.columns:
        return None

    m = doc_map.copy()
    m["Year"] = pd.to_numeric(m["Year"], errors="coerce").fillna(0).astype(int)
    m = m[m["Year"] > 0].copy()
    if m.empty:
        return None

    tmap = frontier[["Topic", "Frontier_Type"]].copy()
    tmap = tmap[tmap["Topic"] >= 0]
    merged = m.merge(tmap, on="Topic", how="left")
    merged["Frontier_Type"] = merged["Frontier_Type"].fillna("未知")
    g = merged.groupby(["Year", "Frontier_Type"]).size().reset_index(name="Docs")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=g, x="Year", y="Docs", hue="Frontier_Type", marker="o", ax=ax)
    ax.set_title(f"Frontier Evolution by Type ({method.upper()})")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Documents")
    ax.legend(title="Frontier_Type", loc="best")
    fig.tight_layout()
    p = out_dir / "fig02_frontier_evolution.png"
    fig.savefig(p, dpi=300)
    plt.close(fig)
    return str(p)


def _plot_fig03_frontier_network_placeholder(frontier: pd.DataFrame, out_dir: Path, method: str):
    # 旧仓库里有 fig03_frontier_network.png；当前没有对应脚本源码。
    # 这里生成一个“信息面板型”占位图，确保报告不会引用到过期图。
    plt, _ = _ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")
    ax.set_title(f"Frontier Network (placeholder, regenerated) ({method.upper()})")
    n = len(_filter_valid_topics(frontier)) if frontier is not None else 0
    txt = [
        "This figure was regenerated from the latest CSV.",
        "The original network layout script is not present in this repo snapshot.",
        f"Valid topics in frontier table: {n}",
    ]
    ax.text(0.02, 0.8, "\n".join(txt), fontsize=12)
    p = out_dir / "fig03_frontier_network.png"
    fig.tight_layout()
    fig.savefig(p, dpi=300)
    plt.close(fig)
    return str(p)


def _plot_fig04_topic_size_violin(topic_info: pd.DataFrame, out_dir: Path, method: str):
    plt, sns = _ensure_matplotlib()
    if topic_info is None or topic_info.empty or "Count" not in topic_info.columns:
        return None
    data = _filter_valid_topics(topic_info)
    if data.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.violinplot(y=data["Count"], ax=ax, inner="quartile", color="#4C78A8")
    ax.set_title(f"Topic Size Distribution (Violin) ({method.upper()})")
    ax.set_ylabel("Documents per Topic")
    fig.tight_layout()
    p = out_dir / "fig04_topic_size_violin.png"
    fig.savefig(p, dpi=300)
    plt.close(fig)
    return str(p)


def _plot_fig05_citation_heatmap(frontier: pd.DataFrame, out_dir: Path, method: str):
    plt, sns = _ensure_matplotlib()
    if frontier is None or frontier.empty:
        return None
    df = _filter_valid_topics(frontier)
    cols = [c for c in ["Strength", "Novelty_AvgYear", "Heat_RecentRatio", "Avg_Citations", "HighCited_Count"] if c in df.columns]
    if not cols or "Topic" not in df.columns:
        return None

    top = df.sort_values("Composite_Index", ascending=False).head(20) if "Composite_Index" in df.columns else df.head(20)
    mat = top[["Topic"] + cols].copy()
    mat["Topic"] = mat["Topic"].astype(int).astype(str)
    mat = mat.set_index("Topic")
    z = (mat - mat.min(axis=0)) / (mat.max(axis=0) - mat.min(axis=0) + 1e-12)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(z, cmap="YlOrRd", ax=ax, cbar=True)
    ax.set_title(f"Metric Heatmap (Top20 by Composite) ({method.upper()})")
    ax.set_xlabel("Metrics (normalized)")
    ax.set_ylabel("Topic")
    fig.tight_layout()
    p = out_dir / "fig05_citation_heatmap.png"
    fig.savefig(p, dpi=300)
    plt.close(fig)
    return str(p)


def _plot_fig06_frontier_bubble(frontier: pd.DataFrame, out_dir: Path, method: str):
    plt, sns = _ensure_matplotlib()
    if frontier is None or frontier.empty:
        return None
    df = _filter_valid_topics(frontier)
    need = {"Novelty_AvgYear", "Strength", "Heat_RecentRatio"}
    if not need.issubset(set(df.columns)):
        return None

    df = df.copy()
    # Bubble size represents Strength; y-axis uses Heat to form interpretable quadrants:
    # left-bottom=classical/low-heat, right-top=emerging hot stars.
    df["StrengthSize"] = pd.to_numeric(df["Strength"], errors="coerce").fillna(0.0)
    s = df["StrengthSize"].clip(lower=0.0)
    if float(s.max()) > float(s.min()):
        s_norm = (s - s.min()) / (s.max() - s.min() + 1e-12)
    else:
        s_norm = s * 0.0
    df["StrengthSize"] = (s_norm * 1100 + 80).astype(float)

    hue = "Frontier_Type" if "Frontier_Type" in df.columns else None

    # Lancet-like red/blue + neutral grays
    palette = None
    if hue:
        palette = {
            "热点": "#C41E3A",  # Lancet-like red
            "新兴": "#1F77B4",  # medical blue
            "潜在": "#6C757D",
            "衰退": "#ADB5BD",
            "一般": "#9AA0A6",
        }

    fig, ax = plt.subplots(figsize=(10.5, 6.2))

    # Quadrant split by medians (robust, no extra tuning)
    x_cut = float(pd.to_numeric(df["Novelty_AvgYear"], errors="coerce").median())
    y_cut = float(pd.to_numeric(df["Heat_RecentRatio"], errors="coerce").median())

    # Background shading (very light)
    xmin, xmax = float(df["Novelty_AvgYear"].min()), float(df["Novelty_AvgYear"].max())
    ymin, ymax = 0.0, 1.0
    ax.set_xlim(xmin - 0.5, xmax + 0.5)
    ax.set_ylim(ymin, ymax)

    ax.axvspan(x_cut, xmax + 1.0, ymin=0.0, ymax=1.0, facecolor="#E3F2FD", alpha=0.25, zorder=0)
    ax.axvspan(xmin - 1.0, x_cut, ymin=0.0, ymax=1.0, facecolor="#F5F5F5", alpha=0.25, zorder=0)
    ax.axhspan(y_cut, ymax, xmin=0.0, xmax=1.0, facecolor="#FFEBEE", alpha=0.22, zorder=0)

    # Quadrant lines
    ax.axvline(x_cut, linestyle="--", linewidth=1.2, color="#666666", alpha=0.8, zorder=1)
    ax.axhline(y_cut, linestyle="--", linewidth=1.2, color="#666666", alpha=0.8, zorder=1)

    sc = sns.scatterplot(
        data=df,
        x="Novelty_AvgYear",
        y="Heat_RecentRatio",
        size="StrengthSize",
        sizes=(80, 1200),
        hue=hue,
        palette=palette,
        alpha=0.78,
        edgecolor="white",
        linewidth=0.5,
        ax=ax,
        legend=True,
    )

    # Quadrant captions (bilingual, concise)
    ax.text(xmin, ymin + 0.02, "Classical\n经典", fontsize=10, color="#555555", va="bottom")
    ax.text(xmax, ymin + 0.02, "Emerging\n新兴", fontsize=10, color="#1F77B4", va="bottom", ha="right")
    ax.text(xmin, ymax - 0.02, "Declining\n回落", fontsize=10, color="#666666", va="top")
    ax.text(xmax, ymax - 0.02, "Hot\n热点", fontsize=10, color="#C41E3A", va="top", ha="right")

    def _pick_key_rows() -> List[pd.Series]:
        rows: List[pd.Series] = []
        # Keyword-based anchors (do not expose Topic IDs)
        if "TopWords" in df.columns:
            tw = df["TopWords"].astype(str)
            vono = df[tw.str.contains("vonoprazan", case=False, na=False)].copy()
            if not vono.empty:
                rows.append(vono.sort_values(["Heat_RecentRatio", "Novelty_AvgYear"], ascending=False).iloc[0])

            ai = df[tw.str.contains(r"\bai\b|images|learning|model", case=False, na=False, regex=True)].copy()
            if not ai.empty:
                rows.append(ai.sort_values(["Heat_RecentRatio", "Novelty_AvgYear"], ascending=False).iloc[0])
        return rows

    key_rows = _pick_key_rows()
    # Manual annotations (few only -> avoid overlap)
    for i, r in enumerate(key_rows):
        x = float(r.get("Novelty_AvgYear"))
        y = float(r.get("Heat_RecentRatio"))
        topwords = str(r.get("TopWords", ""))
        label = "Key Topic"
        if "vonoprazan" in topwords.lower():
            label = "Vonoprazan\n(dual therapy)"
        elif "ai" in topwords.lower() or "images" in topwords.lower() or "learning" in topwords.lower():
            label = "AI\n(imaging/diagnosis)"

        dx = 0.9 if i % 2 == 0 else -0.9
        dy = 0.08 if i % 2 == 0 else 0.10
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(x + dx, min(0.98, y + dy)),
            textcoords="data",
            fontsize=10,
            ha="left" if dx > 0 else "right",
            va="bottom",
            arrowprops=dict(arrowstyle="-", color="#333333", lw=1.0, shrinkA=0, shrinkB=6, alpha=0.9),
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#999999", alpha=0.92),
            zorder=5,
        )

    title = "Research Frontiers Bubble Plot" if method != "baseline" else "Research Frontiers Bubble Plot"
    ax.set_title(f"{title} ({method.upper()})")
    ax.set_xlabel("Novelty (Average Year)")
    ax.set_ylabel("Heat (Recent Ratio)")

    # Cleaner spines (journal-like)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # De-clutter legend a bit
    leg = ax.get_legend()
    if leg is not None:
        leg.set_title("Frontier Type / Strength")
        for t in leg.get_texts():
            t.set_fontsize(9)

    fig.tight_layout()
    p = out_dir / "fig06_frontier_bubble.png"
    fig.savefig(p, dpi=300)
    plt.close(fig)
    return str(p)


def _plot_fig07_temporal_evolution(
    doc_map: pd.DataFrame,
    frontier: pd.DataFrame,
    topic_info: Optional[pd.DataFrame],
    out_dir: Path,
    method: str,
):
    plt, _ = _ensure_matplotlib()
    if doc_map is None or doc_map.empty or frontier is None or frontier.empty:
        return None
    if not {"Topic", "Year"}.issubset(set(doc_map.columns)):
        return None

    # ---- New: journal-style "succession" comparisons (less is more) ----
    # Select representative topics by keywords (do not expose topic IDs).
    def _topic_text_map() -> Dict[int, str]:
        m: Dict[int, str] = {}
        if topic_info is None or topic_info.empty or "Topic" not in topic_info.columns:
            return m
        valid = topic_info[topic_info["Topic"] >= 0].copy()
        for _, r in valid.iterrows():
            try:
                tid = int(r.get("Topic"))
            except Exception:
                continue
            parts: List[str] = []
            if "Topic_Label" in valid.columns and pd.notna(r.get("Topic_Label")):
                parts.append(str(r.get("Topic_Label")))
            if "TopWords" in valid.columns and pd.notna(r.get("TopWords")):
                parts.append(str(r.get("TopWords")))
            if "Representation" in valid.columns and pd.notna(r.get("Representation")):
                parts.append(str(r.get("Representation")))
            m[tid] = " | ".join([p for p in parts if p])
        return m

    def _pick_topic_by_keywords(keywords: List[str], *, negative: Optional[List[str]] = None) -> Optional[int]:
        txt = _topic_text_map()
        if not txt:
            return None
        negative = negative or []
        cand: List[int] = []
        for tid, s in txt.items():
            ss = s.lower()
            if any(k.lower() in ss for k in keywords) and not any(n.lower() in ss for n in negative):
                cand.append(tid)
        if not cand:
            return None
        # choose the most frequent topic as representative
        try:
            counts = (
                doc_map[doc_map["Topic"].isin(cand)]
                .groupby("Topic")
                .size()
                .sort_values(ascending=False)
            )
            return int(counts.index[0]) if len(counts) else int(cand[0])
        except Exception:
            return int(cand[0])

    therapy_old = _pick_topic_by_keywords(["triple", "sequential", "clarithromycin", "eradication", "regimen"], negative=["vonoprazan"])
    therapy_new = _pick_topic_by_keywords(["vonoprazan", "dual"], negative=[])
    dx_old = _pick_topic_by_keywords(["urea", "breath", "ubt", "biopsy", "endoscopy", "histology", "rapid"], negative=["ai"])
    dx_new = _pick_topic_by_keywords(["ai", "images", "imaging", "deep learning", "intelligence"], negative=[])

    pairs: List[Tuple[str, Optional[int], str, Optional[int]]] = [
        ("Therapy", therapy_old, "Traditional therapy", therapy_new),
        ("Diagnosis", dx_old, "Traditional diagnosis", dx_new),
    ]
    if any(t is None for _, t, _, n in pairs for t in [t, n]):
        # If we can't find all representatives, fall back to the previous Top8 strategy.
        if "Composite_Index" not in frontier.columns or "Frontier_Type" not in frontier.columns:
            return None
        f = _filter_valid_topics(frontier)
        hot = f[f["Frontier_Type"].astype(str) == "热点"].copy()
        if hot.empty:
            hot = f.copy()
        top_topics = hot.sort_values("Composite_Index", ascending=False).head(8)["Topic"].astype(int).tolist()
        if not top_topics:
            return None

        d = doc_map.copy()
        d["Year"] = pd.to_numeric(d["Year"], errors="coerce").fillna(0).astype(int)
        d = d[(d["Year"] > 0) & (d["Topic"].isin(top_topics))].copy()
        if d.empty:
            return None

        g = d.groupby(["Year", "Topic"]).size().reset_index(name="Docs")
        pivot = g.pivot(index="Year", columns="Topic", values="Docs").fillna(0).sort_index()

        short_map: Dict[int, str] = {}
        if topic_info is not None and not topic_info.empty:
            short_map, _ = _build_topic_label_maps(topic_info)

        used: Dict[str, int] = {}

        def legend_label(tid: int) -> str:
            base = _clean_label(short_map.get(tid, ""))
            if not base and topic_info is not None and not topic_info.empty:
                base = _clean_label(_make_topic_display(topic_info, tid, max_len=42))
            if not base or base.lower().startswith("topic "):
                base = "Unlabeled"
            base = base[:60] + ("…" if len(base) > 60 else "")
            used[base] = used.get(base, 0) + 1
            return f"{base} ({used[base]})" if used[base] > 1 else base

        fig, ax = plt.subplots(figsize=(12, 6))
        for topic_id in pivot.columns:
            tid = int(topic_id)
            ax.plot(
                pivot.index.values,
                pivot[topic_id].values,
                marker="o",
                linewidth=1.6,
                label=legend_label(tid),
            )
        ax.set_title("Temporal Evolution of Top 8 Topics")
        ax.set_xlabel("Year")
        ax.set_ylabel("Number of Documents")
        ax.legend(loc="upper left", fontsize=8, ncol=2)
        ax.grid(True, linestyle="--", alpha=0.3)
        fig.tight_layout()
        p = out_dir / "fig07_temporal_evolution.png"
        fig.savefig(p, dpi=300)
        plt.close(fig)
        return str(p)

    # Build year series for each selected topic
    d = doc_map.copy()
    d["Year"] = pd.to_numeric(d["Year"], errors="coerce").fillna(0).astype(int)
    d = d[d["Year"] > 0].copy()
    if d.empty:
        return None

    years = sorted(int(y) for y in d["Year"].unique().tolist() if int(y) > 0)
    if not years:
        return None

    def _series(topic_id: int) -> pd.Series:
        s = d[d["Topic"] == topic_id].groupby("Year").size()
        s = s.reindex(years, fill_value=0)
        return s

    # Pretty labels from topic_info (no IDs)
    short_map: Dict[int, str] = {}
    if topic_info is not None and not topic_info.empty:
        short_map, _ = _build_topic_label_maps(topic_info)

    def _nice(tid: int, fallback: str) -> str:
        base = _clean_label(short_map.get(tid, ""))
        if not base:
            return fallback
        return base

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True)
    colors = {
        "old": "#6C757D",
        "new_therapy": "#C41E3A",
        "new_dx": "#1F77B4",
    }

    # Panel A: Therapy
    ax = axes[0]
    s_old = _series(int(therapy_old))
    s_new = _series(int(therapy_new))
    ax.plot(years, s_old.values, color=colors["old"], linewidth=2.2, label="Traditional triple therapy")
    ax.plot(years, s_new.values, color=colors["new_therapy"], linewidth=2.6, label="Vonoprazan dual therapy")
    ax.set_title("Therapy: replacement")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Documents")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)

    # Panel B: Diagnosis
    ax = axes[1]
    s_old = _series(int(dx_old))
    s_new = _series(int(dx_new))
    ax.plot(years, s_old.values, color=colors["old"], linewidth=2.2, label="Conventional tests")
    ax.plot(years, s_new.values, color=colors["new_dx"], linewidth=2.6, label="AI imaging")
    ax.set_title("Diagnosis: disruption")
    ax.set_xlabel("Year")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)

    fig.suptitle("Temporal Evolution (Succession Narrative)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = out_dir / "fig07_temporal_evolution.png"
    fig.savefig(p, dpi=300)
    plt.close(fig)
    return str(p)


def plot_topic_distribution(topic_info: pd.DataFrame, out_dir: Path, method: str):
    """UMAP semantic overview with macro classes (journal-style overview).

    Note: filename kept for backward compatibility with manuscript links.
    """
    plt, sns = _ensure_matplotlib()
    if topic_info is None or topic_info.empty or "Topic" not in topic_info.columns:
        return None

    df = _filter_valid_topics(topic_info).copy()
    if df.empty:
        return None

    # Build per-topic text for clustering
    def _topic_text(r: pd.Series) -> str:
        parts: List[str] = []
        if "Topic_Label" in df.columns and pd.notna(r.get("Topic_Label")):
            parts.append(str(r.get("Topic_Label")))
        if "TopWords" in df.columns and pd.notna(r.get("TopWords")):
            parts.append(str(r.get("TopWords")).replace(";", " "))
        if "Representation" in df.columns and pd.notna(r.get("Representation")):
            parts.append(str(r.get("Representation")).replace(",", " "))
        return " ".join(parts)

    texts = [_topic_text(r) for _, r in df.iterrows()]

    # Vectorize + cluster into macro classes
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import AgglomerativeClustering

        vec = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), stop_words="english")
        X = vec.fit_transform(texts)
        n_topics = int(len(df))
        n_clusters = int(min(7, max(5, round(n_topics / 30))))  # 5–7 clusters
        cl = AgglomerativeClustering(n_clusters=n_clusters, linkage="average", metric="cosine")
        labels = cl.fit_predict(X.toarray())
    except Exception:
        # Fallback: single class
        labels = np.zeros(len(df), dtype=int)
        n_clusters = 1

    df["MacroCluster"] = labels

    # Name clusters with lightweight keyword heuristics (readable blocks)
    def _name_cluster(cluster_id: int) -> str:
        sub = df[df["MacroCluster"] == cluster_id]
        blob = " ".join([t.lower() for t in (sub.get("TopWords") if "TopWords" in sub.columns else []) if isinstance(t, str)])
        blob += " " + " ".join([t.lower() for t in (sub.get("Topic_Label") if "Topic_Label" in sub.columns else []) if isinstance(t, str)])

        def has(*ks: str) -> bool:
            return any(k in blob for k in ks)

        if has("therapy", "eradication", "triple", "quadruple", "vonoprazan", "amoxicillin", "clarithromycin"):
            return "Therapy"
        if has("ai", "images", "imaging", "endoscopy", "diagnosis", "breath", "urea", "ubt", "biopsy", "test"):
            return "Diagnosis"
        if has("caga", "vaca", "genotype", "epiya", "virulence", "baba"):
            return "Genetics/Virulence"
        if has("prevalence", "seroprevalence", "epidemiology", "children", "population"):
            return "Epidemiology"
        if has("cancer", "gc", "gastric", "lymphoma", "malt", "tumor"):
            return "Oncology"
        if has("microbiome", "microbiota", "microbial", "flora"):
            return "Microbiome"
        return "Mechanisms"

    names: Dict[int, str] = {cid: _name_cluster(int(cid)) for cid in sorted(df["MacroCluster"].unique().tolist())}
    # Ensure uniqueness
    inv: Dict[str, int] = {}
    for cid, nm in list(names.items()):
        inv[nm] = inv.get(nm, 0) + 1
        if inv[nm] > 1:
            names[cid] = f"{nm} {inv[nm]}"
    df["MacroName"] = df["MacroCluster"].map(names)

    # 2D embedding via UMAP (or PCA fallback)
    try:
        import umap

        reducer = umap.UMAP(n_neighbors=12, min_dist=0.25, metric="cosine", random_state=42)
        XY = reducer.fit_transform(X)
    except Exception:
        try:
            from sklearn.decomposition import PCA

            XY = PCA(n_components=2, random_state=42).fit_transform(X.toarray())
        except Exception:
            XY = np.random.default_rng(42).normal(size=(len(df), 2))

    df["x"] = XY[:, 0]
    df["y"] = XY[:, 1]

    # Plot: points + translucent ellipses (macro blocks)
    fig, ax = plt.subplots(figsize=(10.5, 7.0))
    order = sorted(df["MacroName"].unique().tolist())
    palette = dict(zip(order, sns.color_palette("tab10", n_colors=len(order))))

    sns.scatterplot(
        data=df,
        x="x",
        y="y",
        hue="MacroName",
        palette=palette,
        s=22,
        alpha=0.78,
        linewidth=0,
        ax=ax,
        legend=True,
    )

    # Ellipse helper
    from matplotlib.patches import Ellipse

    for name in order:
        sub = df[df["MacroName"] == name]
        if len(sub) < 6:
            continue
        pts = sub[["x", "y"]].to_numpy(dtype=float)
        mean = pts.mean(axis=0)
        cov = np.cov(pts.T)
        try:
            vals, vecs = np.linalg.eigh(cov)
            order_idx = vals.argsort()[::-1]
            vals = vals[order_idx]
            vecs = vecs[:, order_idx]
            angle = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))
            width, height = 2.6 * np.sqrt(np.maximum(vals, 1e-12))
            e = Ellipse(xy=mean, width=width, height=height, angle=angle)
            e.set_facecolor(palette[name])
            e.set_alpha(0.12)
            e.set_edgecolor(palette[name])
            e.set_linewidth(1.2)
            ax.add_patch(e)
        except Exception:
            pass

    ax.set_title(f"UMAP Overview of Topics (Macro Classes) ({method.upper()})")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.grid(True, linestyle="--", alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    leg = ax.get_legend()
    if leg is not None:
        leg.set_title("Macro Classes")
        for t in leg.get_texts():
            t.set_fontsize(9)

    p = out_dir / f"{PROJECT_PREFIX}_{method}_topic_distribution.png"
    fig.tight_layout()
    fig.savefig(p, dpi=300)
    plt.close(fig)
    return str(p)


def plot_frontier_bar(frontier: pd.DataFrame, out_dir: Path, method: str):
    plt, sns = _ensure_matplotlib()
    score_col = None
    # Prefer Composite_Index when available (this repo's core frontier metric)
    for c in ["Composite_Index", "frontier_score", "Frontier_Score", "score"]:
        if c in frontier.columns:
            score_col = c
            break
    if score_col is None:
        return None

    df = frontier.copy()
    if "Topic" in df.columns:
        df = df[df["Topic"] >= 0]
    df = df.sort_values(score_col, ascending=False).head(15)

    # Map Topic id to readable short labels if present
    labels = df["Topic"].astype(int).astype(str).tolist()
    if "Topic_Label" in df.columns:
        labels = [_clean_label(x)[:42] for x in df["Topic_Label"].astype(str).tolist()]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=labels, y=df[score_col], ax=ax, color="#54A24B")
    ax.set_title(f"Frontier Scores Top15 ({method.upper()})")
    ax.set_xlabel("Topic (label)")
    ax.set_ylabel(score_col)
    ax.tick_params(axis="x", rotation=45)
    p = out_dir / f"{PROJECT_PREFIX}_{method}_frontier_top15.png"
    fig.tight_layout()
    fig.savefig(p, dpi=300)
    plt.close(fig)
    return str(p)


def plot_wordcloud(topic_info: pd.DataFrame, out_dir: Path, method: str):
    try:
        from wordcloud import WordCloud  # type: ignore[import-not-found]
    except Exception:
        return None

    col = "TopWords" if "TopWords" in topic_info.columns else ("Representation" if "Representation" in topic_info.columns else None)
    if col is None:
        return None

    words = []
    for v in topic_info[col].dropna().astype(str).tolist()[:50]:
        s = v.strip().strip("[]").replace("'", "").replace('"', "")
        words.extend([p.strip() for p in s.split(",") if p.strip()])

    if not words:
        return None

    freq: Dict[str, int] = {}
    for w in words:
        if len(w) < 2:
            continue
        freq[w] = freq.get(w, 0) + 1

    wc = WordCloud(width=1600, height=900, background_color="white")
    wc.generate_from_frequencies(freq)

    plt, _ = _ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"WordCloud ({method.upper()})")
    p = out_dir / f"{PROJECT_PREFIX}_{method}_wordcloud.png"
    fig.tight_layout()
    fig.savefig(p, dpi=300)
    plt.close(fig)
    return str(p)


def main() -> int:
    parser = argparse.ArgumentParser(description="Step09: 可视化")
    parser.add_argument(
        "--base_dir",
        default=None,
        help="工作目录（包含 07_topic_models/08_model_selection/09_visualization）。默认使用 MAIN_WORKDIR.txt 指向的目录。",
    )
    parser.add_argument("--only", choices=ALL_METHODS, help="只跑指定方法")
    args = parser.parse_args()

    base_dir = _resolve_base_dir(args.base_dir)
    methods = [args.only] if args.only else ALL_METHODS

    best_path = base_dir / "08_model_selection" / "best_mc_by_method.json"
    if not best_path.exists():
        print("缺少 best_mc_by_method.json，请先运行 step08_cv_select.py")
        return 2

    best_map = json.loads(best_path.read_text(encoding="utf-8"))

    print("=" * 80)
    print(f"Step 09 可视化 - {get_project_name()} ({PROJECT_PREFIX})")
    print("=" * 80)

    ok_any = False

    for m in methods:
        m_best = best_map.get(m)
        if not m_best:
            print(f"\n→ {m}: ✗ 未找到 best mc（请先运行 Step08）")
            continue
        mc = int(m_best["mc"])

        method_dir = base_dir / "07_topic_models" / _safe_method_dir(m)
        topic_info = method_dir / f"{PROJECT_PREFIX}_mc{mc}_topic_info.csv"
        frontier = method_dir / f"{PROJECT_PREFIX}_mc{mc}_frontier_indicators.csv"
        doc_map = method_dir / f"{PROJECT_PREFIX}_mc{mc}_doc_topic_mapping.csv"

        if not topic_info.exists():
            print(f"\n→ {m}: ✗ 缺少 {topic_info.name}")
            continue

        out_dir = base_dir / "09_visualization" / m.upper()
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n→ {m}: 使用 best_mc={mc} 生成图表...", end="", flush=True)

        df_topic = pd.read_csv(topic_info)
        df_front = pd.read_csv(frontier) if frontier.exists() else pd.DataFrame()
        df_map = pd.read_csv(doc_map) if doc_map.exists() else pd.DataFrame()
        charts: List[str] = []

        # Regenerate fig01-fig07 with latest CSVs (overwrite old images)
        p01 = _plot_fig01_topic_noise_comparison(df_map, out_dir, m)
        if p01:
            charts.append(p01)
        p02 = _plot_fig02_frontier_evolution(df_map, df_front, out_dir, m)
        if p02:
            charts.append(p02)
        p03 = _plot_fig03_frontier_network_placeholder(df_front, out_dir, m)
        if p03:
            charts.append(p03)
        p04 = _plot_fig04_topic_size_violin(df_topic, out_dir, m)
        if p04:
            charts.append(p04)
        p05 = _plot_fig05_citation_heatmap(df_front, out_dir, m)
        if p05:
            charts.append(p05)
        p06 = _plot_fig06_frontier_bubble(df_front, out_dir, m)
        if p06:
            charts.append(p06)
        p07 = _plot_fig07_temporal_evolution(df_map, df_front, df_topic, out_dir, m)
        if p07:
            charts.append(p07)

        p1 = plot_topic_distribution(df_topic, out_dir, m)
        if p1:
            charts.append(p1)
        p3 = plot_wordcloud(df_topic, out_dir, m)
        if p3:
            charts.append(p3)

        if not df_front.empty:
            # Enrich frontier with readable labels for bar charts / tables
            short_map, long_map = _build_topic_label_maps(df_topic)
            if "Topic" in df_front.columns:
                df_front = df_front.copy()
                df_front["Topic_Label"] = df_front["Topic"].apply(lambda t: long_map.get(int(t), f"Topic {int(t)}") if int(t) >= 0 else "Outliers")

            p2 = plot_frontier_bar(df_front, out_dir, m)
            if p2:
                charts.append(p2)

        # Write an explainer HTML that tells readers how to interpret the figures
        try:
            report_path = _write_viz_report_html(out_dir, m, mc, charts, df_topic, df_front, df_map)
            charts.append(report_path)
        except Exception:
            pass

        manifest = {
            "method": m,
            "best_mc": mc,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "charts": charts,
        }
        (out_dir / "viz_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"✓ ({len(charts)} 张)")
        ok_any = ok_any or len(charts) > 0

    print("\n" + "=" * 80)
    print("Step 09 完成")
    print("输出目录: 09_visualization/")
    print("=" * 80)

    return 0 if ok_any else 1


if __name__ == "__main__":
    raise SystemExit(main())
