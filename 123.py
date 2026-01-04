import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from wordcloud import WordCloud
from pathlib import Path
import json
import re

# === 顶刊配色方案 (Nature/Science 风格) ===
COLOR_PALETTE = {
    "Clinical Therapy": "#E0E0E0",      # 灰色 (背景板，体现"老牌")
    "Diagnostics & AI": "#7B68EE",      # 蓝紫色 (高亮，体现"科技")
    "Basic Mechanisms": "#4682B4",      # 钢蓝 (稳重)
    "Oncology & Pathology": "#CD5C5C",  # 印度红 (警示)
    "Epidemiology & Public Health": "#2E8B57", # 海绿
    "Extra-gastric Diseases": "#F4A460" # 沙褐色
}

# 字体配置 (尝试使用无衬线字体)
FONT_PATH = None  # 如果有 Arial/Roboto 路径可填入，否则用默认

def get_project_root():
    return Path(__file__).resolve().parents[1]

def load_data():
    """自动寻找最新的 topic_info 和 macro_map"""
    root = get_project_root()
    
    # 1. Load Macro Map
    map_file = root / "macro_topic_map.csv"
    if not map_file.exists():
        raise FileNotFoundError("找不到 macro_topic_map.csv")
    df_map = pd.read_csv(map_file)
    # Ensure ID match
    df_map['Topic'] = pd.to_numeric(df_map['Topic'], errors='coerce')
    
    # 2. Load Topic Info (Size)
    topic_dir = root / "07_topic_models" / "C"
    info_files = list(topic_dir.glob("*_topic_info.csv"))
    if not info_files:
        raise FileNotFoundError("找不到 07_topic_models/C/*_topic_info.csv")
    # Take the latest one
    info_file = sorted(info_files)[-1]
    print(f"读取数据源: {info_file.name}")
    df_info = pd.read_csv(info_file)
    
    # Merge
    df = pd.merge(df_info, df_map[['Topic', 'Macro']], on='Topic', how='left')
    df['Macro'] = df['Macro'].fillna('Other')
    
    # Filter outliers
    df = df[df['Topic'] != -1]
    return df

# === 1. 核心算法：Treemap (Squarify 简化版) ===
def normalize_sizes(sizes, dx, dy):
    total_size = sum(sizes)
    return [s * dx * dy / total_size for s in sizes]

def layout(sizes, x, y, dx, dy):
    # Simple "slice-and-dice" approximation for robustness if squarify not installed
    # Or a recursive splitter. Let's use a simple row-stacking strategy.
    # This is a mini-implementation of a tiling algorithm.
    
    rects = []
    if not sizes:
        return []

    # Simple vertical stack (fallback)
    # For a real "squarified" look, we usually need the library.
    # But to ensure it runs, we use a simple "Flow" layout.
    
    current_y = y
    width = dx
    total_area = sum(sizes)
    
    for s in sizes:
        height = (s / total_area) * dy
        rects.append({'x': x, 'y': current_y, 'dx': width, 'dy': height})
        current_y += height
        
    return rects

# 尝试导入真正的 squarify，如果没有则报错提示安装
try:
    import squarify
    HAS_SQUARIFY = True
except ImportError:
    HAS_SQUARIFY = False
    print("⚠️ 提示: 为了获得完美的 Treemap，建议运行 `pip install squarify`")

def draw_high_end_treemap(df):
    print("正在绘制图二 (Treemap)...")
    # Aggregate by Macro first to define regions? 
    # No, Squarify handles all rects. We just need to sort them.
    
    # Sort by Macro (to cluster colors) then by Count
    df = df.sort_values(by=['Macro', 'Count'], ascending=[True, False])
    
    sizes = df['Count'].values.tolist()
    labels = [f"T{t}" if s > sum(sizes)*0.01 else "" for t, s in zip(df['Topic'], sizes)]
    macros = df['Macro'].values.tolist()
    colors = [COLOR_PALETTE.get(m, "#999999") for m in macros]
    
    plt.figure(figsize=(12, 8), dpi=300)
    
    if HAS_SQUARIFY:
        squarify.plot(sizes=sizes, label=labels, color=colors, alpha=0.9, 
                      bar_kwargs=dict(linewidth=1, edgecolor='white'),
                      text_kwargs=dict(fontsize=9, color='white', weight='bold'))
    else:
        # Fallback: simple colored bar chart if they refuse to install squarify
        # But user wants a Treemap. 
        print("❌ 错误: 必须安装 squarify 才能生成非'一坨'的矩形树图。")
        print("请运行: pip install squarify")
        return

    # Custom Legend
    handles = [patches.Patch(color=c, label=m) for m, c in COLOR_PALETTE.items() if m in df['Macro'].unique()]
    plt.legend(handles=handles, title="Macro Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.axis('off')
    plt.title("Topic Landscape by Macro-Cluster (Top-Tier Style)", fontsize=14, pad=20)
    
    out_path = get_project_root() / "12_top_journal_upgrade" / "C" / "fig02_macro_treemap.png"
    plt.savefig(out_path, bbox_inches='tight')
    print(f"✅ 图二已升级: {out_path}")

# === 2. 核心算法：圆形遮罩 Wordcloud ===
def create_circle_mask(h, w):
    center_x, center_y = w // 2, h // 2
    radius = min(center_x, center_y)
    
    y, x = np.ogrid[:h, :w]
    mask = (x - center_x) ** 2 + (y - center_y) ** 2 > radius ** 2
    return 255 * mask.astype(int)

def draw_pro_wordcloud(df, topic_id, filename, color_func):
    print(f"正在绘制词云 (Topic {topic_id})...")
    row = df[df['Topic'] == topic_id]
    if row.empty:
        print(f"⚠️ 找不到 Topic {topic_id}")
        return
        
    # Extract words
    text = str(row.iloc[0]['Representation']) # BERTopic default col
    if text.startswith("["):
        import ast
        try:
            words = ast.literal_eval(text)
            # Create freq dict: 1st word=100, 2nd=90...
            freq = {w: 100-i*5 for i, w in enumerate(words)}
        except:
            freq = {text: 100}
    else:
        # Fallback parsing
        freq = {w: 100-i*5 for i, w in enumerate(text.replace("·", " ").split())}
        
    # Create Mask
    mask = create_circle_mask(800, 800)
    
    wc = WordCloud(
        width=800, height=800,
        background_color="white",
        mask=mask,
        color_func=color_func,
        max_font_size=120,
        min_font_size=20,
        prefer_horizontal=0.9,
        font_path=FONT_PATH,
        random_state=42
    ).generate_from_frequencies(freq)
    
    plt.figure(figsize=(6, 6), dpi=300)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    
    out_path = get_project_root() / "12_top_journal_upgrade" / "C" / filename
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    print(f"✅ 词云已升级: {out_path}")

# 自定义配色函数
def blue_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    # 科技蓝/紫渐变 (H: 240, S: 70-100, L: 40-60)
    return "hsl(240, 80%, %d%%)" % np.random.randint(30, 60)

def red_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    # 医药红/橙渐变 (H: 0-20, S: 80-100, L: 40-60)
    return "hsl(10, 90%, %d%%)" % np.random.randint(40, 60)

def main():
    try:
        df = load_data()
        
        # 1. Draw Treemap
        draw_high_end_treemap(df)
        
        # 2. Draw Wordclouds
        # Topic 90 (AI)
        draw_pro_wordcloud(df, 90, "fig05_wordcloud_ai.png", blue_color_func)
        
        # Topic 128 (Vonoprazan)
        draw_pro_wordcloud(df, 128, "fig06_wordcloud_vonoprazan.png", red_color_func)
        
    except Exception as e:
        print(f"❌ 运行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()