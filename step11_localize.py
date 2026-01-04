#!/usr/bin/env python3
"""step11_localize.py
独立脚本：在 `11_chinese/` 目录下生成中文化份本，供人工检查。
做法：
- 复制 `10_report/*/*_report.md` 为中文化版本（不修改原文件）
- 将图片（来自 `09_visualization`）复制到对应输出目录并修正链接
- 收集所有包含 "frontier_indicators" 的 CSV，生成带 `Topic_Description_CN` 的副本
- 所有输出放在 `11_chinese/<METHOD>/` 或 `11_chinese/csvs/`

注意：不会修改其他文件。
"""

from pathlib import Path
import re
import shutil
import csv
import sys
import json

try:
    import pandas as pd
except Exception:
    pd = None

# 导入配置
try:
    from config import get_project_name
except ImportError:
    def get_project_name():
        return "幽门螺杆菌"

BASE = Path(__file__).resolve().parent
REPORTS_DIR = BASE / '10_report'
OUT_BASE = BASE / '11_chinese'
VIS_DIR = BASE / '09_visualization'

# 简单关键词到中文的映射，用于CSV中文描述（覆盖常见词）
SIMPLE_CN = {
    'therapy': '治疗', 'eradication': '根除', 'treatment': '治疗', 'resistance': '耐药',
    'cancer': '癌症', 'gastric': '胃', 'ulcer': '溃疡', 'lymphoma': '淋巴瘤',
    'microbiota': '微生物群', 'microbiome': '微生物组', 'diagnosis': '诊断', 'detection': '检测',
    'ai': '人工智能', 'images': '影像', 'immunotherapy': '免疫治疗', 'lncrna': '长链非编码RNA',
    'urease': '尿素酶', 'omvs': '外膜囊泡', 'biofilm': '生物膜', 'probiotics': '益生菌',
}

# md 文本替换映射（将关键英文表头或短语替换为中文）
MD_REPLACEMENTS = {
    '| Rank | Topic | Score | Topic Description |': '| 排名 | 主题ID | 得分 | 主题描述 |',
    'Documents:': '文献数:',
    'Keywords:': '关键词:',
    'Current Research Hotspots': '当前热点研究方向',
    'Emerging Research Directions': '新兴研究方向',
    'High-Impact Research': '高影响力研究',
    'Research Recommendations': '研究建议',
    'Avg. citations': '平均被引',
    'Recent ratio': '近期占比',
    'Avg. year': '平均年份',
}


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def translate_keywords_to_cn(keywords_str: str) -> str:
    """把TopWords字符串（逗号分隔）转换为简短中文描述（供人工检测使用）"""
    kws = [k.strip().lower() for k in re.split('[,;]', keywords_str) if k.strip()]
    cn_parts = []
    for k in kws[:4]:
        if k in SIMPLE_CN:
            cn_parts.append(SIMPLE_CN[k])
        else:
            # 若为缩写或无映射，尽量保留原词（英）并加括号
            cn_parts.append(k)
    if not cn_parts:
        return ''
    return ' '.join(cn_parts)


### 英文短语 -> 中文短句 映射（用于将专业英文短语翻译为高质量中文）
PHRASE_EN2CN = {
    "H. pylori eradication therapy": "幽门螺杆菌根除治疗",
    "Peptic ulcer disease": "消化性溃疡",
    "Gastric cancer pathogenesis": "胃癌发生机制",
    "Immune response mechanisms": "免疫反应机制",
    "MALT lymphoma": "MALT 淋巴瘤",
    "CagA/VacA virulence factors": "CagA/VacA 毒力因子",
    "Gut microbiota interaction": "肠道微生物群相互作用",
    "Natural extract activity": "天然提取物活性",
    "Infection prevalence": "感染患病率",
    "Cancer risk factors": "癌症危险因素",
    "AI-based image analysis": "基于人工智能的影像分析",
    "Tumor immune microenvironment": "肿瘤免疫微环境",
    "Electrochemical biosensor detection": "电化学生物传感检测",
    "NAFLD and liver disease": "非酒精性脂肪肝及相关肝病",
    "LncRNA in cancer progression": "长链非编码 RNA 与癌症进展",
    "Biofilm formation": "生物膜形成",
    "Urease enzyme activity": "尿素酶活性",
    "Outer membrane vesicles": "外膜囊泡 (OMVs)",
    "Drug delivery systems": "药物递送系统",
    "Antibiotic resistance": "抗生素耐药",
    "Antibiotic resistance in treatment": "治疗中的抗生素耐药问题",
    "Clinical guidelines and consensus": "临床指南与共识",
    "Probiotic effects": "益生菌作用研究",
    "Infection-attributable cancers": "感染相关的癌症负担",
}

# 如果存在最终词典（由 merge 脚本生成），优先加载
FINAL_MAP_PATH = OUT_BASE / 'PHRASE_EN2CN_final.json'
try:
    if FINAL_MAP_PATH.exists():
        import json as _json
        _fm = _json.loads(FINAL_MAP_PATH.read_text(encoding='utf-8'))
        # FINAL json may store nested entries; normalize to english->final_chinese when status accepted/provisional
        FINAL_MAP = {}
        for k, v in _fm.items():
            if isinstance(v, dict):
                status = v.get('status')
                final_cn = v.get('final_chinese') or ''
                if status in ('accepted', 'provisional') and final_cn:
                    FINAL_MAP[k] = final_cn
        if FINAL_MAP:
            # extend PHRASE_EN2CN with FINAL_MAP (prioritize final)
            PHRASE_EN2CN.update(FINAL_MAP)
            print(f'Loaded {len(FINAL_MAP)} entries from {FINAL_MAP_PATH}')
except Exception:
    pass


def convert_md_to_html(md_content: str, title: str) -> str:
    """将 Markdown 转换为漂亮的 HTML 页面（简化版）"""
    # 基础 HTML 转换
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
    
    # 如果有 markdown 库，使用它
    try:
        import markdown
        html_body = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])
    except ImportError:
        pass
    
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
            Generated by Topic Modeling Pipeline | 中文化版本
        </div>
    </div>
</body>
</html>'''
    return html


def collect_and_expand_phrase_mapping() -> dict:
    """从 step10_report.py 提取硬编码的英文短语，并生成中文草案映射（返回字典）。
    会写入 `11_chinese/PHRASE_EN2CN_draft.json` 以供人工校对。
    """
    src = BASE / 'step10_report.py'
    if not src.exists():
        return {}

    txt = src.read_text(encoding='utf-8')

    # 捕获可能的短语：位于 dict 值处的双引号字符串，且包含空格或括号（简易筛选）
    cand = re.findall(r'"([A-Za-z0-9\-\.,() /]+ [A-Za-z0-9\-\.,() /]+)"', txt)
    cand = sorted(set([c.strip() for c in cand if len(c.strip()) > 3]))

    # 也尝试从单关键词映射中抓取短小短语（如 "Urease enzyme studies"）
    more = re.findall(r'"([A-Za-z0-9 ]+ (?:studies|analysis|mechanisms|therapy|detection|research|activity|effects|management))"', txt)
    for m in more:
        cand.append(m.strip())

    cand = sorted(set(cand))

    # 生成中文草案映射
    draft = {}
    for en in cand:
        if en in PHRASE_EN2CN:
            draft[en] = PHRASE_EN2CN[en]
            continue
        cn = generate_cn_for_phrase(en)
        draft[en] = cn

    # 写入草案文件
    try:
        ensure_dir(OUT_BASE)
        outf = OUT_BASE / 'PHRASE_EN2CN_draft.json'
        outf.write_text(json.dumps(draft, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'Wrote phrase mapping draft: {outf} ({len(draft)} entries)')
    except Exception as e:
        print(f'Failed to write draft mapping: {e}')

    return draft


def generate_cn_for_phrase(en: str) -> str:
    """基于词典和启发式规则生成中文候选（草案）。"""
    s = en.strip()
    # 简单 token 化（保留缩写 H., H. pylori 处理）
    tokens = re.findall(r"[A-Za-z0-9]+", s)
    tokens_l = [t.lower() for t in tokens]

    # 预定义术语映射（扩展 SIMPLE_CN）
    TERM_MAP = {
        'h': '幽门螺杆菌', 'h.pylori': '幽门螺杆菌', 'hpylori': '幽门螺杆菌', 'h.pylori.': '幽门螺杆菌',
        'pylori': '幽门螺杆菌', 'ai': '人工智能', 'ml': '机器学习', 'dna': 'DNA', 'rna': 'RNA',
        'lncrna': '长链非编码 RNA', 'lncrnas': '长链非编码 RNA', 'omvs': '外膜囊泡',
        'urease': '尿素酶', 'biofilm': '生物膜', 'probiotics': '益生菌', 'probiotic': '益生菌',
        'antibiotic': '抗生素', 'resistance': '耐药', 'cancer': '癌', 'gastric': '胃', 'ulcer': '溃疡',
        'therapy': '治疗', 'eradication': '根除', 'detection': '检测', 'diagnosis': '诊断',
        'infection': '感染', 'prevalence': '患病率', 'microbiome': '微生物组', 'microbiota': '微生物群',
        'tumor': '肿瘤', 'immune': '免疫', 'immunotherapy': '免疫治疗', 'vaccine': '疫苗',
        'endoscopy': '内镜', 'endoscopic': '内镜', 'biopsy': '活检', 'mice': '小鼠', 'mouse': '小鼠',
    }

    parts = []
    for t in tokens_l:
        if t in SIMPLE_CN:
            parts.append(SIMPLE_CN[t])
        elif t in TERM_MAP:
            parts.append(TERM_MAP[t])
        else:
            # 若包含常见词后缀，尝试直接保留英文关键字（作为候选）
            parts.append(t)

    # 尝试构造较自然的中文短句：若包含胃/癌同时存在则合并为 '胃癌...'
    if any(p == '胃' for p in parts) and any('癌' in p or p == '癌' for p in parts):
        # remove separate '胃' and '癌' tokens
        joined = ''.join([p for p in parts if p not in ('胃','癌')])
        return '胃癌' + joined

    # 常见模式： <部位><疾病><机制/分析>
    # 若首个是中文词则直接拼接
    if parts and any(ord(ch) > 128 for ch in parts[0]):
        cand = ''.join(parts)
        # 美化：替换重复空格
        cand = re.sub(r'\s+', ' ', cand).strip()
        return cand

    # 回退：翻译组合为中文词并用空格分隔
    return ' '.join(parts)


def translate_phrases_in_file(md_file: Path):
    """在已生成的 _zh.md 中将英文短语替换为中文短句（生成中英双语：中文（原英文））。"""
    txt = md_file.read_text(encoding='utf-8')

    # 尝试从 step10_report.py 中收集未覆盖的英文短语并生成中文草案
    try:
        draft_map = collect_and_expand_phrase_mapping()
    except Exception:
        draft_map = {}

    # 合并已有映射（PHRASE_EN2CN 优先）并按长度降序替换，避免短子串优先替换
    merged = dict(draft_map)
    merged.update(PHRASE_EN2CN)

    def _has_cjk(s: str) -> bool:
        return bool(re.search(r"[\u4e00-\u9fff]", str(s or "")))

    for en in sorted(merged.keys(), key=lambda s: -len(s)):
        cn = str(merged[en] or "").strip()
        # 避免生成 "English (English)"：中文候选必须真的包含中文，或至少与英文不同
        if not cn:
            continue
        try:
            pattern = re.compile(re.escape(en), flags=re.IGNORECASE)

            def _repl(m):
                orig = m.group(0)
                # 若命中片段已经处在括号内（很可能已经是“中文（英文）”），跳过
                start, end = m.start(), m.end()
                before = txt[start - 1] if start - 1 >= 0 else ''
                after = txt[end] if end < len(txt) else ''
                if before in ('(', '（') or after in (')', '）'):
                    return orig

                # 若已包含中文或已经是中英形式，则跳过替换
                if cn in orig or _has_cjk(orig):
                    return orig
                if (not _has_cjk(cn)) and cn.casefold() == orig.casefold():
                    return orig
                # 若文本中已经存在 (orig) 格式，避免重复
                if re.search(re.escape(f"({orig})"), txt):
                    return orig
                return f"{cn} ({orig})"

            txt = pattern.sub(_repl, txt)
        except Exception:
            try:
                txt = txt.replace(en, f"{cn} ({en})")
            except Exception:
                pass

    md_file.write_text(txt, encoding='utf-8')
    print(f'Translated phrases in: {md_file} (used {len(merged)} mappings)')



def localize_md_file(md_path: Path, out_dir: Path):
    content = md_path.read_text(encoding='utf-8')

    # 替换常用表头/短语
    for k, v in MD_REPLACEMENTS.items():
        content = content.replace(k, v)

    # 处理图片引用并复制图片
    def repl_img(m):
        alt, path = m.group(1), m.group(2)
        orig_path = (md_path.parent / path).resolve()
        # 若路径相对并指向 09_visualization 或包含该部分，则复制
        if '09_visualization' in str(path) or '09_visualization' in str(orig_path):
            ensure_dir(out_dir / 'images')
            if orig_path.exists():
                dst = out_dir / 'images' / orig_path.name
                try:
                    shutil.copy2(orig_path, dst)
                except Exception:
                    pass
                new_rel = './images/' + orig_path.name
                return f'![{alt}]({new_rel})'
        # 其他情况：保持原引用
        return m.group(0)

    content = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', repl_img, content)

    out_file = out_dir / (md_path.stem + '_zh.md')
    out_file.write_text(content, encoding='utf-8')
    print(f'Wrote: {out_file}')
    # 生成后尝试将已知的英文短语替换为中文短句（可人工校对）
    try:
        translate_phrases_in_file(out_file)
    except Exception:
        pass
    return out_file


def process_reports():
    if not REPORTS_DIR.exists():
        print('No 10_report directory found. Abort.')
        return
    ensure_dir(OUT_BASE)

    for method_dir in REPORTS_DIR.iterdir():
        if not method_dir.is_dir():
            continue
        out_sub = OUT_BASE / method_dir.name
        ensure_dir(out_sub)

        # 复制并中文化 Markdown 报告
        md_files = list(method_dir.glob('*_report.md'))
        for md in md_files:
            try:
                zh_md = localize_md_file(md, out_sub)
                # 生成中文化的 HTML
                zh_md_content = zh_md.read_text(encoding='utf-8')
                zh_title = f"{get_project_name()} 主题建模研究报告（{method_dir.name} 中文化版本）"
                zh_html_content = convert_md_to_html(zh_md_content, zh_title)
                zh_html_file = out_sub / md.name.replace('.md', '_zh.html')
                zh_html_file.write_text(zh_html_content, encoding='utf-8')
                print(f'Generated localized HTML: {zh_html_file}')
            except Exception as e:
                print(f'Error processing {md}: {e}')

        # 复制 HTML 报告原件（供人工对照）
        html_files = list(method_dir.glob('*_report.html'))
        for h in html_files:
            try:
                shutil.copy2(h, out_sub / h.name)
            except Exception:
                pass

    print('Reports localized.')


def collect_and_localize_csvs():
    out_csv_dir = OUT_BASE / 'csvs'
    ensure_dir(out_csv_dir)

    # 寻找 workspace 下含 frontier_indicators 的 CSV
    candidates = list(BASE.rglob('*frontier*indicators*.csv')) + list(BASE.rglob('*frontier_indicators*.csv'))

    # 过滤掉输出目录自身、备份目录等，避免重复运行时出现“同文件复制”和 _zh_zh.csv
    filtered = []
    for p in candidates:
        try:
            rp = p.resolve()
        except Exception:
            rp = p

        if OUT_BASE in p.parents:
            continue

        parts_lower = {part.lower() for part in p.parts}
        if any('backup' in part for part in parts_lower):
            continue

        if p.name.lower().endswith('_zh.csv'):
            continue

        filtered.append(rp)

    # 同名 CSV 可能在多个目录存在：优先选择 10_report 下的版本（更贴近最终报告输出）
    best_by_name = {}
    for p in filtered:
        name = p.name
        score = 0
        p_str = str(p).lower()
        if '10_report' not in p_str:
            score += 10
        if '10_report' in p_str and 'backups' in p_str:
            score += 100
        if name not in best_by_name or score < best_by_name[name][0]:
            best_by_name[name] = (score, p)

    found = [v[1] for v in best_by_name.values()]
    found = sorted(found, key=lambda p: str(p))

    for f in found:
        try:
            dst = out_csv_dir / f.name
            try:
                if f.resolve() != dst.resolve():
                    shutil.copy2(f, dst)
                    print(f'Copied CSV: {dst}')
            except Exception:
                # 回退：无法 resolve 时仍尝试复制，但避免同名同路径
                if str(f) != str(dst):
                    shutil.copy2(f, dst)
                    print(f'Copied CSV: {dst}')

            # 已经是中文化版本就不再二次本地化，避免 _zh_zh.csv
            if f.name.lower().endswith('_zh.csv'):
                continue

            # 用 pandas 生成带中文描述的副本（若 pandas 可用）
            if pd is not None:
                try:
                    df = pd.read_csv(f)
                    if 'TopWords' in df.columns:
                        df['Topic_Description_CN'] = df['TopWords'].fillna('').apply(translate_keywords_to_cn)
                    else:
                        # 尝试用列名包含 'Words' 的列
                        candidates = [c for c in df.columns if 'word' in c.lower() or 'top' in c.lower()]
                        if candidates:
                            df['Topic_Description_CN'] = df[candidates[0]].fillna('').apply(translate_keywords_to_cn)
                        else:
                            df['Topic_Description_CN'] = ''

                    out_local = out_csv_dir / (f.stem + '_zh.csv')
                    df.to_csv(out_local, index=False)
                    print(f'Wrote localized CSV: {out_local}')
                except Exception as e:
                    print(f'Failed to localize CSV {f}: {e}')
        except Exception as e:
            print(f'Error copying {f}: {e}')

    if not found:
        print('No frontier CSV found to copy.')


def write_readme():
    ensure_dir(OUT_BASE)
    readme = OUT_BASE / 'README.md'
    readme.write_text(
        """11_chinese/ - 人工检查用的中文化副本

说明:
- 所有原始文件（位于 10_report/）均未被修改；本目录为独立复制品。
- 每个方法在 `11_chinese/<METHOD>/` 下包含：
  - `<report>_zh.md` : 中文化的 Markdown 报告（主要替换表头、关键短语并复制可视化图片）
  - `<report>.html` : 原始 HTML 报告（未修改，供对照）
  - `images/` : 复制的可视化图片（仅从 09_visualization 中复制在报告中引用的图片）
- `11_chinese/csvs/` 包含复制的 frontier 指标 CSV，并针对每个 CSV 生成带 `Topic_Description_CN` 的副本（若 pandas 可用）。

使用说明:
- 请在 `11_chinese/` 中人工检查 `_zh.md` 文件，核对中文化是否符合学术表述。
- 脚本不会修改原始数据或报告，适合人工审核后再决定是否覆盖原文件。
""", encoding='utf-8')
    print(f'Wrote README: {readme}')


if __name__ == '__main__':
    process_reports()
    collect_and_localize_csvs()
    write_readme()
    print('Step11 localization complete.')

