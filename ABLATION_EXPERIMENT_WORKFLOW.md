#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ABLATION_EXPERIMENT_WORKFLOW.md
消融实验工作流说明和中间结果

本文档记录：
1. 消融实验的完整工作流
2. 已有的完成情况（baseline）
3. 预期结果分析
4. 论文撰写指导
"""

import json
from pathlib import Path
import pandas as pd

# ============================================================================
# 收集现有数据
# ============================================================================

print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║              消融实验（Ablation Study）工作流总结                          ║
╚═══════════════════════════════════════════════════════════════════════════╝

【第 1 阶段】数据准备 ✓ 完成
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ 创建了 4 个消融版本的向量：
  1. baseline:            无投影（原始向量）
  2. M_S:                 投影移除 26 个词（Methodology + Statistics）
  3. M_S_B:               投影移除 40 个词（+ Background）
  4. M_S_B_Anatomy:       投影移除 48 个词（+ Anatomy 解剖学词）

✓ 向量验证：
""")

# 检查向量文件
versions = ["baseline", "M_S", "M_S_B", "M_S_B_Anatomy"]
import numpy as np

vector_status = []
for version in versions:
    vector_file = Path(f"ablation_outputs/{version}/c_{version}_final_clean_vectors.npz")
    if vector_file.exists():
        data = np.load(vector_file)
        shape = data['embeddings'].shape
        norm_mean = np.linalg.norm(data['embeddings'], axis=1).mean()
        vector_status.append({
            "版本": version,
            "文件": "✓",
            "向量数": shape[0],
            "维度": shape[1],
            "归一化": f"{norm_mean:.4f}",
        })
        print(f"  ✓ {version:20s} shape={shape} norm={norm_mean:.4f}")
    else:
        print(f"  ✗ {version:20s} 向量文件不存在")

print(f"""
【第 2 阶段】BERTopic 主题建模 ⏳ 进行中
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

当前进度：
""")

# 检查 BERTopic 结果
bert_status = []
for version in versions:
    output_dir = Path(f"07_topic_models/ABLATION_{version}")
    topic_files = list(output_dir.glob("*_topic_info.csv"))
    
    if topic_files:
        # 有结果
        df = pd.read_csv(topic_files[0])
        num_topics = int((df["Topic"] >= 0).sum())
        status = "✓ 完成"
        mc = topic_files[0].name.split("_mc")[1].split("_")[0]
        print(f"  {status}  {version:20s} (mc={mc}, {num_topics} 个主题)")
        bert_status.append({
            "版本": version,
            "状态": "完成",
            "mc": int(mc),
            "主题数": num_topics,
        })
    else:
        print(f"  ⏳ 进行中 {version:20s}")
        bert_status.append({
            "版本": version,
            "状态": "进行中",
        })

print(f"""
【第 3 阶段】结构指标计算 ⏳ 待进行
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

计划计算的指标：
  1. Silhouette 系数（簇紧凑性，-1 到 1）
  2. kNN 混合率（跨簇邻居比例，0 到 1）
  3. Davies-Bouldin 指数（簇分离，越小越好）
  4. 簇隔离度（簇间/簇内距离比，>1 为好）

命令：python compute_ablation_structural_metrics.py

【预期结果分析】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

基于 VPD（M_S_B_Anatomy）的 C_v 分数对比：

方案        | C_v 分数  | 改进幅度
────────────┼──────────┼─────────
Baseline    | 0.5950   | 0%
M_S         | ~0.6050  | ~1.7%
M_S_B       | ~0.6150  | ~3.4%
M_S_B_Anat  | 0.6261   | +5.2%   ← 已验证的 VPD 效果

推断：M_S 和 M_S_B 的分数应该介于 baseline 和 VPD 之间。

【论文撰写指导】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 方法部分（Methods）
   "To investigate the contribution of different noise components, we 
   conducted an ablation study by progressively removing noise word 
   groups from the denoising projection:
   - M_S: Remove Methodology and Statistics words (26 words)
   - M_S_B: Add Background words (40 words total)
   - M_S_B_Anatomy: Full denoising with Anatomy words (48 words total)"

2. 结果部分（Results）
   表格：消融实验对比
   
   版本              | 主题数 | 噪声% | C_v 分数 | Silhouette | 隔离度
   ─────────────────┼────────┼──────┼─────────┼────────────┼──────
   Baseline         |   82   | 2.41 | 0.5950  |  TBD       | TBD
   M_S (26 词)      |  TBD   | TBD  | TBD     |  TBD       | TBD
   M_S_B (40 词)    |  TBD   | TBD  | TBD     |  TBD       | TBD
   M_S_B_Anatomy    |  TBD   | TBD  | 0.6261  |  TBD       | TBD
   
   分析：
   "The ablation study demonstrates that..."
   [根据实际数据补充]

3. 讨论部分（Discussion）
   • 哪个噪声组贡献最大？M、S 还是 B？
   • Anatomy 词的角色（应该改进还是不需要投影？）
   • 结构指标如何支持"拓扑分离"的论证？

【下一步操作】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 等待 BERTopic 完成所有 4 个版本：
   python run_ablation_step07.py

2. 计算结构指标：
   python compute_ablation_structural_metrics.py

3. 生成最终对比表：
   python generate_ablation_comparison.py

4. 复制对比表到论文中

【当前文件清单】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

# 列出已生成的文件
files_to_check = [
    ("ablation_outputs", "*"),
    ("07_topic_models/ABLATION_*", "*"),
    ("06_denoised_data", "helicobacter_pylori_topic_modeling_*.csv"),
]

print("\n✓ 已生成的关键文件：")
print(f"  ablation_outputs/")
for version in versions:
    vector_file = Path(f"ablation_outputs/{version}/c_{version}_final_clean_vectors.npz")
    if vector_file.exists():
        size_mb = vector_file.stat().st_size / (1024*1024)
        print(f"    {version}/c_{version}_final_clean_vectors.npz ({size_mb:.1f} MB)")

print(f"\n  07_topic_models/")
for version in versions:
    output_dir = Path(f"07_topic_models/ABLATION_{version}")
    if output_dir.exists():
        files = list(output_dir.glob("*_topic_info.csv"))
        if files:
            print(f"    ABLATION_{version}/  ✓ 已完成")
        else:
            print(f"    ABLATION_{version}/  ⏳ 进行中")

print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

完成进度：{len([s for s in bert_status if s.get('状态')=='完成'])}/{len(versions)} 个版本的 BERTopic 处理

预计总时间：2-3 小时（每个版本 ~30 分钟）

""")
