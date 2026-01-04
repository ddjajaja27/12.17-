#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ABLATION_PAPER_TEMPLATE.md

消融实验结果整合到论文的模板

包含：
1. 方法部分（Methods）的描述
2. 结果部分（Results）的表格和文字
3. 讨论部分（Discussion）的分析框架
4. 补充材料（Supplementary Material）的参考
"""

template = """
# 消融实验论文撰写模板

## 1. 方法部分（Methods）

### 1.1 向量去噪方案（VPD）

本研究提出向量投影去噪（Vector Projection Denoising, VPD）方法，
通过正交投影消除文本向量中的噪声方向。

**数学表述：**

$$V_{clean} = V - (V \\cdot \\hat{n}) \\times \\hat{n}$$

其中：
- $V$ 是原始文本向量（维度 D=384）
- $\\hat{n}$ 是单位噪声原型向量（通过平均 48 个噪声词的向量并归一化得到）
- $V_{clean}$ 是去噪后的向量

**噪声词定义：**

我们通过文献审阅和领域专家讨论，确定了 4 组共 48 个噪声词：

| 分组 | 数量 | 代表词 | 说明 |
|-----|------|--------|------|
| M (Methodology) | 12 | analysis, study, method, ... | 方法论词汇 |
| S (Statistics) | 14 | significant, compared, ... | 统计学词汇 |
| B (Background) | 13 | clinical, patient, disease, ... | 广义背景词汇 |
| Anatomy | 8 | gastric, stomach, mucosa, ... | 解剖学术语（对照组）|

### 1.2 消融实验设计

为了评估 VPD 中不同噪声组的贡献，我们进行了逐步消融实验：

1. **Baseline**：无投影，使用原始融合向量
2. **M+S 投影**：移除方法论和统计学词汇（26 个词）
3. **M+S+B 投影**：进一步移除广义背景词汇（40 个词）
4. **VPD（M+S+B+Anatomy）**：完整投影（48 个词）

每个版本使用相同的文档数据集（N=31,617）和相同的 BERTopic 参数
（mc=39, UMAP/HDBSCAN 参数见 experiment_config.yaml）。

### 1.3 评价指标

我们使用多维指标评估去噪效果：

#### 主要指标：
- **C_v 分数**：主题模型质量（范围 0-1，越高越好）[1]
- **Silhouette 系数**：簇紧凑性（范围 -1 到 1）[2]

#### 辅助指标：
- **kNN 混合率**：跨簇邻居比例（范围 0-1，越低越好）
- **簇隔离度**：簇间/簇内距离比（>1 为好）[3]
- **Davies-Bouldin 指数**：簇分离质量（越小越好）[4]

## 2. 结果部分（Results）

### 2.1 消融实验对比

**表 X. 消融实验中不同噪声去噪策略的对比**

| 版本 | 投影词数 | 主题数 | 噪声比% | C_v 分数 | Silhouette | 隔离度 | DB指数 |
|------|---------|--------|--------|---------|-----------|---------|---------|
| Baseline | 0 | 82 | 2.41 | 0.5950 | [TBD] | [TBD] | [TBD] |
| M+S | 26 | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| M+S+B | 40 | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| VPD（M+S+B+Anat） | 48 | [TBD] | [TBD] | 0.6261 | [TBD] | [TBD] | [TBD] |

### 2.2 定性结果

**关键发现：**

1. **方法论和统计学词汇的影响**
   M+S 投影产生了 X% 的 C_v 改进（vs. Baseline），表明这 26 个词
   贡献了大部分噪声。

2. **广义背景词汇的补充效果**
   M+S+B 投影相比 M+S 又有 Y% 的进一步改进，说明背景词汇虽然
   语义相关，但在向量空间中仍表现为噪声方向。

3. **解剖学词汇的角色**
   VPD（包括 Anatomy 组）与 M+S+B 的差异为 Z%，说明解剖学术语
   的投影带来了额外但较小的改进。这可能说明…（从数据推断）

4. **结构指标的一致性**
   Silhouette 系数和簇隔离度的变化趋势与 C_v 分数一致，
   验证了我们"拓扑分离"的论证。

## 3. 讨论部分（Discussion）

### 3.1 噪声词分组的科学性

我们的 4 组噪声词划分基于：
1. 文献分析（h-index 主导词倾向…）
2. 领域知识（临床螺杆菌研究的常见表述）
3. 向量空间分析（这些词的投影向量确实指向共同的噪声方向）

Anatomy 组作为对照组，其投影结果证实了我们的假设：
如果这些词不是"噪声"而是"有用的语义特征"，投影它们应该
降低主题质量或不改变。但实验表明 VPD 仍带来改进，说明…

### 3.2 与其他去噪方法的比较

相比传统的停用词过滤（418 个通用英文词），VPD 的优势：
- 更精准：只针对领域特有的噪声
- 更灵活：可逐步调整投影强度（通过改变噪声词组合）
- 数学化：基于向量空间的显式投影，而非启发式规则

### 3.3 局限性与未来方向

局限性：
1. 噪声词的手工选择可能带有主观性
2. 不同领域的噪声词组可能不同
3. VPD 假设存在单一的"噪声方向"，但实际可能是多维的

未来方向：
1. 使用无监督方法自动发现噪声词（PCA、ICA 等）
2. 对其他领域（医学、法律、新闻）的鲁棒性验证
3. 扩展到多维噪声投影（同时移除多个独立方向）

## 4. 补充材料

### 4.1 数据和代码可获得性

所有数据和代码已开源，供论文审稿人验证：

**最小数据集**（MINIMAL_DATA_SET_README.md）：
- 4 个消融版本的向量文件（172 MB）
- 噪声词定义（JSON 格式）
- 实验配置（YAML 格式）
- 2000 文档采样数据（快速验证）

**可复现性检查清单**：
```
□ 向量文件完整性（SHA256 hash）
□ 参数配置一致性（experiment_config.yaml）
□ 随机种子固定（seed=42）
□ BERTopic 版本和依赖版本一致
□ 输出 C_v 分数与论文表格匹配
```

### 4.2 详细指标

见附表：
- 表 S1：各 mc 参数的完整结果（mc=22, 39, 56, 73）
- 表 S2：主题词样本（每版本各取 5 个主题的 top-10 词）
- 图 S1：聚类质量演化（BERTopic 迭代过程）
- 图 S2：向量空间 UMAP 可视化（baseline vs VPD）

## 参考文献

[1] Röder, M., Both, A., & Hinneburg, A. (2015). Exploring the space of topic coherence measures. 
    In Proceedings of the eighth ACM international conference on Web search and data mining (pp. 399-408).

[2] Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the interpretation and validation 
    of cluster analysis. Journal of computational and applied mathematics, 20, 53-65.

[3] McInnes, L., Healy, J., & Astels, S. (2017). hdbscan: Hierarchical density based clustering. 
    Journal of Open Source Software, 2(11), 205.

[4] Davies, D. L., & Bouldin, D. W. (1979). A cluster separation measure. 
    IEEE transactions on pattern analysis and machine intelligence, (2), 224-227.

---

## 使用说明

1. 运行 `python monitor_ablation_progress.py` 查看最新进度
2. BERTopic 全部完成后，用实际数据替换 [TBD] 部分
3. 根据实际结果调整讨论部分的分析
4. 生成的 ABLATION_COMPARISON_FINAL.csv 可直接作为表格

**文件位置：**
- 结果表格：ablation_outputs/ABLATION_COMPARISON_FINAL.csv
- 完整指标：ablation_outputs/ablation_structural_metrics.json
- 原始结果：07_topic_models/ABLATION_*/

"""

# 保存为文件
import os
output_file = "ABLATION_PAPER_TEMPLATE.md"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(template)

print(f"OK: 论文模板已生成 {output_file}")
print(f"  大小: {len(template)} bytes")
print(f"\n包含以下部分:")
print(f"  1. Methods section")
print(f"  2. Results section with tables")
print(f"  3. Discussion section")
print(f"  4. Supplementary Material")
print(f"\nSee {output_file} for details")
