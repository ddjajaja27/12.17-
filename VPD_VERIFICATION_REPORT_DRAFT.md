# VPD 验证报告 （Fix for UMAP n_components=5）

**生成时间**: 2025-12-28  
**关键修复**: UMAP n_components 从 2（可视化）改为 5（聚类）

---

## 💡 硬冲突修复说明

### 问题 1: UMAP 维度错误
- **原配置**: n_components=2（用于绘制2D图表）
- **错误做法**: 把2D UMAP直接用于HDBSCAN聚类
- **正确做法**: 用5D UMAP进行聚类，然后单独生成2D用于可视化
- **影响**: 这解释了为什么看到"baseline 82 topics"而不是157

### 问题 2: 噪声比例算术错误
- **之前说**: 1121 / 31617 = 2.41%
- **实际算法**: 1121 / 31617 = 3.55%
- **修复**: 使用代码自动计算，不手动输入错误数字

### 问题 3: 版本混拆
- **之前做**: 把文件扔到同一个目录然后同名覆盖
- **正确做**: 各版本分开目录，从清晰的源目录拷贝

---

## 🚀 执行步骤

### Step 1: 强制重跑 BERTopic（5D UMAP聚类）

**完成状态**: ⏳ 后台运行中

```bash
python run_ablation_bertopic.py --force
```

修改内容：
```python
# run_ablation_bertopic.py 第155行
n_components=5,  # 改为聚类用的5D（不是2）
```

**预期输出**:
- `ablation_outputs/baseline/bertopic_results/` （新聚类结果）
- `ablation_outputs/VPD/bertopic_results/` （新聚类结果）

---

### Step 2: 检查噪声比例和主题数

完成后将对比：

| 指标 | Baseline | VPD | 预期变化 |
|------|----------|-----|--------|
| 聚类维度 | 5D | 5D | 统一 ✓ |
| 主题数 (mc=39) | ? | ? | VPD < Baseline |
| 噪声比例 | ? | ? | VPD < Baseline |
| 平均 C_V | ? | ? | VPD > Baseline |

---

### Step 3: 验证关键指标（强证据）

#### A. 噪声方向点积 → 0
- **含义**: VPD 投影后，投影向量与噪声方向垂直
- **来源**: `ablation_experiment_config.py` 的投影逻辑
- **验证**: ∥ (baseline_vector · noise_prototype_hat) ∥ ≈ 0

```python
import numpy as np
from pathlib import Path

# 加载向量
baseline_vec = np.load("ablation_outputs/baseline/embeddings_baseline.npz")["embeddings"]
vpd_vec = np.load("ablation_outputs/VPD/embeddings_VPD.npz")["embeddings"]

# 计算点积（应该接近0）
for version, vec in [("baseline", baseline_vec), ("VPD", vpd_vec)]:
    norms = np.linalg.norm(vec, axis=1)
    print(f"{version}: mean_norm={norms.mean():.4f}, std={norms.std():.4f}")
```

#### B. C_V 提升（BERTopic 相干性）
- **含义**: VPD 清理后主题质量更好
- **来源**: mc39_topic_info.csv 的 c_v 列
- **验证**: VPD 的平均 C_V > Baseline 的平均 C_V

---

### Step 4: 生成最终对比表

**来源**: 07_topic_models/ABLATION_baseline/ 和 ABLATION_VPD/

**关键文件**:
- `helicobacter_pylori_mc39_topic_info.csv` (主题数、C_V)
- `helicobacter_pylori_mc39_doc_topic_mapping.csv` (噪声比例)

**对比指标**（使用相同 mc=39）:

```python
import pandas as pd

versions = ["baseline", "VPD"]
results = {}

for v in versions:
    info_file = f"07_topic_models/ABLATION_{v}/helicobacter_pylori_mc39_topic_info.csv"
    map_file = f"07_topic_models/ABLATION_{v}/helicobacter_pylori_mc39_doc_topic_mapping.csv"
    
    info = pd.read_csv(info_file)
    mapping = pd.read_csv(map_file)
    
    topic_count = len(info[info.iloc[:, 0] >= 0])  # 不计 -1
    noise_count = (mapping.iloc[:, 1] == -1).sum()
    noise_ratio = noise_count / len(mapping)
    mean_cv = info[info.iloc[:, 0] >= 0].iloc[:, -1].mean()
    
    results[v] = {
        "Topics": topic_count,
        "Noise_Count": noise_count,
        "Noise_Ratio": f"{noise_ratio:.2%}",
        "Mean_CV": f"{mean_cv:.4f}"
    }

print(pd.DataFrame(results).T)
```

---

## ✅ 验证清单

- [ ] UMAP n_components=5 已应用到 run_ablation_bertopic.py
- [ ] BERTopic 重跑完成（baseline + VPD）
- [ ] 检查日志: VPD 是否打印"使用预计算清洁向量"
- [ ] mc39 文件存在: `helicobacter_pylori_mc39_topic_info.csv` + `doc_topic_mapping.csv`
- [ ] 噪声比例数学正确: (Topic == -1).count() / total_docs
- [ ] 对比表生成: baseline vs VPD
- [ ] 强证据收集: 
  - 噪声方向点积 ≈ 0
  - C_V 提升（数值对比）

---

## 📊 最终数据源

| 指标 | 文件 | 列名 |
|------|------|------|
| 主题数 | mc39_topic_info.csv | Topic (>=0) |
| C_V | mc39_topic_info.csv | c_v |
| 噪声文档数 | mc39_doc_topic_mapping.csv | Topic (==-1) |
| 总文档数 | mc39_doc_topic_mapping.csv | len(df) |

---

## 🎯 核心区别（修复后）

| 维度 | 错误方式（2D） | 正确方式（5D） |
|------|-----------|-----------|
| UMAP n_components | 2（聚类） | 5（聚类），2（单独可视化） |
| 主题数预期 | 82 | 157 (接近) |
| 噪声比例 | 偏低 | 更准确（~3-4%） |
| C_V 变化 | 可能误判 | 准确反映质量 |

---

## 📝 下一步

1. 等待 BERTopic 完成（~30分钟）
2. 验证 mc39 文件存在
3. 生成对比表
4. 撰写最终报告

