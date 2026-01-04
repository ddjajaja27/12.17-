# 最小数据集：供审稿人验证 VPD 链路的代码和数据

## 📋 概述

这个最小集合包含所有必要的代码和数据，可以让审稿人独立验证：
1. 噪声词的定义和分组
2. 向量是否被正确处理
3. 投影计算是否数学正确
4. 消融实验的四个版本是否可重复

---

## 📁 文件清单

### 脚本（代码链路验证）

```
ablation_experiment_config.py
  ├─ NoiseWordGroups: 噪声词分组定义（M/S/B/Anatomy）
  └─ ABLATION_CONFIGS: 4 个实验版本的配置

run_ablation_experiments.py
  ├─ load_raw_embeddings(): 加载原始向量
  ├─ build_noise_prototype(): 构建噪声方向
  ├─ project_vectors(): 正交投影实现
  └─ run_single_ablation(): 单个版本的完整流程

compute_structural_metrics.py
  ├─ compute_silhouette(): 簇紧凑性
  ├─ compute_knn_mixing(): 跨簇邻居比例
  └─ compute_cluster_isolation(): 簇隔离度

experiment_config.py
  ├─ ExperimentConfig: 统一参数配置
  ├─ create_experiment_manifest(): 记录元数据
  └─ print_vector_loading_checklist(): 向量链路检查
```

### 数据

```
noise_words_groups.json
  └─ 4 个噪声词分组（M/S/B/Anatomy），总 48 词

ablation_outputs/
  ├─ baseline/
  │  ├─ embeddings_baseline.npz (31617 × 384)
  │  └─ experiment_manifest.json
  ├─ M_S/
  │  ├─ embeddings_M_S.npz (投影 M+S，26 词)
  │  └─ experiment_manifest.json
  ├─ M_S_B/
  │  ├─ embeddings_M_S_B.npz (投影 M+S+B，40 词)
  │  └─ experiment_manifest.json
  ├─ M_S_B_Anatomy/
  │  ├─ embeddings_M_S_B_Anatomy.npz (全部投影，48 词)
  │  └─ experiment_manifest.json
  │
  ├─ sample_embeddings_raw.npz
  │  └─ 2000 条采样（原始融合向量）
  ├─ sample_embeddings_clean.npz
  │  └─ 2000 条采样（完全去噪后）
  └─ ablation_summary.json
     └─ 4 个版本的投影效果汇总
```

---

## 🔍 如何验证

### 1. 验证噪声词定义

```bash
# 查看噪声词分组
cat noise_words_groups.json
```

期望：
- Group M: 12 个方法论词（analysis, study, method...）
- Group S: 14 个统计词（significant, compared...）
- Group B: 13 个背景词（clinical, patient...）
- Group Anatomy: 8 个解剖词（gastric, stomach...）
- 总计 47 词（有 1 个可能重复）

### 2. 验证向量处理链路

```bash
# 运行向量投影
python run_ablation_experiments.py

# 输出应该包括：
#   ✓ 4 个向量文件已生成
#   • baseline: 原始融合向量（无投影）
#   • M_S: 投影 M+S（26 词）
#   • M_S_B: 投影 M+S+B（40 词）
#   • M_S_B_Anatomy: 投影全部（48 词）
```

### 3. 验证投影数学正确性

```bash
# 加载采样向量并验证
python -c "
import numpy as np

# 加载原始和去噪向量
raw = np.load('ablation_outputs/sample_embeddings_raw.npz')
clean = np.load('ablation_outputs/sample_embeddings_clean.npz')

raw_vec = raw['embeddings']
clean_vec = clean['embeddings']

# 验证向量统计
print('原始向量范数均值:', np.linalg.norm(raw_vec, axis=1).mean())
print('去噪向量范数均值:', np.linalg.norm(clean_vec, axis=1).mean())

# 检查是否有 NaN
print('原始有 NaN:', np.any(np.isnan(raw_vec)))
print('去噪有 NaN:', np.any(np.isnan(clean_vec)))

# 向量移位量
shift = np.linalg.norm(clean_vec - raw_vec, axis=1).mean()
print('向量平均移位:', shift)
"
```

期望：
- 所有向量范数都接近 1.0（已归一化）
- 没有 NaN 或无穷大
- 去噪向量与原始向量的移位不太大（投影只改变方向）

### 4. 验证实验可复现性

```bash
# 查看 manifest（每个版本都有）
cat ablation_outputs/baseline/experiment_manifest.json

# 检查关键元数据：
#   • embedding_file: 向量文件路径
#   • file_hash_sha256: 文件完整性检查
#   • noise_words: 使用的噪声词列表
#   • random_state: 随机种子（保证可重复）
#   • sample_vectors_first_5_elements: 前 5 个向量的前 5 个元素
```

期望：
- 每个版本都有明确的 hash
- noise_words 列表与 noise_words_groups.json 一致
- 所有版本使用相同的 random_state (42)

---

## 🧮 消融实验设计

### 四个版本

| 版本 | 投影的词 | 词数 | 目标 |
|------|---------|------|------|
| Baseline | （无） | 0 | Control |
| M+S | 方法论 + 统计 | 26 | 核心背景噪声 |
| M+S+B | +背景词 | 40 | 包括宽泛噪声 |
| M+S+B+Anatomy | +解剖词 | 48 | 完整去噪（当前版本） |

### 关键问题

1. **M+S (26 词) 能做到多少效果？**
   - 如果 M+S 已经达到 +3% C_v，说明核心噪声在这里
   - 如果 M+S 没有效果，说明背景词（B）很关键

2. **Anatomy 词应该投影吗？**
   - 如果全部投影 vs M+S+B 性能差不多，说明 Anatomy 词是有益的，不该删
   - 如果全部投影明显更差，说明 Anatomy 词有重要信息

3. **哪个分组贡献最大？**
   - 通过对比四个版本的 C_v，分解各分组的贡献

---

## 📊 对比指标

### 已有指标（常用）
- **C_v 一致性**（范围 [0,1]，越高越好）
- **Topic=-1 噪音比例**

### 新增指标（论文需要）
- **Silhouette coefficient**（-1 ~ 1，1 最优）
  - 衡量样本与自己的簇的相似性
- **kNN mixing**（0 ~ 1，0 最优）
  - k 个最近邻中，有多少来自其他簇
  - 纯度 = 1 - mixing
- **簇隔离度**（ratio，> 1 最优）
  - 簇间距离 / 簇内距离
- **拓扑稳定性**（新增 seed 后结果是否一致）

---

## 🎯 使用说明（给审稿人）

1. **快速验证链路**
   ```bash
   # 1. 检查噪声词定义
   cat noise_words_groups.json
   
   # 2. 检查 4 个向量文件是否存在
   ls -lh ablation_outputs/*/embeddings_*.npz
   
   # 3. 验证向量统计信息
   python -c "import numpy as np; ..." (上面的代码)
   
   # 4. 查看 manifest
   cat ablation_outputs/baseline/experiment_manifest.json
   ```

2. **重现消融实验**
   ```bash
   # 重新运行向量投影（应该得到相同结果）
   python run_ablation_experiments.py
   
   # 对比生成的 hash 值是否相同
   ```

3. **计算结构指标**（需要完整 BERTopic 结果）
   ```bash
   python compute_structural_metrics.py
   ```

---

## 🔐 数据完整性

每个向量文件都有：
- **File hash**: SHA256 校验码（防篡改）
- **Vector sample**: 前 5 条向量的前 5 个元素（防混淆）
- **Metadata**: 使用的参数和随机种子（防疑惑）

---

## 📝 论文引用建议

> "为保证可复现性，我们提供了：
> 1. 噪声词的完整分组定义（noise_words_groups.json）
> 2. 四个消融实验的向量文件（ablation_outputs/）
> 3. 每个实验的元数据清单（experiment_manifest.json）
> 4. 采样数据供验证（sample_embeddings_*.npz）
> 
> 审稿人可以独立验证向量处理链路的正确性。"

---

## ⚠️ 常见问题

**Q: 为什么需要这么多数据？**  
A: 因为"感觉性能改进"可能来自参数变化、缓存、seed 差异等。有了这个最小集合，可以排除所有这些因素。

**Q: 4 个向量文件 43M 很大吗？**  
A: 不大。31,617 篇文档 × 384 维 × 4 字节 ≈ 48 MB。补充材料可以承受。

**Q: 需要提供 BERTopic 的完整结果吗？**  
A: 不必。只需要聚类标签（topic assignments）和 C_v 分数。

**Q: 采样 2000 条够吗？**  
A: 够。用来验证向量统计性质和数值稳定性。完整 31617 条用于正式实验。

---

## 📞 如何使用这个集合

1. **自己验证时**：跑一遍 run_ablation_experiments.py，检查输出
2. **给审稿人时**：把整个 ablation_outputs/ 目录放在补充材料里
3. **论文里引用**：在 Methods 里说"见补充材料 Table X 和 File Y"
4. **公开发布时**：上传到 GitHub/Zenodo，做 long-term archival
