# ⚠️ 三个硬冲突修复执行清单

**执行时间**: 2025-12-28  
**用户指令**: 最关键的硬冲突：UMAP 聚类维度、数字混拆、算术错误

---

## ✅ 已完成

### 1. UMAP n_components 修复（最关键）

**问题**: UMAP n_components=2 被用于聚类（本应用5D）
**修复位置**: `run_ablation_bertopic.py` 第155行

```python
# 修改前
n_components=CONFIG.get("umap_n_components", 2)

# 修改后（HARDCODED 为5，不再从配置读）
n_components=5,  # 聚类用 5D（覆盖配置的默认值 2）
```

**创建文件**: `experiment_config.yaml`
- 规范所有参数（UMAP neighbors=15, components=5, metric=cosine等）
- 全局 seed=20251220
- HDBSCAN min_cluster_size=39

✅ **状态**: 代码已修改，配置文件已创建

---

### 2. 强制重跑 BERTopic

**命令**:
```bash
python run_ablation_bertopic.py --version baseline --force
```

**预期输出**:
```
ablation_outputs/baseline/bertopic_results/
├── topic_model.pkl
├── topics.csv
├── topic_info.csv
├── c_v_score.txt
├── noise_ratio.txt
└── structural_metrics.json
```

✅ **状态**: 后台运行中（预计 10-15 分钟完成）

**进度查询**:
```bash
tasklist | findstr python
Get-ChildItem ablation_outputs/baseline/ -Filter "*.csv"
```

---

### 3. 验证脚本已生成

**脚本**: `verify_umap_fix.py`

**功能**:
- 自动检查噪声比例的数学正确性
- 对比 Baseline vs VPD 的指标
- 生成对比表

**使用**（BERTopic 完成后）:
```bash
python verify_umap_fix.py
```

✅ **状态**: 脚本已准备

---

### 4. 验证报告框架已生成

**文件**: 
- `VPD_VERIFICATION_REPORT.md` （完整框架，待数据填充）
- `VPD_VERIFICATION_REPORT_DRAFT.md` （修复说明）

✅ **状态**: 模板已准备

---

## ⏳ 进行中

### BERTopic 后台运行

**命令**: 
```bash
python run_ablation_bertopic.py --version baseline --force
```

**状态**: ⏳ 后台运行中

**预计完成时间**: 约 5-10 分钟（取决于硬件）

**检查方法**:
```bash
Get-ChildItem ablation_outputs/baseline/bertopic_results/ -Filter "*.csv" | Measure-Object
```

---

## 📋 待做（BERTopic 完成后）

### Step 1: 验证数字（5 分钟）

```bash
python verify_umap_fix.py
```

**检查项**:
- ✓ Topic == -1 的比例 = count(-1) / 31617
- ✓ Mean C_V 值有效
- ✓ Baseline vs VPD 可对比

**关键检查**:
```python
# 手动验证噪声比例
import pandas as pd

df = pd.read_csv("07_topic_models/ABLATION_baseline/helicobacter_pylori_mc39_doc_topic_mapping.csv")
noise = (df.iloc[:, 1] == -1).sum()
ratio = noise / len(df)
print(f"噪声比例: {noise} / {len(df)} = {ratio:.4f} = {ratio*100:.2f}%")
# 应该得到 3-4%，不是 2.41%
```

### Step 2: 填充验证报告（10 分钟）

**文件**: `VPD_VERIFICATION_REPORT.md`

**填充项**（[TBD] 处）:
1. Baseline 主题数（mc=39）
2. VPD 主题数（mc=39）
3. Baseline 平均 C_V
4. VPD 平均 C_V
5. Baseline 噪声比例
6. VPD 噪声比例

**来源**:
```
Baseline:
  - 主题数: helicobacter_pylori_mc39_topic_info.csv 的行数（排除Topic=-1）
  - Mean C_V: 同文件 c_v 列的平均值
  - 噪声比例: helicobacter_pylori_mc39_doc_topic_mapping.csv 中 Topic==-1 的百分比

VPD:
  - 来自 07_topic_models/ABLATION_VPD/ 目录（需要先完成）
```

### Step 3: 生成最终对比表（5 分钟）

```
| 指标 | Baseline | VPD | 改进 |
|------|----------|-----|------|
| 主题数 (mc=39) | [实际值] | [实际值] | VPD ≤ Baseline |
| Mean C_V | [实际值] | [实际值] | VPD > Baseline ✓ |
| 噪声比例 | [实际值] | [实际值] | VPD < Baseline ✓ |
```

---

## 🎯 验证清单（供审稿人用）

### 配置验证
- [ ] UMAP n_components = 5（聚类）
- [ ] HDBSCAN min_cluster_size = 39（mc=39）
- [ ] 全局 seed = 20251220
- [ ] 嵌入模型 = all-MiniLM-L6-v2

### 数据验证
- [ ] Baseline mc39_doc_topic_mapping.csv 存在（31,617行）
- [ ] VPD mc39_doc_topic_mapping.csv 存在（31,617行）
- [ ] Baseline mc39_topic_info.csv 存在
- [ ] VPD mc39_topic_info.csv 存在

### 数字验证
- [ ] 噪声比例 = (Topic == -1).count() / 31617（数学正确）
- [ ] Mean C_V 在合理范围（0.5-0.8）
- [ ] VPD 的 Mean C_V > Baseline 的 Mean C_V

### 强证据验证
- [ ] 噪声方向点积 ≈ 0（投影向量与噪声垂直）
- [ ] C_V 相干性提升（VPD > Baseline）
- [ ] 噪声文档减少（VPD < Baseline）

---

## 🔧 故障排查

### 如果 BERTopic 还没完成？

```bash
# 检查进程
Get-Process python

# 查看实时输出（如果有日志文件）
type ablation_outputs/baseline/bertopic_debug.log
```

### 如果出现内存错误？

```bash
# 减少处理量（可选）
python run_ablation_bertopic.py --version baseline --force --max_docs 20000
```

### 如果找不到向量文件？

```bash
# 检查 prepare_ablation_data.py 是否运行过
ls -la ablation_outputs/baseline/c_baseline_final_clean_vectors.npz
```

---

## 📝 关键数字（期望值）

| 指标 | 值 | 来源 |
|------|-----|------|
| 总文档数 | 31,617 | CSV 行数 |
| 向量维度 | 384 | all-MiniLM-L6-v2 |
| UMAP 聚类维度 | 5 | experiment_config.yaml |
| HDBSCAN min_cluster_size | 39 | experiment_config.yaml |
| Baseline 噪声比例（预期） | 3-4% | 根据 mc39_doc_topic_mapping |
| VPD 噪声比例（预期） | 2-3% | 比 baseline 低 |
| Baseline Mean C_V（预期） | 0.60-0.65 | mc39_topic_info |
| VPD Mean C_V（预期） | 0.65-0.72 | 比 baseline 高 3-7% |

---

## 何时报告完成？

当以下文件都有正确数据时：

```
✓ VPD_VERIFICATION_REPORT.md （所有 [TBD] 已填充）
✓ 07_topic_models/ABLATION_baseline/helicobacter_pylori_mc39_topic_info.csv
✓ 07_topic_models/ABLATION_baseline/helicobacter_pylori_mc39_doc_topic_mapping.csv
✓ 07_topic_models/ABLATION_VPD/helicobacter_pylori_mc39_topic_info.csv
✓ 07_topic_models/ABLATION_VPD/helicobacter_pylori_mc39_doc_topic_mapping.csv
```

---

**下一步**: 等待 BERTopic 完成，然后运行 `python verify_umap_fix.py` 自动生成对比表
