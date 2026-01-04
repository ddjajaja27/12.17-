# 回滚复跑（复现旧噪音 11.21%）

目标：在不覆盖当前主流程输出的前提下，复现“旧时期 baseline 噪音比例 ≈ 11.21%（3545/31617）”并生成对应 Step08/Step10 产物。

## 背景
- 旧噪音 11.21% 对应的 Step07 baseline 产物保存在：
  - `07_topic_models/BASELINE_污染备份/`
- 该目录内的 `helicobacter_pylori_mc39_doc_topic_mapping.csv` 统计结果为：
  - `noise_docs=3545`，`rows=31617`，`noise_ratio=0.1121`

## 一键回滚复跑
在项目根目录运行：

- 仅生成 Step08 + Step10（默认只回滚 baseline）：
  - `\.venv\Scripts\python.exe tools\run_rollback_old_noise.py`

- 同时生成 DOCX（需要 pandoc）：
  - `\.venv\Scripts\python.exe tools\run_rollback_old_noise.py --docx`

输出会写到：
- `rollback_outputs/old_noise_20251222_<timestamp>/`

其中包括：
- `07_topic_models/BASELINE/`（从 `BASELINE_污染备份` 复制而来）
- `08_model_selection/`（Step08 重新计算得到）
- `10_report/BASELINE/`（Step10 重新生成的 MD/HTML，可选 DOCX）
- `rollback_verify.json`（噪音核验与 best_mc 记录）

## 关键实现点（为“可复现”服务）
- Step08/Step10 已支持 `--base_dir`：可以对任意“工作目录”运行，避免覆盖主输出。
- 回滚运行默认 `best_mc=39`（由回滚目录内 Step08 重新选择得到，当前已验证一致）。

## 结果校验
以一次已完成的回滚输出为例：
- `rollback_outputs/old_noise_20251222_20260104_162448/rollback_verify.json`
- 核验项包含：`best_mc` 与 `noise_docs/noise_ratio`。
