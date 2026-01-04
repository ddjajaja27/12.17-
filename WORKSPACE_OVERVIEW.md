# 工作区结构概况（2026-01-04）

本仓库是一个“从原始检索 → 清洗 → 去噪 → 主题建模 → 选模 → 可视化/报告/论文”的流水线目录；其中 **reproducible_pipeline/** 用来固定主流程输入输出（避免目录串用）。

## 1) 顶层目录（你最常用的）

### A. 数据流水线目录（01~12）
- 01_raw_data/：原始抓取/检索结果（最上游输入）
- 02_citations_data/：补充引用信息后的数据
- 03_cleaned_data/：清洗后的数据
- 04_filtered_data/：按类型/规则过滤后的数据与过滤日志
- 05_stopwords/：停用词、去噪/删词方案与相关 manifest
- 06_denoised_data/：主题建模输入表（每个方法一份 `*_topic_modeling_<method>.csv`）
- 07_topic_models/：BERTopic 产物（按方法分目录；每个 mc 一套 `*_topic_info.csv`/`*_frontier_indicators.csv`/`*_doc_topic_mapping.csv` 等）
- 08_model_selection/：选模结果（`best_mc_by_method.json`、评分轨迹 `cv_scores_full.json`、报告 `cv_select_best_mc_report.txt`）
- 09_visualization/：可视化图表输出
- 10_report/：自动化报告输出（Markdown/HTML/DOCX 等）
- 10_report_auto_fix_backups/：报告自动修复备份（可忽略，除非追溯）
- 11_chinese/：中文本地化输出
- 12_top_journal_upgrade/：期刊风格升级/版式相关脚本与输出

### B. “可复现主流程”与审计/回滚
- reproducible_pipeline/：主流程工作目录与指针（建议后续所有步骤都基于这里跑）
  - MAIN_WORKDIR.txt：当前主流程 workdir 指针
  - main_workdir_legacyB11_VPD34_.../：一次完整可复现的工作目录快照（包含 06~10 等关键子目录）
- audit_bundle_20260104_1700/：审计包（把“支撑论文结论的源文件 + 生成代码 + 映射说明”集中在一起）
- rollback_outputs/：旧 baseline 噪声回滚复跑的独立输出（避免覆盖主流程）

### C. 其它
- tools/：工具脚本（打包、核验、导出证据、回滚等）
- paper_evidence/：论文证据导出（也会在 main_workdir 内有一份）
- ablation_outputs/：消融实验输出
- backups/：历史备份杂项
- html_报告/、word_报告/：导出的 HTML/Word 报告归档
- .venv/：Python 虚拟环境

## 2) 主流程（你现在应该“以哪个目录为准”）

当前主流程建议以 `reproducible_pipeline/MAIN_WORKDIR.txt` 指向的 workdir 为准。
- 该 workdir 结构通常是：
  - 06_denoised_data/ → 07_topic_models/ → 08_model_selection/ → 09_visualization/ → 10_report/（+ paper_evidence/）

## 3) 常用入口脚本（按步骤）
- Step07：step07_topic_model.py（调用 _engine_bertopic.py 写出 07_topic_models 的 CSV/模型）
- Step08：step08_cv_select.py（生成 08_model_selection 的 best_mc 与评分轨迹）
- Step09：step09_visualization.py
- Step10：step10_report.py
- Step12：step12_manuscript.py

## 4) 快速定位“论文结论用到哪些文件”

最关键的 3 类文件（都在主 workdir 的 07/08 下）：
- `07_topic_models/<METHOD>/*_topic_info.csv`：主题词/Topic_Label/Count（用来做关键词命中与规模）
- `07_topic_models/<METHOD>/*_frontier_indicators.csv`：Heat/Novelty 等指标（用来做榜单排名）
- `08_model_selection/best_mc_by_method.json` + `cv_scores_full.json`：best_mc、noise_ratio、C_v 与完整评分轨迹

如需“打包审计”，直接看：`audit_bundle_20260104_1700/mapping/AUDIT_MAPPING.md`。
