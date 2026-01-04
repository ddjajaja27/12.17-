# 12.17.幽门螺杆菌：文件夹总览（对外说明版）

这个文件夹包含一套从“文献数据 → 清洗/去噪 → 主题建模 → 选模评估 → 可视化/报告/论文”的完整研究流水线，以及对应的可复现快照、审计与实验输出。

## 1) 主流水线（按步骤分目录）
- 01_raw_data/：原始检索/抓取的最上游数据
- 02_citations_data/：补充引用信息后的数据
- 03_cleaned_data/：清洗后的数据
- 04_filtered_data/：过滤后的数据与过滤日志
- 05_stopwords/：停用词与去噪方案配置/产物
- 06_denoised_data/：主题建模输入表（每个方法一份 CSV）
- 07_topic_models/：主题建模输出（按方法与参数版本保存模型与指标表）
- 08_model_selection/：模型选择与评分轨迹（best_mc、C_v、噪声比例、候选 mc 全表）
- 09_visualization/：图表与可视化输出
- 10_report/：自动化研究报告（Markdown/HTML/Word 等）
- 11_chinese/：中文本地化输出
- 12_top_journal_upgrade/：期刊风格升级与版式相关输出

## 2) 可复现与防“串目录”机制（建议他人从这里理解结果）
- reproducible_pipeline/：可复现的主流程快照目录
  - MAIN_WORKDIR.txt：指向“当前主流程工作目录”的指针
  - main_workdir_.../：一次完整可复现运行的工作目录快照（包含 06~10、paper_evidence 等）

## 3) 审计/回滚/实验输出（用于核验、对比与追溯）
- audit_bundle_*/：审计包（把论文关键结论所需的源文件、生成代码、映射说明集中到一处）
- rollback_outputs/：回滚复跑输出（通常用于恢复历史 baseline 口径，不覆盖主流程）
- ablation_outputs/：消融实验输出
- backups/：历史备份

## 4) 脚本与说明文档
- step01_*.py ~ step12_*.py：按步骤执行的主脚本
- tools/：辅助工具脚本（核验、导出证据、打包审计等）
- README.md / README_FULL.md / 各类 *.md：项目说明、执行清单与报告模板

## 5) 给新读者的“最快入口”
1. 看 reproducible_pipeline/MAIN_WORKDIR.txt 指向哪个 main_workdir
2. 在该 main_workdir 的 07_topic_models/ 与 08_model_selection/ 中查看核心结果（主题信息表、Heat/Novelty 指标、best_mc、C_v、噪声比例与评分轨迹）
3. 需要严格核验时查看 audit_bundle_*（已把关键文件与代码固定打包）
