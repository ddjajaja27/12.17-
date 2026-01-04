#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_filter_publication_types.py
第四步：文献类型过滤

功能：
1. 过滤非研究性文献（Comment, Editorial, Letter, Erratum 等）
2. 可选：删除摘要中的独立数字
3. 生成主题建模专用数据（仅保留必要列）
4. 输出过滤统计日志

输入：03_cleaned_data/{project_prefix}_cleaned.csv
输出：04_publication_type_filtered/{project_prefix}_main.csv
      04_publication_type_filtered/{project_prefix}_topic_modeling.csv
      04_publication_type_filtered/{project_prefix}_filtered_out.csv
      04_publication_type_filtered/{project_prefix}_filter_log.txt
"""

import re
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

try:
    from config import PATHS, FILTER_CONFIG, get_project_name
except ImportError:
    print("请确保 config.py 存在且配置正确")
    sys.exit(1)


class PublicationTypeFilter:
    """文献类型过滤器"""
    
    def __init__(self):
        self.input_file = Path(PATHS["file_03_cleaned"])
        self.output_dir = Path(PATHS["dir_04_filtered"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.output_main = Path(PATHS["file_04_main"])
        self.output_topic = Path(PATHS["file_04_topic"])
        self.output_filtered = Path(PATHS["file_04_filtered_out"])
        self.log_file = Path(PATHS["file_04_log"])
        
        self.exclude_types = FILTER_CONFIG.get("exclude_types", [])
        self.remove_numbers = FILTER_CONFIG.get("remove_abstract_numbers", True)
        
        self.stats = {
            "original": 0,
            "filtered_out": 0,
            "remaining": 0,
            "numbers_removed": 0,
            "type_counts": {},
        }
    
    def run(self):
        """执行文献类型过滤"""
        print("=" * 60)
        print(f"第四步：文献类型过滤 - {get_project_name()}")
        print("=" * 60)
        
        # 加载数据
        if not self.input_file.exists():
            print(f"错误：未找到输入文件 {self.input_file}")
            print("请先运行第三步数据清洗")
            return
        
        df = pd.read_csv(self.input_file, dtype={"PMID": str})
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        self.stats["original"] = len(df)
        print(f"加载数据: {len(df)} 条")

        # 先去除没有摘要的文献
        if "Abstract" in df.columns:
            no_abstract_mask = df["Abstract"].isna() | (df["Abstract"].astype(str).str.strip() == "")
            num_no_abstract = no_abstract_mask.sum()
            if num_no_abstract > 0:
                print(f"去除没有摘要的文献: {num_no_abstract} 条")
            df = df[~no_abstract_mask].copy()
        else:
            print("警告：没有 Abstract 列，无法去除无摘要文献")

        # 过滤步骤
        df_main, df_filtered = self._filter_publication_types(df)

        if self.remove_numbers:
            df_main = self._remove_abstract_numbers(df_main)

        # 生成主题建模数据
        df_topic = self._create_topic_data(df_main)

        # 保存结果
        self._save_results(df_main, df_topic, df_filtered)
        self._write_log()

        print("\n" + "=" * 60)
        print("第四步完成！")
        print(f"  原始记录: {self.stats['original']}")
        print(f"  过滤掉: {self.stats['filtered_out']}")
        print(f"  保留: {self.stats['remaining']}")
        print(f"  输出文件: {self.output_main}")
        print(f"  主题建模数据: {self.output_topic}")
        print("=" * 60)
    
    def _filter_publication_types(self, df: pd.DataFrame) -> tuple:
        """过滤指定的文献类型"""
        print(f"\n1. 过滤文献类型: {self.exclude_types}")
        
        if "Publication_Types" not in df.columns:
            print("   警告：没有 Publication_Types 列，跳过过滤")
            self.stats["remaining"] = len(df)
            return df, pd.DataFrame()
        
        # 构建过滤正则表达式
        pattern = "|".join(re.escape(t) for t in self.exclude_types)
        mask = df["Publication_Types"].str.contains(pattern, na=False, case=False)
        
        # 统计各类型被过滤的数量
        for pub_type in self.exclude_types:
            count = df["Publication_Types"].str.contains(pub_type, na=False, case=False).sum()
            if count > 0:
                self.stats["type_counts"][pub_type] = count
        
        df_filtered = df[mask].copy()
        df_main = df[~mask].copy()
        
        self.stats["filtered_out"] = len(df_filtered)
        self.stats["remaining"] = len(df_main)
        
        print(f"   过滤掉 {self.stats['filtered_out']} 条，保留 {self.stats['remaining']} 条")
        
        return df_main, df_filtered
    
    def _remove_abstract_numbers(self, df: pd.DataFrame) -> pd.DataFrame:
        """删除摘要中的独立数字"""
        print("\n2. 删除摘要中的独立数字...")
        
        def remove_standalone_numbers(text):
            if pd.isna(text) or not isinstance(text, str):
                return text
            # 删除独立数字（保留如 COVID-19 中的数字）
            cleaned = re.sub(r'\b\d+\.?\d*\b', '', text)
            return re.sub(r'\s+', ' ', cleaned).strip()
        
        if "Abstract" in df.columns:
            original_lens = df["Abstract"].astype(str).str.len().sum()
            df["Abstract"] = df["Abstract"].apply(remove_standalone_numbers)
            new_lens = df["Abstract"].astype(str).str.len().sum()
            self.stats["numbers_removed"] = original_lens - new_lens
            print(f"   删除了约 {self.stats['numbers_removed']} 个字符")
        
        return df
    
    def _create_topic_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成主题建模专用数据"""
        print("\n3. 生成主题建模数据...")
        
        # 只保留主题建模需要的列
        cols = ["PMID", "Title", "Abstract", "Year", "Journal", "Citation_Count"]
        available_cols = [c for c in cols if c in df.columns]
        df_topic = df[available_cols].copy()
        
        print(f"   保留列: {available_cols}")
        
        return df_topic
    
    def _save_results(self, df_main, df_topic, df_filtered):
        """保存结果"""
        print("\n4. 保存结果...")
        
        df_main.to_csv(self.output_main, index=False, encoding="utf-8-sig")
        print(f"   主数据: {self.output_main}")
        
        df_topic.to_csv(self.output_topic, index=False, encoding="utf-8-sig")
        print(f"   主题建模数据: {self.output_topic}")
        
        if len(df_filtered) > 0:
            df_filtered.to_csv(self.output_filtered, index=False, encoding="utf-8-sig")
            print(f"   被过滤记录: {self.output_filtered}")
    
    def _write_log(self):
        """写入日志"""
        log_content = f"""文献类型过滤日志
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
项目名称: {get_project_name()}
============================================================

处理统计:
  原始记录数: {self.stats['original']}
  过滤掉: {self.stats['filtered_out']}
  保留: {self.stats['remaining']}
  保留率: {self.stats['remaining']/self.stats['original']*100:.2f}%

过滤规则:
  排除的文献类型: {', '.join(self.exclude_types)}
  删除摘要数字: {'是' if self.remove_numbers else '否'}

各类型过滤数量:
"""
        for pub_type, count in self.stats["type_counts"].items():
            log_content += f"  {pub_type}: {count}\n"
        
        log_content += f"""
输出文件:
  主数据: {self.output_main}
  主题建模数据: {self.output_topic}
  被过滤记录: {self.output_filtered}
"""
        
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write(log_content)
        print(f"\n   日志文件: {self.log_file}")


def main():
    filter_tool = PublicationTypeFilter()
    filter_tool.run()


if __name__ == "__main__":
    main()
