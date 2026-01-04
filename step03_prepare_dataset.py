#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_prepare_dataset.py
第三步：数据清洗与预处理

功能：
1. 去重（基于 PMID）
2. 合并标题和摘要，生成建模用文本
3. 过滤过短文本
4. 标准化年份字段
5. 保留丢弃记录日志

输入：02_citations_data/{project_prefix}_with_citations.csv
输出：03_cleaned_data/{project_prefix}_cleaned.csv
      03_cleaned_data/{project_prefix}_dropped.csv（被丢弃的记录）
"""

import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

try:
    from config import PATHS, PROCESSING_CONFIG, get_project_name, fix_encoding
except ImportError:
    print("请确保 config.py 存在且配置正确")
    sys.exit(1)


class DataCleaner:
    """数据清洗器"""
    
    def __init__(self):
        self.input_file = Path(PATHS["file_02_citations"])
        self.output_dir = Path(PATHS["dir_03_cleaned"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = Path(PATHS["file_03_cleaned"])
        self.dropped_file = self.output_dir / f"{PATHS['file_03_cleaned'].split('/')[-1].replace('.csv', '_dropped.csv')}"
        
        self.min_text_length = PROCESSING_CONFIG.get("min_text_length", 100)
        
        self.stats = {
            "original": 0,
            "after_dedup": 0,
            "after_filter": 0,
            "dropped_short": 0,
            "dropped_dup": 0,
        }
    
    def run(self):
        """执行数据清洗"""
        print("=" * 60)
        print(f"第三步：数据清洗 - {get_project_name()}")
        print("=" * 60)
        
        # 加载数据
        if not self.input_file.exists():
            print(f"错误：未找到输入文件 {self.input_file}")
            print("请先运行第二步添加引用信息")
            return
        
        df = pd.read_csv(self.input_file, dtype={"PMID": str})
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        self.stats["original"] = len(df)
        print(f"加载数据: {len(df)} 条")
        
        # 清洗步骤
        df, dropped_dup = self._remove_duplicates(df)
        df, dropped_short = self._filter_short_texts(df)
        df = self._standardize_fields(df)
        
        # 合并丢弃的记录
        dropped_all = pd.concat([dropped_dup, dropped_short], ignore_index=True)
        
        # 保存结果
        df.to_csv(self.output_file, index=False, encoding="utf-8-sig")
        if len(dropped_all) > 0:
            dropped_all.to_csv(self.dropped_file, index=False, encoding="utf-8-sig")
        
        print("\n" + "=" * 60)
        print("第三步完成！")
        print(f"  原始记录: {self.stats['original']}")
        print(f"  去重后: {self.stats['after_dedup']} (删除 {self.stats['dropped_dup']} 条重复)")
        print(f"  过滤后: {self.stats['after_filter']} (删除 {self.stats['dropped_short']} 条短文本)")
        print(f"  输出文件: {self.output_file}")
        if len(dropped_all) > 0:
            print(f"  丢弃记录: {self.dropped_file}")
        print("=" * 60)
    
    def _remove_duplicates(self, df: pd.DataFrame) -> tuple:
        """去重"""
        print("\n1. 去重（基于 PMID）...")
        before = len(df)
        
        # 找出重复记录
        duplicates = df[df.duplicated(subset=["PMID"], keep="first")].copy()
        duplicates["Drop_Reason"] = "Duplicate_PMID"
        
        # 保留第一条
        df = df.drop_duplicates(subset=["PMID"], keep="first")
        
        self.stats["after_dedup"] = len(df)
        self.stats["dropped_dup"] = before - len(df)
        print(f"   删除 {self.stats['dropped_dup']} 条重复记录")
        
        return df, duplicates
    
    def _filter_short_texts(self, df: pd.DataFrame) -> tuple:
        """过滤短文本"""
        print(f"\n2. 过滤短文本（<{self.min_text_length} 字符）...")
        
        # 填充空值
        df["Title"] = df["Title"].fillna("").astype(str)
        df["Abstract"] = df["Abstract"].fillna("").astype(str)
        
        # 生成建模用文本
        df["text_for_model"] = (
            df["Title"].str.strip() + ". " + df["Abstract"].str.strip()
        ).str.strip()
        
        # 计算文本长度
        df["_text_len"] = df["text_for_model"].str.len()
        
        # 分离短文本
        mask_short = df["_text_len"] <= self.min_text_length
        dropped = df[mask_short].copy()
        dropped["Drop_Reason"] = f"Text_Too_Short (<{self.min_text_length})"
        
        df = df[~mask_short].copy()
        df = df.drop(columns=["_text_len"])
        
        self.stats["after_filter"] = len(df)
        self.stats["dropped_short"] = len(dropped)
        print(f"   删除 {self.stats['dropped_short']} 条短文本")
        
        return df, dropped
    
    def _standardize_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化字段"""
        print("\n3. 标准化字段...")
        
        # 年份转数值
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
        
        # 引用数转数值
        if "Citation_Count" in df.columns:
            df["Citation_Count"] = pd.to_numeric(df["Citation_Count"], errors="coerce").fillna(0).astype(int)
        
        # 再次应用编码修复
        text_cols = ["Title", "Abstract", "Authors", "Keywords", "MeSH_Terms"]
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].apply(fix_encoding)
        
        print("   完成")
        
        return df


def main():
    cleaner = DataCleaner()
    cleaner.run()


if __name__ == "__main__":
    main()
