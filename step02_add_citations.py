#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_add_citations.py
第二步：为文献补充引用信息

功能：
1. 使用 OpenAlex API 基于 DOI 获取被引次数
2. 并发请求，支持重试和失败自动降速
3. 记录引用数据来源

输入：01_raw_data/{project_prefix}_basic.csv
输出：02_citations_data/{project_prefix}_with_citations.csv
"""

import sys
import time
import concurrent.futures as cf
from pathlib import Path
from typing import Optional, Dict

import pandas as pd
import requests
from tqdm import tqdm

try:
    from config import PATHS, CITATION_SOURCES, get_project_name, fix_encoding
except ImportError:
    print("请确保 config.py 存在且配置正确")
    sys.exit(1)


def fetch_openalex_citation(doi: str, mailto: str, timeout: int = 15, max_retries: int = 3) -> Optional[int]:
    """调用 OpenAlex API 获取被引次数"""
    if not doi or not isinstance(doi, str):
        return None
    
    doi_clean = doi.strip().lower()
    if not doi_clean:
        return None
    
    url = f"https://api.openalex.org/works/https://doi.org/{doi_clean}"
    params = {"mailto": mailto} if mailto else {}
    
    for retry in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code == 404:
                return None
            if resp.status_code == 429:  # 频率限制
                time.sleep(2 ** retry)
                continue
            resp.raise_for_status()
            return resp.json().get("cited_by_count")
        except requests.exceptions.Timeout:
            time.sleep(2 ** retry)
        except Exception:
            time.sleep(2 ** retry)
    
    return None


class CitationEnricher:
    """引用信息补充器"""
    
    def __init__(self):
        self.input_file = Path(PATHS["file_01_basic"])
        self.output_dir = Path(PATHS["dir_02_citations"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = Path(PATHS["file_02_citations"])
        
        self.cfg = CITATION_SOURCES.get("openalex", {})
        self.stats = {
            "total": 0,
            "with_doi": 0,
            "fetched": 0,
            "failed": 0,
        }
    
    def run(self):
        """执行引用信息补充"""
        print("=" * 60)
        print(f"第二步：引用信息补充 - {get_project_name()}")
        print("=" * 60)
        
        # 加载数据
        if not self.input_file.exists():
            print(f"错误：未找到输入文件 {self.input_file}")
            print("请先运行第一步抓取数据")
            return
        
        df = pd.read_csv(self.input_file, dtype={"PMID": str})
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        self.stats["total"] = len(df)
        print(f"加载数据: {len(df)} 条")
        
        # 初始化列
        if "Citation_Count" not in df.columns:
            df["Citation_Count"] = None
        if "Citation_Source" not in df.columns:
            df["Citation_Source"] = "Not_Fetched"
        
        # 检查是否启用 OpenAlex
        if not self.cfg.get("enabled", True):
            print("OpenAlex 未启用，跳过引用抓取")
            df.to_csv(self.output_file, index=False, encoding="utf-8-sig")
            return
        
        # 获取引用信息
        df = self._fetch_citations(df)
        
        # 保存结果
        df.to_csv(self.output_file, index=False, encoding="utf-8-sig")
        
        print("\n" + "=" * 60)
        print("第二步完成！")
        print(f"  总记录数: {self.stats['total']}")
        print(f"  有 DOI: {self.stats['with_doi']}")
        print(f"  成功获取引用: {self.stats['fetched']}")
        print(f"  输出文件: {self.output_file}")
        print("=" * 60)
    
    def _fetch_citations(self, df: pd.DataFrame) -> pd.DataFrame:
        """并发获取引用信息"""
        mailto = self.cfg.get("mailto", "")
        timeout = self.cfg.get("timeout", 15)
        max_workers = self.cfg.get("max_workers", 16)
        max_retries = self.cfg.get("max_retries", 3)
        
        # 筛选需要获取的记录（有 DOI 且无 Citation_Count）
        mask = df["Citation_Count"].isna() & df["DOI"].notna() & (df["DOI"].str.strip() != "")
        targets = df[mask][["PMID", "DOI"]].to_dict(orient="records")
        self.stats["with_doi"] = len(targets)
        
        if not targets:
            print("没有需要获取引用的记录")
            return df
        
        print(f"\n使用 OpenAlex 获取引用信息（{len(targets)} 条待处理）...")
        results: Dict[str, Optional[int]] = {}
        
        with cf.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_pmid = {
                executor.submit(fetch_openalex_citation, item["DOI"], mailto, timeout, max_retries): item["PMID"]
                for item in targets
            }
            for future in tqdm(cf.as_completed(future_to_pmid), total=len(targets), desc="获取引用"):
                pmid = future_to_pmid[future]
                try:
                    cited = future.result()
                    results[pmid] = cited
                    if cited is not None:
                        self.stats["fetched"] += 1
                except Exception:
                    results[pmid] = None
        
        # 回填结果
        for pmid, cited in results.items():
            if cited is not None:
                df.loc[df["PMID"] == pmid, "Citation_Count"] = cited
                df.loc[df["PMID"] == pmid, "Citation_Source"] = "OpenAlex"
        
        return df


def main():
    enricher = CitationEnricher()
    enricher.run()


if __name__ == "__main__":
    main()
