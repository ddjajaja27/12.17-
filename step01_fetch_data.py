#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_fetch_data.py
第一步：从 PubMed 抓取文献数据

功能：
1. 按年份分批检索 PubMed，获取 PMID 列表
2. 批量获取文献详情（标题、摘要、作者、期刊等）
3. 自动修复编码乱码（希腊字母、特殊标点等）
4. 支持断点续传

输入：config.py 中的检索配置
输出：01_raw_data/{project_prefix}_pmids.txt
      01_raw_data/{project_prefix}_basic.csv
"""

import os
import sys
import time
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

try:
    from Bio import Entrez
    import xml.etree.ElementTree as ET
except ImportError:
    print("请安装 biopython: pip install biopython")
    sys.exit(1)

try:
    from config import (
        API_CONFIG, PUBMED_CONFIG, PROCESSING_CONFIG, PATHS,
        PROJECT_CONFIG, fix_encoding, get_project_name
    )
except ImportError:
    print("请确保 config.py 存在且配置正确")
    sys.exit(1)


class PubMedFetcher:
    """PubMed 数据抓取器"""
    
    def __init__(self):
        Entrez.email = API_CONFIG["email"]
        if API_CONFIG.get("api_key"):
            Entrez.api_key = API_CONFIG["api_key"]
        
        self.keyword = PUBMED_CONFIG["keyword"]
        self.start_date = PUBMED_CONFIG["start_date"]
        self.end_date = PUBMED_CONFIG["end_date"]
        self.date_field = PUBMED_CONFIG["date_field"]
        
        # 输出路径
        self.output_dir = Path(PATHS["dir_01_raw"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pmid_file = Path(PATHS["file_01_pmids"])
        self.output_file = Path(PATHS["file_01_basic"])
        
        # 统计
        self.stats = {
            "total_pmids": 0,
            "downloaded": 0,
            "failed": 0,
            "encoding_fixes": 0,
        }
    
    def run(self):
        """执行完整抓取流程"""
        print("=" * 60)
        print(f"第一步：PubMed 数据抓取 - {get_project_name()}")
        print(f"检索词: {self.keyword}")
        print(f"时间范围: {self.start_date} ~ {self.end_date}")
        print("=" * 60)
        
        # 1. 获取 PMID 列表
        pmids = self._search_pmids()
        if not pmids:
            print("未找到任何文献，请检查检索条件")
            return
        
        # 2. 批量获取文献详情
        df = self._fetch_details(pmids)
        
        # 3. 保存结果
        self._save_results(df)
        
        print("\n" + "=" * 60)
        print("第一步完成！")
        print(f"  PMID 总数: {self.stats['total_pmids']}")
        print(f"  成功下载: {self.stats['downloaded']}")
        print(f"  编码修复: {self.stats['encoding_fixes']} 处")
        print(f"  输出文件: {self.output_file}")
        print("=" * 60)
    
    def _search_pmids(self) -> List[str]:
        """按年份检索获取 PMID 列表"""
        all_pmids = []
        start_year = int(self.start_date.split("/")[0])
        end_year = int(self.end_date.split("/")[0])
        
        print(f"\n按年份检索 ({start_year} ~ {end_year})...")
        
        for year in tqdm(range(start_year, end_year + 1), desc="检索进度"):
            query = f'{self.keyword} AND {year}/01/01:{year}/12/31{self.date_field}'
            try:
                handle = Entrez.esearch(db="pubmed", term=query, retmax=10000, usehistory="y")
                record = Entrez.read(handle)
                handle.close()
                
                count = int(record["Count"])
                if count == 0:
                    continue
                
                webenv = record["WebEnv"]
                query_key = record["QueryKey"]
                
                # 分批获取该年份的 PMID
                for retstart in range(0, count, 10000):
                    batch_handle = Entrez.esearch(
                        db="pubmed", term="",
                        retstart=retstart, retmax=10000,
                        webenv=webenv, query_key=query_key, usehistory="y"
                    )
                    batch_record = Entrez.read(batch_handle)
                    batch_handle.close()
                    all_pmids.extend(batch_record["IdList"])
                    time.sleep(API_CONFIG["delay_between_requests"])
                
            except Exception as e:
                print(f"\n  年份 {year} 检索出错: {e}")
                time.sleep(API_CONFIG["retry_delay"])
        
        # 去重
        unique_pmids = list(set(all_pmids))
        self.stats["total_pmids"] = len(unique_pmids)
        
        # 保存 PMID 列表
        with open(self.pmid_file, "w", encoding="utf-8") as f:
            f.write("\n".join(unique_pmids))
        print(f"\nPMID 列表已保存: {self.pmid_file} ({len(unique_pmids)} 个)")
        
        return unique_pmids
    
    def _fetch_details(self, pmids: List[str]) -> pd.DataFrame:
        """批量获取文献详情"""
        batch_size = PROCESSING_CONFIG["details_batch_size"]
        all_records = []
        downloaded_pmids = set()
        
        # 断点续传：检查已有数据
        if self.output_file.exists():
            try:
                df_existing = pd.read_csv(self.output_file, dtype={"PMID": str})
                all_records = df_existing.to_dict(orient="records")
                downloaded_pmids = set(df_existing["PMID"].astype(str))
                print(f"检测到已有数据，继续下载（已有 {len(downloaded_pmids)} 条）")
            except Exception as e:
                print(f"读取已有数据失败: {e}")
        
        # 筛选未下载的 PMID
        pmids_to_download = [p for p in pmids if p not in downloaded_pmids]
        if not pmids_to_download:
            print("所有 PMID 均已下载完成")
            return pd.DataFrame(all_records)
        
        print(f"\n开始下载文献详情（共 {len(pmids_to_download)} 条待下载）...")
        total_batches = (len(pmids_to_download) + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(total_batches), desc="下载进度"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(pmids_to_download))
            batch_pmids = pmids_to_download[start_idx:end_idx]
            
            for retry in range(API_CONFIG["max_retries"]):
                try:
                    handle = Entrez.efetch(db="pubmed", id=batch_pmids, retmode="xml")
                    xml_data = handle.read()
                    handle.close()
                    
                    root = ET.fromstring(xml_data)
                    for article in root.findall(".//PubmedArticle"):
                        record = self._parse_article(article)
                        if record:
                            all_records.append(record)
                            self.stats["downloaded"] += 1
                    
                    # 定期保存中间结果
                    if batch_idx % 20 == 0 and batch_idx > 0:
                        pd.DataFrame(all_records).to_csv(
                            self.output_file, index=False, encoding="utf-8-sig"
                        )
                    
                    time.sleep(API_CONFIG["delay_between_requests"])
                    break
                    
                except Exception as e:
                    if retry < API_CONFIG["max_retries"] - 1:
                        time.sleep(API_CONFIG["retry_delay"] * (retry + 1))
                    else:
                        print(f"\n批次 {batch_idx + 1} 失败: {e}")
                        self.stats["failed"] += len(batch_pmids)
        
        return pd.DataFrame(all_records)
    
    def _get_text(self, elem) -> str:
        """递归获取 XML 元素的完整文本（包括子元素）"""
        if elem is None:
            return ""
        text = elem.text or ""
        for child in elem:
            text += self._get_text(child)
            text += child.tail or ""
        return text.strip()
    
    def _fix_text(self, text: str) -> str:
        """修复文本编码问题"""
        if not text:
            return ""
        fixed = fix_encoding(text)
        if fixed != text:
            self.stats["encoding_fixes"] += 1
        return fixed
    
    def _parse_article(self, article) -> Optional[Dict]:
        """解析单篇文献 XML"""
        try:
            # PMID
            pmid_elem = article.find(".//PMID")
            pmid = self._get_text(pmid_elem)
            if not pmid:
                return None
            
            # 标题
            title_elem = article.find(".//ArticleTitle")
            title = self._fix_text(self._get_text(title_elem))
            
            # 摘要
            abstract_parts = []
            for abstract_text in article.findall(".//AbstractText"):
                text = self._fix_text(self._get_text(abstract_text))
                if text:
                    label = abstract_text.get("Label")
                    if label:
                        abstract_parts.append(f"{label}: {text}")
                    else:
                        abstract_parts.append(text)
            abstract = " ".join(abstract_parts)
            
            # 作者
            authors = []
            for author in article.findall(".//Author"):
                lastname = author.find("LastName")
                forename = author.find("ForeName")
                if lastname is not None:
                    name = self._get_text(lastname)
                    if forename is not None:
                        name = f"{self._get_text(forename)} {name}"
                    authors.append(self._fix_text(name))
            
            # 期刊
            journal_elem = article.find(".//Journal/Title")
            journal = self._fix_text(self._get_text(journal_elem))
            
            # 年份
            year = ""
            pub_date_elem = article.find(".//PubMedPubDate[@PubStatus='pubmed']")
            if pub_date_elem is not None:
                year_elem = pub_date_elem.find("Year")
                if year_elem is not None:
                    year = self._get_text(year_elem)
            if not year:
                pub_date = article.find(".//PubDate/Year")
                if pub_date is not None:
                    year = self._get_text(pub_date)
            
            # DOI
            doi = ""
            for article_id in article.findall(".//ArticleId"):
                if article_id.get("IdType") == "doi":
                    doi = self._get_text(article_id)
                    break
            
            # MeSH 术语
            mesh_terms = [self._fix_text(self._get_text(m)) 
                          for m in article.findall(".//MeshHeading/DescriptorName")]
            mesh_terms = [m for m in mesh_terms if m]
            
            # 文献类型
            pub_types = [self._fix_text(self._get_text(p)) 
                         for p in article.findall(".//PublicationType")]
            pub_types = [p for p in pub_types if p]
            
            # 关键词
            keywords = [self._fix_text(self._get_text(k)) 
                        for k in article.findall(".//Keyword")]
            keywords = [k for k in keywords if k]
            
            return {
                "PMID": pmid,
                "Title": title,
                "Abstract": abstract,
                "Authors": "; ".join(authors),
                "Journal": journal,
                "Year": year,
                "DOI": doi,
                "MeSH_Terms": "; ".join(mesh_terms),
                "Publication_Types": "; ".join(pub_types),
                "Keywords": "; ".join(keywords),
                "Download_Date": datetime.now().strftime("%Y-%m-%d"),
            }
            
        except Exception as e:
            print(f"\n解析文献出错: {e}")
            return None
    
    def _save_results(self, df: pd.DataFrame):
        """保存最终结果"""
        if df.empty:
            print("无数据可保存")
            return
        
        # 使用 utf-8-sig 编码，确保 Excel 正确显示中文和特殊字符
        df.to_csv(self.output_file, index=False, encoding="utf-8-sig")
        print(f"\n数据已保存: {self.output_file}")


def main():
    fetcher = PubMedFetcher()
    fetcher.run()


if __name__ == "__main__":
    main()
