# -*- coding: utf-8 -*-
"""
config.py - 项目统一配置文件

【重要说明】
1. 只需修改最前面的【用户配置区】，后续所有配置自动同步
2. 支持多数据源（PubMed / Web of Science / Scopus / 自定义导入）
3. 标准化字段结构，确保不同数据源输出格式一致
"""


#region  ======【用户配置区 - 只改这里】======
# ⚠️ 只需修改本区域，建议不要动后面内容！


# ---------- 1. 项目基本信息 ----------
PROJECT_NAME_CN = "幽门螺杆菌"                    # 项目中文名（用于日志显示）
PROJECT_PREFIX = "helicobacter_pylori"           # 项目英文前缀（用于生成文件名）
SEARCH_KEYWORD = "Helicobacter pylori"           # 检索关键词（英文）

# ---------- 2. 时间范围 ----------
START_DATE = "2005/01/01"                        # 开始日期
END_DATE = "2025/12/31"                          # 结束日期

# ---------- 3. 数据源选择 ----------
# 可选: "pubmed", "wos", "scopus", "openalex", "custom"
# - pubmed:   从 PubMed 在线检索
# - wos:      从 Web of Science 导出文件读取
# - scopus:   从 Scopus 导出文件读取
# - openalex: 从 OpenAlex API 检索
# - custom:   自定义导入（需提供符合标准结构的 CSV）
DATA_SOURCE = "pubmed"

# ---------- 4. 导入文件路径（仅 wos/scopus/custom 模式需要）----------
# 当 DATA_SOURCE 为 "wos", "scopus", "custom" 时，需指定导入文件
IMPORT_FILE_PATH = ""                            # 例如: "C:/data/wos_export.txt"

# ---------- 5. API 配置 ----------
USER_EMAIL = "ddjajaja27@gmail.com"              # 用于 API 请求的邮箱
PUBMED_API_KEY = "5f7a88f263b77c32f86fbc3d31203013c808"  # PubMed API Key

# ---------- 6. 主题模型参数 ----------
MIN_TOPIC_SIZE = 60                              # 最小主题文献数
MODEL_NAME = "all-MiniLM-L6-v2"                  # 嵌入模型名称

# 嵌入后端选择（Windows上sentence-transformers/torch偶发崩溃时建议用tfidf_svd）
# 可选: "sentence_transformers", "tfidf_svd"
EMBEDDING_BACKEND = "tfidf_svd"

# ---------- 7. 过滤配置 ----------
MIN_TEXT_LENGTH = 100                            # 最小文本长度（字符）
EXCLUDE_PUB_TYPES = ["Comment", "Editorial", "Letter", "Erratum", "Correction", "Retraction"]

# ---------- 7.5 去噪配置（可选，通用，不随研究方向大改） ----------
# 说明：
# - stopwords_files: 可以填多个停用词表文件（逐个合并）；不存在会自动跳过
# - whitelist_file: 可选；用于“保护词”，避免领域核心词被误删（不填也没关系）
# - auto_protect: 默认开启；自动从检索词/项目名/Keywords/MeSH_Terms 提取保护词
DENOISE_CONFIG = {
    "enabled": True,
    "stopwords_files": [
        # A: 统计去噪（EVT+IDF 合并）
        "05_stopwords/rigor/outputs/A_stopwords_protected.txt",
        "05_stopwords/Experiment_A_Statistical/output/combined_stopwords.txt",
        # B: 语义扩展去噪
        "05_stopwords/rigor/outputs/B_stopwords_protected.txt",
        "05_stopwords/Experiment_B_Semantic/output/stopwords_semantic_extended.txt",
    ],
    "whitelist_file": "05_stopwords/common_data/whitelist.txt",
    "auto_protect": {
        "enabled": True,
        "use_search_keyword": True,
        "use_project_prefix": True,
        "use_keywords_fields": True,
        "min_token_len": 3,
        "max_tokens": 8000,
    },
    "apply_to_embeddings": False,
}

# ---------- 8. 自定义参数（最多3个，超出报错） ----------
CUSTOM_PARAMS = {
    # "param1": "value1",
    # "param2": "value2",
    # "param3": "value3",
}
assert len(CUSTOM_PARAMS) <= 3, "自定义参数最多只能有3个！如需更多请联系开发者扩展。"

#endregion


# ╔════════════════════════════════════════════════════════════╗
# ║              【标准化数据结构 - 所有数据源统一输出】            ║
# ╚════════════════════════════════════════════════════════════╝

# 标准字段定义（不管从哪个数据源获取，最终输出的 CSV 都包含这些字段）
STANDARD_FIELDS = {
    # 核心字段（必须）
    "id": "ID",                     # 唯一标识符（PMID / WOS_ID / Scopus_ID 等）
    "title": "Title",               # 文献标题
    "abstract": "Abstract",         # 摘要
    "year": "Year",                 # 发表年份
    
    # 引用信息
    "citation_count": "Citation_Count",  # 被引次数
    "doi": "DOI",                   # DOI
    
    # 作者与期刊
    "authors": "Authors",           # 作者列表（分号分隔）
    "journal": "Journal",           # 期刊名称
    
    # 分类与关键词
    "keywords": "Keywords",         # 作者关键词（分号分隔）
    "mesh_terms": "MeSH_Terms",     # MeSH 主题词（仅 PubMed，分号分隔）
    "pub_types": "Publication_Types",  # 文献类型（分号分隔）
    
    # 数据源信息
    "source_db": "Source_DB",       # 数据来源（pubmed / wos / scopus 等）
    "download_date": "Download_Date",  # 下载日期
    
    # 建模用字段（由 step03 生成）
    "text_for_model": "text_for_model",  # 合并后的建模文本
}

# 各数据源到标准字段的映射
FIELD_MAPPINGS = {
    "pubmed": {
        "PMID": "ID",
        "Title": "Title",
        "Abstract": "Abstract",
        "Year": "Year",
        "Citation_Count": "Citation_Count",
        "DOI": "DOI",
        "Authors": "Authors",
        "Journal": "Journal",
        "Keywords": "Keywords",
        "MeSH_Terms": "MeSH_Terms",
        "Publication_Types": "Publication_Types",
    },
    "wos": {
        "UT": "ID",                  # WOS 唯一ID
        "TI": "Title",               # 标题
        "AB": "Abstract",            # 摘要
        "PY": "Year",                # 发表年份
        "TC": "Citation_Count",      # 被引次数
        "DI": "DOI",                 # DOI
        "AU": "Authors",             # 作者
        "SO": "Journal",             # 期刊
        "DE": "Keywords",            # 作者关键词
        "ID": "Keywords_Plus",       # WOS Keywords Plus
        "DT": "Publication_Types",   # 文献类型
    },
    "scopus": {
        "EID": "ID",                 # Scopus 唯一ID
        "Title": "Title",            # 标题
        "Abstract": "Abstract",      # 摘要
        "Year": "Year",              # 发表年份
        "Cited by": "Citation_Count",# 被引次数
        "DOI": "DOI",                # DOI
        "Authors": "Authors",        # 作者
        "Source title": "Journal",   # 期刊
        "Author Keywords": "Keywords",  # 作者关键词
        "Index Keywords": "Index_Keywords",  # 索引关键词
        "Document Type": "Publication_Types",  # 文献类型
    },
    "openalex": {
        "id": "ID",                  # OpenAlex ID
        "title": "Title",            # 标题
        "abstract": "Abstract",      # 摘要
        "publication_year": "Year",  # 发表年份
        "cited_by_count": "Citation_Count",  # 被引次数
        "doi": "DOI",                # DOI
        "authorships": "Authors",    # 作者
        "primary_location.source.display_name": "Journal",  # 期刊
        "keywords": "Keywords",      # 关键词
        "type": "Publication_Types", # 文献类型
    },
    "custom": {
        # 自定义导入：假设用户已按标准字段命名，直接使用
        "ID": "ID",
        "Title": "Title",
        "Abstract": "Abstract",
        "Year": "Year",
        "Citation_Count": "Citation_Count",
        "DOI": "DOI",
        "Authors": "Authors",
        "Journal": "Journal",
        "Keywords": "Keywords",
        "Publication_Types": "Publication_Types",
    },
}


# ╔════════════════════════════════════════════════════════════╗
# ║                【自动同步区 - 一般无需修改】                   ║
# ╚════════════════════════════════════════════════════════════╝

# 项目配置（自动同步）
PROJECT_CONFIG = {
    "project_name_cn": PROJECT_NAME_CN,
    "project_prefix": PROJECT_PREFIX,
    "search_keyword": SEARCH_KEYWORD,
    "start_date": START_DATE,
    "end_date": END_DATE,
    "data_source": DATA_SOURCE,
    "import_file": IMPORT_FILE_PATH,
}

# API 配置（自动同步）
API_CONFIG = {
    "email": USER_EMAIL,
    "api_key": PUBMED_API_KEY,
    "max_retries": 3,
    "delay_between_requests": 0.34,
    "retry_delay": 5,
}

# PubMed 检索配置（自动同步）
PUBMED_CONFIG = {
    "keyword": SEARCH_KEYWORD,
    "start_date": START_DATE,
    "end_date": END_DATE,
    "date_field": "[edat]",
}

# 处理配置（自动同步）
PROCESSING_CONFIG = {
    "details_batch_size": 100,
    "min_text_length": MIN_TEXT_LENGTH,
}

# 引用数据源配置
CITATION_SOURCES = {
    "openalex": {
        "enabled": True,
        "mailto": USER_EMAIL,
        "timeout": 15,
        "max_workers": 16,
        "max_retries": 3,
        "initial_wait_time": 0.1,
    }
}

# 主题模型配置（自动同步）
MODEL_CONFIG = {
    "model_name": MODEL_NAME,
    "embedding_backend": EMBEDDING_BACKEND,
    "min_topic_size": MIN_TOPIC_SIZE,
    "nr_topics": "auto",
    "n_gram_range": (1, 1),
    "top_n_words": 10,
    "language": "english",
}

# 文献类型过滤配置（自动同步）
FILTER_CONFIG = {
    "exclude_types": EXCLUDE_PUB_TYPES,
    "remove_abstract_numbers": True,
}

# ============================================================
# 路径配置（10步流程，文件夹与步骤一一对应）
# ============================================================
_prefix = PROJECT_PREFIX

# 数据目录的基础路径
import os as _os
_base = _os.path.dirname(_os.path.abspath(__file__))

PATHS = {
    # ========== 目录配置（10步流程） ==========
    # Step 01: 数据采集
    "dir_01_raw": _os.path.join(_base, "01_raw_data"),
    
    # Step 02: 引用补充
    "dir_02_citations": _os.path.join(_base, "02_citations_data"),
    
    # Step 03: 数据清洗
    "dir_03_cleaned": _os.path.join(_base, "03_cleaned_data"),
    
    # Step 04: 类型过滤
    "dir_04_filtered": _os.path.join(_base, "04_filtered_data"),
    
    # Step 05: 停用词生成（A/B/C三方案）
    "dir_05_stopwords": _os.path.join(_base, "05_stopwords"),
    
    # Step 06: 去噪执行（baseline/A/B/C/AB/ABC六版本）
    "dir_06_denoised": _os.path.join(_base, "06_denoised_data"),
    
    # Step 07: 主题建模
    "dir_07_topic_models": _os.path.join(_base, "07_topic_models"),
    
    # Step 08: 参数选择（C_v一致性筛选）
    "dir_08_selection": _os.path.join(_base, "08_model_selection"),
    
    # Step 09: 可视化
    "dir_09_visualization": _os.path.join(_base, "09_visualization"),
    
    # Step 10: 报告生成
    "dir_10_report": _os.path.join(_base, "10_report"),
    
    # ========== 文件配置 ==========
    # Step 01 输出
    "file_01_pmids": _os.path.join(_base, f"01_raw_data/{_prefix}_pmids.txt"),
    "file_01_basic": _os.path.join(_base, f"01_raw_data/{_prefix}_basic.csv"),
    
    # Step 02 输出
    "file_02_citations": _os.path.join(_base, f"02_citations_data/{_prefix}_with_citations.csv"),
    
    # Step 03 输出
    "file_03_cleaned": _os.path.join(_base, f"03_cleaned_data/{_prefix}_cleaned.csv"),
    
    # Step 04 输出
    "file_04_main": _os.path.join(_base, f"04_filtered_data/{_prefix}_main.csv"),
    "file_04_topic": _os.path.join(_base, f"04_filtered_data/{_prefix}_topic_modeling.csv"),
    "file_04_filtered_out": _os.path.join(_base, f"04_filtered_data/{_prefix}_filtered_out.csv"),
    "file_04_log": _os.path.join(_base, f"04_filtered_data/{_prefix}_filter_log.txt"),

    # Step 06 输出（去噪后数据）
    "file_06_denoised_topic": _os.path.join(_base, f"06_denoised_data/{_prefix}_topic_modeling_ABC.csv"),
    
    # ========== 兼容旧代码的别名 ==========
    "dir_05_denoised": _os.path.join(_base, "06_denoised_data"),
    "dir_06_results": _os.path.join(_base, "07_topic_models"),
    "dir_07_visualization": _os.path.join(_base, "09_visualization"),
    "dir_08_report": _os.path.join(_base, "10_report"),
    "import_file": IMPORT_FILE_PATH,
    "data_raw": _os.path.join(_base, "01_raw_data"),
    "data_citations": _os.path.join(_base, "02_citations_data"),
    "data_curated": _os.path.join(_base, "03_cleaned_data"),
    "topic_model_results": _os.path.join(_base, "07_topic_models"),
}

# ============================================================
# 编码修复映射表（用于修复 PubMed 返回的乱码字符）
# ============================================================
ENCODING_FIXES = {}

# ============================================================
# 辅助函数
# ============================================================
def get_project_name() -> str:
    """获取项目中文名"""
    return PROJECT_NAME_CN

def get_prefix() -> str:
    """获取项目文件名前缀"""
    return PROJECT_PREFIX

def get_data_source() -> str:
    """获取当前数据源"""
    return DATA_SOURCE

def get_field_mapping() -> dict:
    """获取当前数据源的字段映射"""
    return FIELD_MAPPINGS.get(DATA_SOURCE, FIELD_MAPPINGS["custom"])

from typing import Optional

def standardize_dataframe(df, source: Optional[str] = None):
    """
    将不同数据源的 DataFrame 标准化为统一格式
    
    参数:
        df: 原始 DataFrame
        source: 数据源类型（如不指定则使用 DATA_SOURCE）
    
    返回:
        标准化后的 DataFrame
    """
    import pandas as pd
    from datetime import datetime
    
    if source is None:
        source = DATA_SOURCE
    
    mapping = FIELD_MAPPINGS.get(source, FIELD_MAPPINGS["custom"])
    
    # 创建新的标准化 DataFrame
    std_df = pd.DataFrame()
    
    # 按映射转换字段
    for orig_col, std_col in mapping.items():
        if orig_col in df.columns:
            std_df[std_col] = df[orig_col]
    
    # 确保所有标准字段存在（缺失的填空）
    for std_col in STANDARD_FIELDS.values():
        if std_col not in std_df.columns:
            std_df[std_col] = ""
    
    # 添加数据源标记
    std_df["Source_DB"] = source
    std_df["Download_Date"] = datetime.now().strftime("%Y-%m-%d")
    
    return std_df

def fix_encoding(text: str) -> str:
    """
    修复文本中的编码乱码
    使用多种策略尝试修复 UTF-8 被误读为 Latin-1 的问题
    """
    if not text or not isinstance(text, str):
        return text if text else ""
    
    # 策略1：尝试修复 UTF-8 被误读为 Latin-1 的情况
    try:
        if any(c in text for c in ['\xc3', '\xc2', '\xe2']):
            fixed = text.encode('latin-1').decode('utf-8')
            return fixed
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass
    
    # 策略2：直接返回原文本（已经是正确的 UTF-8）
    return text


# ============================================================
# 数据源解析器（用于读取不同格式的导入文件）
# ============================================================
def parse_wos_file(file_path: str):
    """
    解析 Web of Science 导出文件（Tab-delimited 或 Plain Text）
    
    支持格式：
    - Tab-delimited (*.txt)
    - Plain Text (*.txt)
    
    返回:
        标准化后的 pandas DataFrame
    """
    import pandas as pd
    
    try:
        # 尝试 Tab-delimited 格式
        df = pd.read_csv(file_path, sep='\t', encoding='utf-8-sig', dtype=str)
        if len(df.columns) > 1:
            return standardize_dataframe(df, "wos")
    except:
        pass
    
    # 尝试 Plain Text 格式（WOS 特有格式）
    records = []
    current_record = {}
    current_field = None
    
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            
            if line.startswith('ER'):  # End of Record
                if current_record:
                    records.append(current_record)
                    current_record = {}
                continue
            
            if len(line) >= 2 and line[2:3] == ' ':
                field = line[:2].strip()
                value = line[3:].strip()
                if field:
                    current_field = field
                    if field in current_record:
                        current_record[field] += "; " + value
                    else:
                        current_record[field] = value
                elif current_field:
                    current_record[current_field] += " " + value
    
    if records:
        df = pd.DataFrame(records)
        return standardize_dataframe(df, "wos")
    
    raise ValueError("无法解析 WOS 文件格式")


def parse_scopus_file(file_path: str):
    """
    解析 Scopus 导出文件（CSV 格式）
    
    返回:
        标准化后的 pandas DataFrame
    """
    import pandas as pd
    
    df = pd.read_csv(file_path, encoding='utf-8-sig', dtype=str)
    return standardize_dataframe(df, "scopus")


def parse_custom_file(file_path: str):
    """
    解析自定义 CSV 文件（假设已按标准字段命名）
    
    返回:
        标准化后的 pandas DataFrame
    """
    import pandas as pd
    
    df = pd.read_csv(file_path, encoding='utf-8-sig', dtype=str)
    return standardize_dataframe(df, "custom")


def load_import_data():
    """
    根据 DATA_SOURCE 加载导入数据
    
    返回:
        标准化后的 DataFrame
    """
    if DATA_SOURCE == "pubmed":
        raise ValueError("PubMed 模式应使用在线检索，不需要导入文件")
    
    if not IMPORT_FILE_PATH:
        raise ValueError(f"DATA_SOURCE 为 '{DATA_SOURCE}'，但未指定 IMPORT_FILE_PATH")
    
    import os
    if not os.path.exists(IMPORT_FILE_PATH):
        raise FileNotFoundError(f"导入文件不存在: {IMPORT_FILE_PATH}")
    
    if DATA_SOURCE == "wos":
        return parse_wos_file(IMPORT_FILE_PATH)
    elif DATA_SOURCE == "scopus":
        return parse_scopus_file(IMPORT_FILE_PATH)
    elif DATA_SOURCE == "custom":
        return parse_custom_file(IMPORT_FILE_PATH)
    else:
        raise ValueError(f"不支持的数据源: {DATA_SOURCE}")
