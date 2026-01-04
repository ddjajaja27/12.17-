#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VPD 消融实验配置与执行
拆分噪声方向，逐步投影，看哪个分量真的驱动了性能改进
"""

from dataclasses import dataclass
from typing import List, Dict

# ============================================================================
# 噪声词分组：M/S/B/Anatomy
# ============================================================================

@dataclass
class NoiseWordGroups:
    """噪声词分组定义"""
    
    # Group M: Methodology（方法论噪音）
    M = [
        "analysis", "study", "results", "method", "conclusion", "data",
        "using", "performed", "evaluated", "aim", "background", "investigated",
    ]
    
    # Group S: Statistics（统计描述噪音）
    S = [
        "significant", "significantly", "increased", "decreased", "higher",
        "lower", "compared", "group", "rate", "ratio", "value", "associated",
        "difference", "respectively",
    ]
    
    # Group B: Broad background（宽泛背景噪音）
    B = [
        "clinical", "patient", "patients", "treatment", "disease", "infection",
        "cases", "years", "time", "effect", "various", "regarding", "reported",
        "recent",
    ]
    
    # Group Anatomy: 解剖/对象词（对照组，理论上不应该删）
    Anatomy = [
        "gastric", "stomach", "mucosa", "biopsy", "human", "tissue",
        "samples", "specimens"
    ]
    
    @classmethod
    def get_groups(cls) -> Dict[str, List[str]]:
        """返回所有分组"""
        return {
            "M": cls.M,
            "S": cls.S,
            "B": cls.B,
            "Anatomy": cls.Anatomy,
        }
    
    @classmethod
    def get_combined(cls, groups: List[str]) -> List[str]:
        """组合多个分组"""
        all_groups = cls.get_groups()
        result = []
        for g in groups:
            if g in all_groups:
                result.extend(all_groups[g])
        return result


# 消融实验配置
ABLATION_CONFIGS = {
    "baseline": {
        "name": "Baseline（无投影）",
        "noise_words": [],
        "description": "Control: 原始融合向量，无去噪"
    },
    "M_S": {
        "name": "投影 M+S（核心背景噪声）",
        "noise_words": NoiseWordGroups.get_combined(["M", "S"]),
        "description": "仅移除方法论+统计噪音，保留背景和对象词"
    },
    "M_S_B": {
        "name": "投影 M+S+B（包括背景）",
        "noise_words": NoiseWordGroups.get_combined(["M", "S", "B"]),
        "description": "移除方法论、统计、背景噪音，保留解剖词"
    },
    "M_S_B_Anatomy": {
        "name": "投影 M+S+B+Anatomy（全部噪声）",
        "noise_words": NoiseWordGroups.get_combined(["M", "S", "B", "Anatomy"]),
        "description": "你现在这版：完整去噪（包括解剖/对象词）"
    },
}


def print_ablation_summary():
    """打印消融实验配置总结"""
    print("=" * 80)
    print("🧪 VPD 消融实验配置")
    print("=" * 80)
    
    for key, config in ABLATION_CONFIGS.items():
        print(f"\n{key.upper()}: {config['name']}")
        print(f"  噪声词数: {len(config['noise_words'])}")
        print(f"  说明: {config['description']}")
        if config['noise_words']:
            print(f"  词汇样例: {', '.join(config['noise_words'][:5])}...")
    
    print("\n" + "=" * 80)
    print("【关键问题】")
    print("  Q1: M+S (26词) 能做到多少效果？")
    print("  Q2: 加B (13词) 后性能如何变？")
    print("  Q3: Anatomy (8词) 是否应该投影（还是有益信息）？")
    print("  Q4: 哪个分组贡献最大的 C_v 提升？")
    print("=" * 80)


if __name__ == "__main__":
    print_ablation_summary()
    
    # 输出分组统计
    groups = NoiseWordGroups.get_groups()
    print("\n【分组统计】")
    for name, words in groups.items():
        print(f"{name}: {len(words)} 个词")
    
    # 总词数
    all_words = set()
    for words in groups.values():
        all_words.update(words)
    print(f"总计（去重）: {len(all_words)} 个词")
