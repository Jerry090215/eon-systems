#!/usr/bin/env python3
"""
可塑基因型 (Plastic Genome)

单果蝇自主进化的核心：35个超参数控制整个大脑的工作模式。
每5分钟发生一次体细胞突变，好的保留，差的回退。

这不是达尔文式的种群进化，而是个体内部的体细胞进化
——对应真实生物大脑的终身神经可塑性。
"""

import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class GeneParameter:
    """单个基因参数"""
    name: str
    value: float
    min_val: float
    max_val: float
    default_val: float
    mutation_sigma: float  # 突变的高斯噪声标准差（相对于范围的比例）
    category: str  # 分类：visual/learning/metabolism/behavior/neuromodulation/motor
    description: str = ""

    def mutate(self, temperature: float = 1.0) -> float:
        """
        发生突变（高斯噪声）

        Args:
            temperature: 突变温度（1.0=正常，>1=更激进，<1=更保守）

        Returns:
            突变后的值
        """
        range_size = self.max_val - self.min_val
        sigma = self.mutation_sigma * range_size * temperature
        delta = random.gauss(0, sigma)
        new_val = self.value + delta
        # clamp 到范围
        new_val = max(self.min_val, min(self.max_val, new_val))
        return new_val

    def clone(self) -> "GeneParameter":
        """克隆（深拷贝）"""
        return GeneParameter(
            name=self.name,
            value=self.value,
            min_val=self.min_val,
            max_val=self.max_val,
            default_val=self.default_val,
            mutation_sigma=self.mutation_sigma,
            category=self.category,
            description=self.description,
        )


class PlasticGenome:
    """
    可塑基因型

    包含35个超参数，控制大脑的视觉、学习、代谢、行为、神经调制、运动系统。
    支持体细胞突变和适应度评估。
    """

    def __init__(self):
        self.params: Dict[str, GeneParameter] = {}
        self._init_default_params()
        self.generation = 0  # 已经历的突变代数
        self.birth_time = time.monotonic()
        self.total_mutations = 0
        self.accepted_mutations = 0
        self.rejected_mutations = 0

    def _init_default_params(self):
        """初始化默认基因型参数"""

        # === 视觉系统 ===
        self._add_param("visual_escape_sensitivity", 0.7, 0.2, 1.0, 0.7, 0.03,
                        "visual", "视觉逃逸敏感度（LC4阈值，越高越容易被吓到）")
        self._add_param("visual_motion_gain", 1.0, 0.3, 2.5, 1.0, 0.04,
                        "visual", "运动检测增益（T4/T5反应强度）")
        self._add_param("looming_response_strength", 0.8, 0.2, 1.5, 0.8, 0.03,
                        "visual", "Looming逼近反应强度")

        # === 学习系统 ===
        self._add_param("stdp_learning_rate", 0.0008, 0.0001, 0.003, 0.0008, 0.05,
                        "learning", "STDP突触可塑性学习率")
        self._add_param("dopamine_reward_sensitivity", 0.7, 0.2, 1.5, 0.7, 0.04,
                        "learning", "多巴胺奖励敏感度（PAM DAN增益）")
        self._add_param("punishment_sensitivity", 0.8, 0.2, 1.5, 0.8, 0.04,
                        "learning", "惩罚敏感度（PPL1 DAN增益）")
        self._add_param("memory_decay_rate", 0.001, 0.0001, 0.005, 0.001, 0.05,
                        "learning", "记忆衰减率（权重遗忘速度）")

        # === 代谢系统 ===
        self._add_param("base_metabolism", 0.002, 0.0005, 0.006, 0.002, 0.04,
                        "metabolism", "基础代谢率（每秒能量消耗）")
        self._add_param("movement_energy_cost", 0.008, 0.002, 0.02, 0.008, 0.04,
                        "metabolism", "运动能量消耗系数")
        self._add_param("food_absorption_rate", 1.0, 0.3, 2.5, 1.0, 0.04,
                        "metabolism", "食物吸收效率（在食物源上恢复能量的速度）")
        self._add_param("fatigue_accumulation_rate", 0.005, 0.001, 0.015, 0.005, 0.04,
                        "metabolism", "疲劳积累率（活动时疲劳增长速度）")

        # === 行为倾向 ===
        self._add_param("exploration_rate", 0.5, 0.05, 1.0, 0.5, 0.05,
                        "behavior", "探索率（好奇心强度，越高越爱动）")
        self._add_param("spontaneous_turn_prob", 0.15, 0.01, 0.5, 0.15, 0.05,
                        "behavior", "自发转弯概率（随机转向的频率）")
        self._add_param("risk_tolerance", 0.5, 0.05, 1.0, 0.5, 0.05,
                        "behavior", "风险容忍度（越高越敢靠近危险/捕食者）")

        # === 神经调制 ===
        self._add_param("baseline_arousal", 0.3, 0.05, 0.8, 0.3, 0.04,
                        "neuromodulation", "基线唤醒度（静息时的警觉水平）")
        self._add_param("baseline_dopamine", 0.5, 0.1, 1.0, 0.5, 0.04,
                        "neuromodulation", "基线多巴胺水平（影响学习和动机）")
        self._add_param("stress_response_gain", 1.0, 0.3, 2.5, 1.0, 0.04,
                        "neuromodulation", "压力反应增益（被吓时唤醒度上升幅度）")
        self._add_param("recovery_rate", 0.02, 0.005, 0.08, 0.02, 0.05,
                        "neuromodulation", "压力恢复率（受惊后恢复平静的速度）")

        # === 运动系统 ===
        self._add_param("max_walk_speed", 1.0, 0.3, 2.0, 1.0, 0.04,
                        "motor", "最大爬行速度系数")
        self._add_param("turn_agility", 1.0, 0.3, 2.5, 1.0, 0.04,
                        "motor", "转向敏捷度（转向速度系数）")
        self._add_param("wing_flap_threshold", 0.6, 0.2, 1.0, 0.6, 0.04,
                        "motor", "振翅阈值（唤醒度超过此值时振翅）")

    def _add_param(self, name, value, min_val, max_val, default_val,
                   mutation_sigma, category, description=""):
        """添加一个基因参数"""
        self.params[name] = GeneParameter(
            name=name,
            value=value,
            min_val=min_val,
            max_val=max_val,
            default_val=default_val,
            mutation_sigma=mutation_sigma,
            category=category,
            description=description,
        )

    def get(self, name: str) -> float:
        """获取参数值"""
        if name in self.params:
            return self.params[name].value
        raise KeyError(f"未知基因参数: {name}")

    def set(self, name: str, value: float):
        """设置参数值（用于突变后应用）"""
        if name in self.params:
            p = self.params[name]
            self.params[name].value = max(p.min_val, min(p.max_val, value))
        else:
            raise KeyError(f"未知基因参数: {name}")

    def mutate(self, temperature: float = 1.0, mutate_count: Optional[int] = None) -> Dict[str, Tuple[float, float]]:
        """
        发生体细胞突变

        Args:
            temperature: 突变温度（1.0=正常，>1=更激进，<1=更保守）
            mutate_count: 突变的参数数量，None=随机1-3个

        Returns:
            突变字典: {param_name: (old_value, new_value)}
        """
        if mutate_count is None:
            mutate_count = random.randint(1, 3)

        # 随机选择要突变的参数
        param_names = list(self.params.keys())
        selected = random.sample(param_names, min(mutate_count, len(param_names)))

        mutations = {}
        for name in selected:
            old_val = self.params[name].value
            new_val = self.params[name].mutate(temperature)
            mutations[name] = (old_val, new_val)

        self.total_mutations += 1
        return mutations

    def apply_mutations(self, mutations: Dict[str, Tuple[float, float]]):
        """应用突变（把新值写入参数）"""
        for name, (old_val, new_val) in mutations.items():
            if name in self.params:
                self.params[name].value = new_val

    def revert_mutations(self, mutations: Dict[str, Tuple[float, float]]):
        """回退突变（恢复旧值）"""
        for name, (old_val, new_val) in mutations.items():
            if name in self.params:
                self.params[name].value = old_val

    def clone(self) -> "PlasticGenome":
        """克隆基因型（深拷贝）"""
        new_genome = PlasticGenome()
        new_genome.params = {name: p.clone() for name, p in self.params.items()}
        new_genome.generation = self.generation
        new_genome.birth_time = self.birth_time
        new_genome.total_mutations = self.total_mutations
        new_genome.accepted_mutations = self.accepted_mutations
        new_genome.rejected_mutations = self.rejected_mutations
        return new_genome

    def get_summary(self) -> Dict:
        """获取基因型摘要（用于日志和显示）"""
        categories = {}
        for name, p in self.params.items():
            if p.category not in categories:
                categories[p.category] = {}
            categories[p.category][name] = round(p.value, 4)

        return {
            "generation": self.generation,
            "total_mutations": self.total_mutations,
            "accepted_mutations": self.accepted_mutations,
            "rejected_mutations": self.rejected_mutations,
            "acceptance_rate": (self.accepted_mutations / max(1, self.total_mutations)),
            "age_seconds": time.monotonic() - self.birth_time,
            "categories": categories,
        }

    def get_personality_description(self) -> str:
        """根据基因型生成性格描述（人类可读）"""
        traits = []

        # 视觉/恐惧
        if self.get("visual_escape_sensitivity") > 0.8:
            traits.append("胆小易惊")
        elif self.get("visual_escape_sensitivity") < 0.4:
            traits.append("胆大无畏")

        # 探索
        if self.get("exploration_rate") > 0.7:
            traits.append("好奇好动")
        elif self.get("exploration_rate") < 0.3:
            traits.append("安静宅居")

        # 代谢
        if self.get("base_metabolism") > 0.004:
            traits.append("高代谢需频繁进食")
        elif self.get("base_metabolism") < 0.001:
            traits.append("低代谢耐饿")

        # 学习
        if self.get("stdp_learning_rate") > 0.0015:
            traits.append("学得快忘得也快")
        elif self.get("stdp_learning_rate") < 0.0004:
            traits.append("学得慢但记得牢")

        # 风险
        if self.get("risk_tolerance") > 0.7:
            traits.append("爱冒险")
        elif self.get("risk_tolerance") < 0.3:
            traits.append("谨慎保守")

        # 压力恢复
        if self.get("recovery_rate") > 0.04:
            traits.append("心态恢复快")
        elif self.get("recovery_rate") < 0.01:
            traits.append("容易长期焦虑")

        return "、".join(traits) if traits else "性格中庸"

    def save(self, path: str):
        """保存基因型到文件"""
        data = {
            "generation": self.generation,
            "birth_time": self.birth_time,
            "total_mutations": self.total_mutations,
            "accepted_mutations": self.accepted_mutations,
            "rejected_mutations": self.rejected_mutations,
            "params": {name: p.value for name, p in self.params.items()},
        }
        with open(path, 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        """从文件加载基因型"""
        with open(path, 'r') as f:
            data = json.load(f)
        self.generation = data.get("generation", 0)
        self.birth_time = data.get("birth_time", time.monotonic())
        self.total_mutations = data.get("total_mutations", 0)
        self.accepted_mutations = data.get("accepted_mutations", 0)
        self.rejected_mutations = data.get("rejected_mutations", 0)
        for name, value in data.get("params", {}).items():
            if name in self.params:
                self.params[name].value = value


@dataclass
class FitnessMetrics:
    """适应度指标（用于评估突变的好坏）"""
    energy_level: float = 0.5  # 平均能量水平（越高越好）
    energy_trend: float = 0.0  # 能量变化趋势（正=在恢复，负=在消耗）
    punishment_count: int = 0  # 被惩罚次数（被吓/碰撞，越低越好）
    punishment_intensity: float = 0.0  # 平均惩罚强度
    exploration_range: float = 0.0  # 探索范围（活动覆盖的面积，适度为好）
    food_visits: int = 0  # 找到食物源的次数（越高越好）
    collision_count: int = 0  # 碰撞次数（越低越好）
    rest_ratio: float = 0.0  # 休息时间比例（适度为好）
    avg_arousal: float = 0.0  # 平均唤醒度（适度为好）

    def compute_fitness(self) -> float:
        """
        计算综合适应度分数（越高越好）

        这是隐式自然选择的核心——不是人工打分，
        而是基于生存指标的综合评估。
        """
        score = 0.0

        # 能量水平（权重最高，生存第一）
        score += self.energy_level * 3.0

        # 能量趋势（在恢复加分，在消耗减分）
        score += self.energy_trend * 5.0

        # 惩罚次数（越少越好）
        score -= self.punishment_count * 0.3
        score -= self.punishment_intensity * 0.5

        # 食物访问（找到食物加分）
        score += self.food_visits * 0.2

        # 碰撞次数（越少越好）
        score -= self.collision_count * 0.2

        # 探索范围（适度为好，太少=宅，太多=浪费能量）
        optimal_exploration = 500.0  # 像素平方
        exploration_deviation = abs(self.exploration_range - optimal_exploration)
        score -= exploration_deviation * 0.001

        # 休息比例（适度为好）
        optimal_rest = 0.3
        rest_deviation = abs(self.rest_ratio - optimal_rest)
        score -= rest_deviation * 1.0

        # 平均唤醒度（适度为好，太低=抑郁，太高=焦虑）
        optimal_arousal = 0.4
        arousal_deviation = abs(self.avg_arousal - optimal_arousal)
        score -= arousal_deviation * 1.0

        return score

    def reset(self):
        """重置指标（开始新的评估周期）"""
        self.energy_level = 0.5
        self.energy_trend = 0.0
        self.punishment_count = 0
        self.punishment_intensity = 0.0
        self.exploration_range = 0.0
        self.food_visits = 0
        self.collision_count = 0
        self.rest_ratio = 0.0
        self.avg_arousal = 0.0


def test_genome():
    """测试基因型模块"""
    print("=" * 60)
    print("可塑基因型测试")
    print("=" * 60)

    genome = PlasticGenome()
    print(f"\n参数数量: {len(genome.params)}")
    print(f"分类: {set(p.category for p in genome.params.values())}")

    print(f"\n默认性格: {genome.get_personality_description()}")

    # 测试突变
    print("\n--- 测试突变 ---")
    for i in range(5):
        mutations = genome.mutate(temperature=1.0)
        print(f"  突变 {i+1}: {len(mutations)} 个参数")
        for name, (old, new) in mutations.items():
            print(f"    {name}: {old:.4f} → {new:.4f} (Δ={new-old:+.4f})")
        genome.apply_mutations(mutations)
        genome.accepted_mutations += 1
        genome.generation += 1

    print(f"\n突变后性格: {genome.get_personality_description()}")

    # 测试适应度
    print("\n--- 测试适应度评估 ---")
    fitness = FitnessMetrics(
        energy_level=0.8,
        energy_trend=0.01,
        punishment_count=1,
        punishment_intensity=0.3,
        exploration_range=400,
        food_visits=3,
        collision_count=0,
        rest_ratio=0.25,
        avg_arousal=0.35,
    )
    print(f"  适应度分数: {fitness.compute_fitness():.3f}")
    print(f"  （高能量+少惩罚+适度探索=高适应度）")

    # 测试保存/加载
    print("\n--- 测试保存/加载 ---")
    test_path = "/tmp/test_genome.json"
    genome.save(test_path)
    print(f"  已保存到 {test_path}")

    genome2 = PlasticGenome()
    genome2.load(test_path)
    print(f"  加载后代数: {genome2.generation}")
    print(f"  加载后性格: {genome2.get_personality_description()}")

    # 清理
    os.remove(test_path)

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    test_genome()
