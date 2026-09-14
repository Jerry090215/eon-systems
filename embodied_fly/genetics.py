#!/usr/bin/env python3
"""
遗传算法模块 - 果蝇繁衍与进化

模拟真实果蝇的基因遗传：
  - 交叉互换（Crossover）：父母突触权重的自由组合
  - 随机变异（Mutation）：1-2% 高斯变异，模拟DNA复制错误
  - 受精卵（Fertilized Egg）：包含父母遗传信息，孵化后出生新个体
  - 孵化期（Incubation）：受精卵需要一定时间孵化

参考：果蝇遗传学、Holland 遗传算法、神经演化（Neuroevolution）
"""

import os
import json
import time
import math
import uuid
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from courtship import Gender


# ============================================================
# 受精卵数据结构
# ============================================================

@dataclass
class FertilizedEgg:
    """
    受精卵 - 包含父母遗传信息的胚胎

    真实果蝇的受精卵在24小时内孵化成幼虫，经过4天幼虫期和5天蛹期后羽化为成虫。
    这里简化为：孵化期后直接出生为成体果蝇。
    """
    egg_id: str                        # 唯一ID
    mother_id: str                     # 母亲ID
    father_id: str                     # 父亲ID
    conception_time: float             # 受孕时间（monotonic）
    hatch_time: float                  # 孵化时间（monotonic）
    incubation_period: float           # 孵化期（秒）
    weights: np.ndarray                # 交叉互换+变异后的KC→MBON突触权重
    gender: Gender                     # 性别（随机）
    birth_position: tuple              # 出生位置 (x, y)
    mother_final_eval: float = 0.0    # 母亲对父亲的最终评估分（遗传质量指标）
    father_song_freq: float = 0.0      # 父亲求偶之歌频率（可遗传特征）
    hatched: bool = False              # 是否已孵化
    generation: int = 1                # 代数（第1代=初代的孩子）

    def is_ready_to_hatch(self, current_time: float) -> bool:
        """是否准备好孵化"""
        return not self.hatched and current_time >= self.hatch_time

    def get_remaining_time(self, current_time: float) -> float:
        """获取剩余孵化时间（秒）"""
        return max(0.0, self.hatch_time - current_time)

    def get_progress(self, current_time: float) -> float:
        """获取孵化进度（0-1）"""
        if self.hatched:
            return 1.0
        elapsed = current_time - self.conception_time
        return min(1.0, elapsed / self.incubation_period)


# ============================================================
# 遗传算法引擎
# ============================================================

class GeneticsEngine:
    """
    遗传算法引擎 - 负责突触权重的交叉互换与变异

    核心原理：
      1. 交叉互换（Uniform Crossover）：每条突触连接随机来自父亲或母亲
         （模拟生物减数分裂时的基因自由组合）
      2. 随机变异（Gaussian Mutation）：随机选择1-2%的连接，
         对权重添加高斯噪声（模拟DNA复制时的随机错误）
      3. 权重约束：变异后权重保持在 [0, weight_max] 范围内
    """

    def __init__(self, mutation_rate: float = 0.015, mutation_std: float = 0.1,
                 weight_min: float = 0.0, weight_max: float = 5.0,
                 incubation_period: float = 120.0):
        """
        Args:
            mutation_rate: 变异率（每条连接被变异的概率，默认1.5%）
            mutation_std: 变异高斯噪声标准差
            weight_min: 权重最小值
            weight_max: 权重最大值
            incubation_period: 孵化期（秒，默认120秒=2分钟）
        """
        self.mutation_rate = mutation_rate
        self.mutation_std = mutation_std
        self.weight_min = weight_min
        self.weight_max = weight_max
        self.incubation_period = incubation_period

    def crossover(self, weights_mother: np.ndarray, weights_father: np.ndarray) -> np.ndarray:
        """
        均匀交叉互换（Uniform Crossover）

        每条突触连接随机来自母亲或父亲，模拟生物基因自由组合定律。

        Args:
            weights_mother: 母亲的KC→MBON权重数组
            weights_father: 父亲的KC→MBON权重数组

        Returns:
            交叉互换后的权重数组
        """
        assert len(weights_mother) == len(weights_father), \
            f"权重长度不匹配: mother={len(weights_mother)}, father={len(weights_father)}"

        n = len(weights_mother)
        # 随机选择每条连接来自母亲还是父亲（50%概率）
        mask = np.random.random(n) < 0.5
        offspring = np.where(mask, weights_mother, weights_father)
        return offspring.copy()

    def mutate(self, weights: np.ndarray) -> np.ndarray:
        """
        高斯变异（Gaussian Mutation）

        随机选择 mutation_rate 比例的连接，对权重添加高斯噪声。
        模拟DNA复制时的随机错误，是进化的原材料。

        Args:
            weights: 输入权重数组

        Returns:
            变异后的权重数组
        """
        mutated = weights.copy()
        n = len(weights)

        # 随机选择要变异的连接
        mutation_mask = np.random.random(n) < self.mutation_rate
        mutation_count = np.sum(mutation_mask)

        if mutation_count > 0:
            # 对选中的连接添加高斯噪声
            noise = np.random.normal(0, self.mutation_std, mutation_count)
            mutated[mutation_mask] += noise

            # 权重约束（保持在合理范围内）
            mutated = np.clip(mutated, self.weight_min, self.weight_max)

        return mutated

    def create_egg(self, weights_mother: np.ndarray, weights_father: np.ndarray,
                   mother_id: str, father_id: str,
                   birth_position: tuple = (0, 0),
                   mother_eval: float = 0.5, father_song_freq: float = 220.0,
                   generation: int = 1) -> FertilizedEgg:
        """
        创建受精卵

        流程：
          1. 交叉互换父母权重
          2. 随机变异
          3. 随机分配性别
          4. 设置孵化时间

        Args:
            weights_mother: 母亲的KC→MBON权重
            weights_father: 父亲的KC→MBON权重
            mother_id: 母亲ID
            father_id: 父亲ID
            birth_position: 出生位置
            mother_eval: 母亲对父亲的评估分（遗传质量）
            father_song_freq: 父亲求偶之歌频率
            generation: 代数

        Returns:
            受精卵对象
        """
        now = time.monotonic()

        # 1. 交叉互换
        crossed = self.crossover(weights_mother, weights_father)

        # 2. 随机变异
        mutated = self.mutate(crossed)

        # 3. 随机性别（50%概率）
        gender = Gender.MALE if np.random.random() < 0.5 else Gender.FEMALE

        # 4. 创建受精卵
        egg = FertilizedEgg(
            egg_id=str(uuid.uuid4())[:8],
            mother_id=mother_id,
            father_id=father_id,
            conception_time=now,
            hatch_time=now + self.incubation_period,
            incubation_period=self.incubation_period,
            weights=mutated,
            gender=gender,
            birth_position=birth_position,
            mother_final_eval=mother_eval,
            father_song_freq=father_song_freq,
            generation=generation,
        )

        return egg

    def save_egg(self, egg: FertilizedEgg, directory: str) -> str:
        """
        保存受精卵到文件（.egg 格式，实际是.npz）

        Args:
            egg: 受精卵对象
            directory: 保存目录

        Returns:
            保存的文件路径
        """
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"egg_{egg.egg_id}.npz")

        # 保存元数据（JSON格式的字符串）
        metadata = {
            "egg_id": egg.egg_id,
            "mother_id": egg.mother_id,
            "father_id": egg.father_id,
            "conception_time": egg.conception_time,
            "hatch_time": egg.hatch_time,
            "incubation_period": egg.incubation_period,
            "gender": egg.gender.value,
            "birth_position": list(egg.birth_position),
            "mother_final_eval": egg.mother_final_eval,
            "father_song_freq": egg.father_song_freq,
            "hatched": egg.hatched,
            "generation": egg.generation,
        }

        np.savez(path, weights=egg.weights, metadata=json.dumps(metadata))
        return path

    def load_egg(self, path: str) -> Optional[FertilizedEgg]:
        """
        从文件加载受精卵

        Args:
            path: .npz 文件路径

        Returns:
            受精卵对象，加载失败返回None
        """
        try:
            data = np.load(path, allow_pickle=True)
            metadata = json.loads(str(data["metadata"]))
            weights = data["weights"]

            egg = FertilizedEgg(
                egg_id=metadata["egg_id"],
                mother_id=metadata["mother_id"],
                father_id=metadata["father_id"],
                conception_time=float(metadata["conception_time"]),
                hatch_time=float(metadata["hatch_time"]),
                incubation_period=float(metadata["incubation_period"]),
                weights=weights,
                gender=Gender(metadata["gender"]),
                birth_position=tuple(metadata["birth_position"]),
                mother_final_eval=float(metadata.get("mother_final_eval", 0.5)),
                father_song_freq=float(metadata.get("father_song_freq", 220.0)),
                hatched=bool(metadata.get("hatched", False)),
                generation=int(metadata.get("generation", 1)),
            )
            return egg
        except Exception as e:
            print(f"[Genetics] 加载受精卵失败: {path}, 错误: {e}")
            return None

    def list_eggs(self, directory: str) -> list:
        """列出目录中所有未孵化的受精卵"""
        eggs = []
        if not os.path.exists(directory):
            return eggs
        for filename in os.listdir(directory):
            if filename.endswith(".npz") and filename.startswith("egg_"):
                path = os.path.join(directory, filename)
                egg = self.load_egg(path)
                if egg and not egg.hatched:
                    eggs.append(egg)
        return eggs


# ============================================================
# 进化统计
# ============================================================

@dataclass
class EvolutionStats:
    """进化统计信息"""
    total_births: int = 0
    total_deaths: int = 0
    generation: int = 1
    avg_mutation_rate: float = 0.0
    weight_diversity: float = 0.0  # 种群权重多样性（标准差）
    lineage: list = field(default_factory=list)  # 谱系记录


class EvolutionTracker:
    """
    进化追踪器 - 记录种群的进化历史

    用于观察多代繁衍后，种群的神经权重是否发生了适应性进化。
    """

    def __init__(self):
        self.stats = EvolutionStats()
        self.history = []  # 每代的统计信息

    def record_birth(self, egg: FertilizedEgg):
        """记录出生"""
        self.stats.total_births += 1
        self.stats.generation = max(self.stats.generation, egg.generation)
        self.history.append({
            "event": "birth",
            "egg_id": egg.egg_id,
            "gender": egg.gender.value,
            "generation": egg.generation,
            "time": time.monotonic(),
        })

    def record_death(self, fly_id: str, cause: str = "unknown"):
        """记录死亡"""
        self.stats.total_deaths += 1
        self.history.append({
            "event": "death",
            "fly_id": fly_id,
            "cause": cause,
            "time": time.monotonic(),
        })

    def record_mating(self, mother_id: str, father_id: str, egg_id: str):
        """记录交配"""
        self.history.append({
            "event": "mating",
            "mother_id": mother_id,
            "father_id": father_id,
            "egg_id": egg_id,
            "time": time.monotonic(),
        })

    def get_summary(self) -> dict:
        """获取进化统计摘要"""
        return {
            "total_births": self.stats.total_births,
            "total_deaths": self.stats.total_deaths,
            "current_generation": self.stats.generation,
            "population_alive": self.stats.total_births - self.stats.total_deaths,
            "history_length": len(self.history),
        }
