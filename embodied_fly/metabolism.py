#!/usr/bin/env python3
"""
代谢系统模块 - 给果蝇真实的生存压力

包含：
  - 能量系统：运动消耗能量，食物恢复能量，耗尽则死亡
  - 疲劳系统：活动积累疲劳，休息恢复疲劳
  - 昼夜节律：白天活跃，晚上活动降低
  - 健康系统：碰撞/危险造成伤害，健康归零则死亡
  - 死亡与复活：能量/健康归零后死亡，一段时间后可"孵化"新个体

这是让果蝇"像活的"的核心——它有真实的生存需求，
不是程序员让它探索，而是为了活下去必须探索。
"""

import os
import sys
import time
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class FlyState(Enum):
    """果蝇生命状态"""
    ALIVE = "alive"           # 活着
    SLEEPING = "sleeping"     # 睡觉（疲劳过高或夜间）
    DYING = "dying"           # 垂死（能量/健康极低）
    DEAD = "dead"             # 死亡


@dataclass
class MetabolismState:
    """代谢状态"""
    # 核心生命指标（0.0 - 1.0）
    energy: float = 1.0       # 能量（饱食度）
    fatigue: float = 0.0      # 疲劳度
    health: float = 1.0       # 健康值

    # 状态
    state: FlyState = FlyState.ALIVE
    death_time: float = 0.0   # 死亡时间（用于复活计时）
    age: float = 0.0          # 年龄（秒）

    # 昼夜节律
    is_night: bool = False    # 是否夜间
    circadian_phase: float = 0.0  # 昼夜节律相位（0-1）

    # 统计
    total_distance: float = 0.0   # 总移动距离
    food_eaten: int = 0           # 吃到食物的次数
    collisions: int = 0           # 碰撞次数
    predator_escapes: int = 0     # 成功逃跑次数

    # 行为影响（输出给主循环）
    speed_modifier: float = 1.0   # 速度修正（疲劳/能量/昼夜）
    arousal_modifier: float = 1.0  # 唤醒度修正
    need_food: bool = False        # 是否需要食物（能量低）
    need_rest: bool = False        # 是否需要休息（疲劳高）
    is_dying: bool = False         # 是否垂死


class Metabolism:
    """
    代谢系统 - 管理果蝇的生命状态

    用法:
        meta = Metabolism()
        meta.update(dt, walk_intensity, on_food, food_rate,
                    collision, danger, predator_escape)
        state = meta.state
        # 使用 state.speed_modifier, state.need_food 等
    """

    # 参数
    ENERGY_DECAY_BASE = 0.002     # 基础能量消耗（每秒）
    ENERGY_DECAY_MOVEMENT = 0.008  # 运动能量消耗系数
    FATIGUE_GAIN_BASE = 0.001      # 基础疲劳积累
    FATIGUE_GAIN_MOVEMENT = 0.005  # 运动疲劳积累系数
    FATIGUE_RECOVERY = 0.02        # 休息时疲劳恢复速率
    HEALTH_RECOVERY = 0.005        # 健康自然恢复速率
    HEALTH_DAMAGE_COLLISION = 0.02  # 碰撞伤害
    HEALTH_DAMAGE_DANGER = 0.03    # 危险区域伤害
    HEALTH_DAMAGE_PREDATOR = 0.05  # 捕食者伤害（被抓到）

    # 阈值
    ENERGY_LOW = 0.3       # 低能量阈值
    ENERGY_CRITICAL = 0.1  # 临界能量阈值
    FATIGUE_HIGH = 0.7     # 高疲劳阈值
    FATIGUE_CRITICAL = 0.9 # 临界疲劳阈值
    HEALTH_CRITICAL = 0.2  # 临界健康阈值

    # 昼夜节律（基于系统时间）
    DAY_START_HOUR = 7     # 白天开始（7点）
    NIGHT_START_HOUR = 22  # 夜间开始（22点）

    # 复活
    RESPAWN_DELAY = 30.0   # 死亡后复活延迟（秒）

    def __init__(self):
        self.state = MetabolismState()
        self._last_update = time.monotonic()
        self._birth_time = time.monotonic()

    def update(self, dt: float,
               walk_intensity: float = 0.0,
               on_food: bool = False,
               food_recovery_rate: float = 0.0,
               is_colliding: bool = False,
               in_danger: bool = False,
               predator_caught: bool = False,
               is_resting: bool = False):
        """
        更新代谢状态

        Args:
            dt: 时间步长（秒）
            walk_intensity: 运动强度（0-1）
            on_food: 是否在食物源上
            food_recovery_rate: 食物恢复速率
            is_colliding: 是否碰撞
            in_danger: 是否在危险区域
            predator_caught: 是否被捕食者抓到
            is_resting: 是否在休息
        """
        s = self.state

        if s.state == FlyState.DEAD:
            # 死亡状态：检查是否复活
            if time.monotonic() - s.death_time > self.RESPAWN_DELAY:
                self._respawn()
            return

        # 更新年龄
        s.age += dt

        # 1. 能量消耗与恢复
        energy_decay = (self.ENERGY_DECAY_BASE +
                        self.ENERGY_DECAY_MOVEMENT * walk_intensity)
        s.energy -= energy_decay * dt

        if on_food and food_recovery_rate > 0:
            s.energy += food_recovery_rate * dt
            if s.energy >= 0.95 and not on_food:
                pass  # 吃饱了
            # 统计吃到食物
            if s.energy < 0.99:
                s.food_eaten += 1

        s.energy = max(0.0, min(1.0, s.energy))

        # 2. 疲劳积累与恢复
        if is_resting or s.state == FlyState.SLEEPING:
            s.fatigue -= self.FATIGUE_RECOVERY * dt
        else:
            fatigue_gain = (self.FATIGUE_GAIN_BASE +
                            self.FATIGUE_GAIN_MOVEMENT * walk_intensity)
            s.fatigue += fatigue_gain * dt

        s.fatigue = max(0.0, min(1.0, s.fatigue))

        # 3. 健康恢复与伤害
        if not is_colliding and not in_danger and not predator_caught:
            s.health += self.HEALTH_RECOVERY * dt

        if is_colliding:
            s.health -= self.HEALTH_DAMAGE_COLLISION * dt * 10  # 碰撞持续伤害
            s.collisions += 1

        if in_danger:
            s.health -= self.HEALTH_DAMAGE_DANGER * dt * 10

        if predator_caught:
            s.health -= self.HEALTH_DAMAGE_PREDATOR
            s.predator_escapes += 1  # 假设抓到后也能逃脱（扣血）

        s.health = max(0.0, min(1.0, s.health))

        # 4. 昼夜节律
        self._update_circadian()

        # 5. 状态判断
        self._update_state()

        # 6. 行为影响计算
        self._compute_behavior_modifiers()

        # 7. 死亡检查
        if s.energy <= 0 or s.health <= 0:
            self._die()

    def _update_circadian(self):
        """更新昼夜节律（基于系统时间）"""
        s = self.state
        now = time.localtime()
        hour = now.tm_hour + now.tm_min / 60.0

        # 计算昼夜相位（0=午夜, 0.5=正午, 1=午夜）
        s.circadian_phase = (hour % 24) / 24.0

        # 判断白天/夜间
        if self.DAY_START_HOUR <= hour < self.NIGHT_START_HOUR:
            s.is_night = False
        else:
            s.is_night = True

    def _update_state(self):
        """更新生命状态"""
        s = self.state

        if s.state == FlyState.DEAD:
            return

        # 垂死状态
        if s.energy < self.ENERGY_CRITICAL or s.health < self.HEALTH_CRITICAL:
            s.state = FlyState.DYING
            s.is_dying = True
            return

        # 睡眠状态（疲劳过高或夜间）
        if s.fatigue > self.FATIGUE_CRITICAL or (s.is_night and s.fatigue > 0.3):
            s.state = FlyState.SLEEPING
            s.is_dying = False
            return

        # 正常活着
        s.state = FlyState.ALIVE
        s.is_dying = False

    def _compute_behavior_modifiers(self):
        """计算行为影响系数"""
        s = self.state

        # 速度修正
        speed = 1.0

        # 低能量减速
        if s.energy < self.ENERGY_LOW:
            speed *= 0.5 + 0.5 * (s.energy / self.ENERGY_LOW)

        # 高疲劳减速
        if s.fatigue > self.FATIGUE_HIGH:
            speed *= 0.4 + 0.6 * ((1.0 - s.fatigue) / (1.0 - self.FATIGUE_HIGH))

        # 夜间减速
        if s.is_night:
            speed *= 0.6

        # 垂死状态极慢
        if s.state == FlyState.DYING:
            speed *= 0.2

        # 睡眠状态不动
        if s.state == FlyState.SLEEPING:
            speed = 0.0

        s.speed_modifier = max(0.0, min(1.5, speed))

        # 唤醒度修正
        arousal = 1.0
        if s.fatigue > 0.5:
            arousal *= 1.0 - (s.fatigue - 0.5) * 0.8
        if s.is_night:
            arousal *= 0.7
        if s.state == FlyState.SLEEPING:
            arousal = 0.1
        if s.state == FlyState.DYING:
            arousal *= 0.3

        s.arousal_modifier = max(0.0, min(1.0, arousal))

        # 需求判断
        s.need_food = s.energy < self.ENERGY_LOW
        s.need_rest = s.fatigue > self.FATIGUE_HIGH

    def _die(self):
        """死亡"""
        s = self.state
        s.state = FlyState.DEAD
        s.death_time = time.monotonic()
        s.speed_modifier = 0.0
        s.arousal_modifier = 0.0
        print(f"\n  💀 果蝇死亡！年龄: {s.age:.0f}秒, "
              f"能量: {s.energy:.2f}, 健康: {s.health:.2f}\n")

    def _respawn(self):
        """复活（孵化新个体）"""
        s = self.state
        print(f"\n  🥚 新果蝇孵化！（上一只存活了 {s.age:.0f}秒）\n")
        self.state = MetabolismState()
        self._birth_time = time.monotonic()

    def get_state_dict(self) -> dict:
        """获取状态字典"""
        s = self.state
        return {
            "energy": round(s.energy, 3),
            "fatigue": round(s.fatigue, 3),
            "health": round(s.health, 3),
            "state": s.state.value,
            "age": round(s.age, 1),
            "is_night": s.is_night,
            "speed_modifier": round(s.speed_modifier, 3),
            "arousal_modifier": round(s.arousal_modifier, 3),
            "need_food": s.need_food,
            "need_rest": s.need_rest,
            "is_dying": s.is_dying,
            "food_eaten": s.food_eaten,
            "collisions": s.collisions,
        }


if __name__ == "__main__":
    # 测试
    meta = Metabolism()
    print("=== 代谢系统测试 ===")

    # 模拟 60 秒
    for i in range(600):
        meta.update(
            dt=0.1,
            walk_intensity=0.5,
            on_food=(i % 200 < 50),  # 每 20 秒有 5 秒在食物上
            food_recovery_rate=0.05,
            is_colliding=False,
            in_danger=False,
        )

    state = meta.state
    print(f"60秒后:")
    print(f"  能量: {state.energy:.3f}")
    print(f"  疲劳: {state.fatigue:.3f}")
    print(f"  健康: {state.health:.3f}")
    print(f"  状态: {state.state.value}")
    print(f"  速度修正: {state.speed_modifier:.3f}")
    print(f"  需要食物: {state.need_food}")
    print(f"  需要休息: {state.need_rest}")
    print(f"  吃到食物次数: {state.food_eaten}")
