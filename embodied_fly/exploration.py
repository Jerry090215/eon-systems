#!/usr/bin/env python3
"""
探索驱动模块 - 果蝇的内在动机系统

模拟真实果蝇的自发探索行为：
  - 好奇心（curiosity）：随时间积累，驱动探索
  - 唤醒度（arousal）：由外部刺激和内在好奇心共同驱动
  - 疲劳度（fatigue）：长时间活动后需要休息
  - 随机游走（correlated random walk）：方向有持续性，不是完全随机

真实果蝇不会问造物主"需不需要探索"——探索是它的本能。
"""

import math
import time
import random
from dataclasses import dataclass, field


@dataclass
class ExplorationState:
    """探索驱动状态"""
    curiosity: float = 0.3          # 好奇心 0-1（随时间积累）
    arousal: float = 0.2            # 唤醒度 0-1
    fatigue: float = 0.0            # 疲劳度 0-1
    exploration_drive: float = 0.0  # 综合探索驱动力
    is_exploring: bool = False      # 是否在主动探索
    heading: float = 0.0            # 当前探索方向（弧度）
    heading_angular_vel: float = 0.0  # 方向角速度（持续性）
    last_novelty_time: float = 0.0  # 上次遇到新刺激的时间
    last_external_input_time: float = 0.0  # 上次有外部输入的时间
    rest_until: float = 0.0         # 休息到这个时间点
    total_distance: float = 0.0     # 累计探索距离


class ExplorationDriver:
    """
    探索驱动引擎

    核心机制：
    1. 好奇心随时间线性积累（每30秒从0到1）
    2. 遇到新刺激时好奇心降低（满足了好奇心）
    3. 探索驱动力 = curiosity * (1 - fatigue) * arousal
    4. 无外部输入时，探索驱动力驱动随机游走
    5. 有外部输入时，探索驱动让位于外部输入
    6. 疲劳度随活动时间增加，达到阈值后强制休息
    """

    def __init__(self):
        self.state = ExplorationState()
        self._start_time = time.monotonic()
        self._last_update = time.monotonic()

        # 参数
        self.curiosity_rate = 1.0 / 10.0      # 好奇心积累速率（每10秒满，更频繁探索）
        self.curiosity_decay_on_novelty = 0.3  # 遇到新刺激时好奇心降低量
        self.fatigue_rate = 1.0 / 180.0        # 疲劳积累速率（每180秒满，更耐玩）
        self.fatigue_recovery_rate = 1.0 / 20.0  # 疲劳恢复速率（每20秒清零）
        self.fatigue_threshold = 0.85           # 疲劳阈值，超过则休息（更高，更少休息）
        self.rest_duration = 5.0                # 休息时长（秒，更短）
        self.heading_persistence = 0.90         # 方向持续性（0-1，越高越直）
        self.max_angular_vel = 2.0              # 最大角速度（弧度/秒，更灵活）
        self.external_input_timeout = 1.0       # 外部输入超时（秒）

        # 随机数生成器（用于可复现的随机游走）
        self._rng = random.Random()
        self._heading_angular_vel = 0.0  # 方向角速度（持续性）

    def update(self, dt: float, has_external_input: bool, novelty: float = 0.0,
               punishment: bool = False, visual_looming: float = 0.0) -> ExplorationState:
        """
        更新探索驱动状态

        Args:
            dt: 时间步长（秒）
            has_external_input: 是否有外部输入（气味/惩罚/视觉等）
            novelty: 新奇度（0-1，窗口切换等）
            punishment: 是否有惩罚（电击）
            visual_looming: 视觉逼近强度（0-1）

        Returns:
            更新后的探索状态
        """
        now = time.monotonic()
        s = self.state

        # === 1. 好奇心积累 ===
        if not has_external_input:
            # 无外部输入时，好奇心积累
            s.curiosity = min(1.0, s.curiosity + self.curiosity_rate * dt)
        else:
            # 有外部输入时，记录时间
            s.last_external_input_time = now

        # === 2. 新奇刺激处理 ===
        if novelty > 0.3:
            # 遇到新刺激，好奇心降低（满足了），唤醒度增加
            s.curiosity = max(0.0, s.curiosity - self.curiosity_decay_on_novelty * novelty)
            s.arousal = min(1.0, s.arousal + novelty * 0.5)
            s.last_novelty_time = now
            # 新刺激触发方向变化（探索新方向）
            self._heading_angular_vel += self._rng.uniform(-0.8, 0.8) * novelty

        # === 3. 惩罚/逼近处理 ===
        if punishment or visual_looming > 0.5:
            # 危险刺激，唤醒度飙升，好奇心暂时抑制
            s.arousal = min(1.0, s.arousal + 0.5)
            s.curiosity = max(0.0, s.curiosity - 0.2)
            # 危险时方向随机变化（逃跑）
            self._heading_angular_vel += self._rng.uniform(-1.0, 1.0)

        # === 4. 唤醒度自然衰减 ===
        s.arousal = max(0.1, s.arousal - 0.1 * dt)
        # 好奇心驱动唤醒度
        s.arousal = max(s.arousal, s.curiosity * 0.5)

        # === 5. 疲劳度管理 ===
        if s.is_exploring and not has_external_input:
            # 主动探索时疲劳积累
            s.fatigue = min(1.0, s.fatigue + self.fatigue_rate * dt)
        else:
            # 休息或有外部输入时疲劳恢复
            s.fatigue = max(0.0, s.fatigue - self.fatigue_recovery_rate * dt)

        # 疲劳超过阈值，强制休息
        if s.fatigue > self.fatigue_threshold and now > s.rest_until:
            s.rest_until = now + self.rest_duration
            s.fatigue = 0.5  # 休息后疲劳降低

        # === 6. 综合探索驱动力 ===
        is_resting = now < s.rest_until
        external_input_active = (now - s.last_external_input_time) < self.external_input_timeout

        if is_resting:
            # 休息中，不探索
            s.exploration_drive = 0.0
            s.is_exploring = False
        elif external_input_active or has_external_input:
            # 有外部输入时，探索驱动让位于外部输入
            s.exploration_drive = 0.0
            s.is_exploring = False
        else:
            # 无外部输入且不休息时，探索驱动力驱动行为
            # 基础驱动力 + 好奇心驱动，确保即使好奇心低也有基本活动
            base_drive = 0.15  # 基础自发活动驱动力
            s.exploration_drive = base_drive + s.curiosity * (1.0 - s.fatigue) * (0.3 + s.arousal * 0.7)
            s.is_exploring = s.exploration_drive > 0.1  # 更低阈值，更容易进入探索

        # === 7. 随机游走方向更新 ===
        if s.is_exploring:
            # 相关随机游走：方向有持续性，偶尔变化
            noise = self._rng.gauss(0, 0.3) * dt
            self._heading_angular_vel = (
                self.heading_persistence * self._heading_angular_vel
                + (1 - self.heading_persistence) * noise * self.max_angular_vel
            )
            # 限制角速度
            self._heading_angular_vel = max(
                -self.max_angular_vel,
                min(self.max_angular_vel, self._heading_angular_vel)
            )
            s.heading += self._heading_angular_vel * dt
            # 归一化到 [-pi, pi]
            s.heading = (s.heading + math.pi) % (2 * math.pi) - math.pi

            # 累计探索距离
            s.total_distance += s.exploration_drive * dt * 50.0
        else:
            # 不探索时，角速度逐渐衰减
            self._heading_angular_vel *= 0.95

        self._last_update = now
        return s

    def get_exploration_behavior(self) -> dict:
        """
        获取探索行为输出（用于无外部输入时的行为决策）

        Returns:
            dict: {
                'walk_drive': 基础行走速度（0-1.5）,
                'turn_bias': 转向偏置（-1到1）,
                'arousal': 唤醒度,
                'is_exploring': 是否在探索,
                'curiosity': 好奇心,
                'fatigue': 疲劳度,
            }
        """
        s = self.state

        if not s.is_exploring:
            # 不探索时，缓慢游走（不是完全不动！真实果蝇即使静息也会微动）
            return {
                'walk_drive': 0.25 + 0.1 * math.sin(time.monotonic() * 0.5),
                'turn_bias': math.sin(time.monotonic() * 0.3) * 0.2,
                'arousal': s.arousal,
                'is_exploring': False,
                'curiosity': s.curiosity,
                'fatigue': s.fatigue,
            }

        # 探索时：速度由探索驱动力决定，转向由方向角速度决定
        walk_drive = 0.4 + s.exploration_drive * 1.2  # 0.4-1.6，更明显的移动
        # 转向：方向角速度映射到 turn_bias
        turn_bias = max(-1.0, min(1.0, self._heading_angular_vel / self.max_angular_vel))
        # 好奇心高时，转向更频繁（探索新方向）
        turn_bias *= 0.5 + s.curiosity * 0.5

        return {
            'walk_drive': round(walk_drive, 3),
            'turn_bias': round(turn_bias, 3),
            'arousal': round(s.arousal, 3),
            'is_exploring': True,
            'curiosity': round(s.curiosity, 3),
            'fatigue': round(s.fatigue, 3),
        }

    def reset(self):
        """重置探索状态"""
        self.state = ExplorationState()
        self._start_time = time.monotonic()
        self._last_update = time.monotonic()
        self._heading_angular_vel = 0.0
