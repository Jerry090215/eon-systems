"""
运动控制层 (Motor Control Layer) —— 果蝇行为精细化控制

接收蘑菇体、中央复合体和感官输入，输出精细化的运动指令：
1. 步态调整：根据速度和转向调整步态模式
2. 梳理翅膀：静息或新奇刺激后触发梳理行为
3. 起飞/降落：强惩罚/强趋向时起飞，安全时降落
4. 后退：回避或边界时触发后退
5. 静止/睡眠：低兴奋性或夜晚时进入静止

真实果蝇的运动控制由腹神经索（VNC）中的中枢模式发生器（CPG）实现，
这里用简化的行为状态机实现核心功能。
"""

import math
import time
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional


@dataclass
class MotorState:
    """运动控制状态"""
    # 基础运动
    walk_drive: float = 0.0      # 行走驱动力 (0-1.5)
    turn_bias: float = 0.0       # 转向偏置 (-1 到 1)
    backward: bool = False        # 是否后退

    # 翅膀行为
    wing_drive: float = 0.0      # 振翅驱动力 (0-1.5)
    groom_drive: float = 0.0     # 梳理驱动力 (0-1)
    flying: bool = False          # 是否在飞行中

    # 状态
    arousal: float = 0.0          # 唤醒度 (0-1)
    nervous: float = 0.0          # 紧张度 (0-1)
    sleep: bool = False           # 是否睡眠

    # 行为状态机
    behavior_mode: str = "rest"   # rest / walk / turn / escape / groom / fly / sleep
    mode_start_time: float = 0.0
    mode_duration: float = 0.0    # 当前模式已持续时间

    # 内部计时器
    last_groom_time: float = 0.0
    last_escape_time: float = 0.0
    in_flight_time: float = 0.0

    # 步态参数
    gait_phase: float = 0.0       # 步态相位 (0-2pi)
    gait_frequency: float = 5.0   # 步态频率 (Hz)
    gait_amplitude: float = 1.0   # 步态幅度


class MotorController:
    """
    运动控制器

    输入：
      - 基础行为指令（walk_drive, turn_bias, escape 等）
      - 蘑菇体输出（approach, avoidance）
      - 中央复合体输出（导航指令）
      - 感官状态（视觉逼近、新奇刺激、昼夜节律）

    输出：
      - 精细化的 MotorState
    """

    def __init__(self):
        self.state = MotorState()
        self._groom_cooldown = 10.0  # 梳理行为冷却时间（秒）
        self._flight_min_duration = 1.5  # 最短飞行时间（秒）
        self._flight_max_duration = 4.0  # 最长飞行时间（秒）

    def update(
        self,
        base_walk: float,
        base_turn: float,
        escape: bool,
        approach: float = 0.0,
        avoidance: float = 0.0,
        looming_strength: float = 0.0,
        novelty: float = 0.0,
        brightness: float = 0.5,
        near_boundary: bool = False,
        cx_turn: float = 0.0,
        cx_speed_mod: float = 1.0,
    ) -> MotorState:
        """
        更新运动控制状态

        Args:
            base_walk: 基础行走驱动力
            base_turn: 基础转向偏置
            escape: 是否逃逸
            approach: 蘑菇体趋向倾向
            avoidance: 蘑菇体回避倾向
            looming_strength: 视觉逼近强度
            novelty: 环境新奇度
            brightness: 环境亮度
            near_boundary: 是否靠近边界
            cx_turn: 中央复合体转向指令
            cx_speed_mod: 中央复合体速度调整

        Returns:
            更新后的 MotorState
        """
        now = time.monotonic()
        s = self.state

        # === 1. 整合基础运动指令 ===
        walk = base_walk * cx_speed_mod
        turn = base_turn * 0.6 + cx_turn * 0.4

        # === 2. 行为状态机 ===
        if escape or looming_strength > 0.6:
            # 逃逸状态：振翅、快速转向、可能起飞
            self._transition_to("escape", now)
            s.wing_drive = max(s.wing_drive, 1.0 + looming_strength * 0.5)
            s.arousal = max(s.arousal, 0.9)
            s.nervous = min(1.0, s.nervous + 0.1)
            walk = max(walk, 0.5)
            turn = turn + math.sin(now * 8.0) * 0.5  # 恐慌性抖动

            # 强逼近触发起飞
            if looming_strength > 0.7 and not s.flying and now - s.last_escape_time > 2.0:
                s.flying = True
                s.in_flight_time = now
                s.last_escape_time = now

        elif s.flying:
            # 飞行中：持续振翅，快速移动
            self._transition_to("fly", now)
            s.wing_drive = 1.2
            walk = max(walk, 0.8)
            s.arousal = max(s.arousal, 0.8)

            # 飞行时间限制
            flight_duration = now - s.in_flight_time
            if flight_duration > self._flight_min_duration:
                # 超过最短时间后，威胁降低则降落
                if looming_strength < 0.3 and flight_duration > self._flight_max_duration * 0.5:
                    s.flying = False
                    s.wing_drive = 0.3
                elif flight_duration > self._flight_max_duration:
                    s.flying = False
                    s.wing_drive = 0.0

        elif novelty > 0.4 and now - s.last_groom_time > self._groom_cooldown:
            # 新奇刺激后触发梳理行为
            self._transition_to("groom", now)
            s.groom_drive = min(1.0, novelty)
            s.wing_drive = 0.2
            walk = walk * 0.3  # 梳理时减速
            s.arousal = max(s.arousal, 0.5)

            # 梳理持续 2-4 秒
            if now - s.mode_start_time > 2.0 + novelty * 2.0:
                s.groom_drive = 0.0
                s.last_groom_time = now

        elif avoidance > 0.5 or near_boundary:
            # 回避状态：减速、转向、可能后退
            self._transition_to("avoid", now)
            walk = min(walk, 0.4)
            s.nervous = min(1.0, avoidance)

            # 强回避或靠近边界时后退
            if (avoidance > 0.7 or near_boundary) and walk < 0.3:
                s.backward = True
                walk = 0.2
            else:
                s.backward = False

        elif walk > 0.3:
            # 行走状态
            self._transition_to("walk", now)
            s.backward = False
            s.arousal = max(s.arousal, 0.3)

            # 步态参数根据速度调整
            s.gait_frequency = 3.0 + walk * 5.0  # 3-8 Hz
            s.gait_amplitude = 0.5 + walk * 0.8

        elif brightness < 0.25:
            # 睡眠/静止状态
            self._transition_to("sleep", now)
            s.sleep = True
            walk = 0.0
            turn = 0.0
            s.wing_drive = 0.0
            s.groom_drive = 0.0
            s.arousal = 0.1
        else:
            # 静息状态
            self._transition_to("rest", now)
            s.sleep = False
            s.backward = False
            s.arousal = max(0.1, s.arousal * 0.95)

            # 静息时偶尔轻微梳理
            if now - s.last_groom_time > 15.0 and random.random() < 0.01:
                s.groom_drive = 0.3
                s.last_groom_time = now
            else:
                s.groom_drive = max(0.0, s.groom_drive - 0.05)

        # === 3. 更新步态相位 ===
        dt = 0.05  # 假设 20fps
        s.gait_phase = (s.gait_phase + s.gait_frequency * dt * 2 * math.pi) % (2 * math.pi)

        # === 4. 衰减翅膀和梳理驱动 ===
        if not escape and not s.flying:
            s.wing_drive = max(0.0, s.wing_drive - 0.1)

        # === 5. 设置最终输出 ===
        s.walk_drive = max(0.0, min(1.5, walk))
        s.turn_bias = max(-1.0, min(1.0, turn))
        s.mode_duration = now - s.mode_start_time

        return s

    def _transition_to(self, new_mode: str, now: float):
        """转换行为模式"""
        if self.state.behavior_mode != new_mode:
            self.state.behavior_mode = new_mode
            self.state.mode_start_time = now

    def get_brain_signals(self) -> Dict:
        """
        获取 DesktopFly BrainSignals 格式的输出

        Returns:
            包含 walkDrive, turnBias, escape, backward, wingDrive, groomDrive, arousal, nervous, sleep 的字典
        """
        s = self.state
        return {
            "walkDrive": round(s.walk_drive, 3),
            "turnBias": round(s.turn_bias, 3),
            "escape": s.behavior_mode == "escape" or s.flying,
            "backward": s.backward,
            "wingDrive": round(s.wing_drive, 3),
            "groomDrive": round(s.groom_drive, 3),
            "arousal": round(s.arousal, 3),
            "nervous": round(s.nervous, 3),
            "sleep": s.sleep,
        }

    def reset(self):
        """重置运动控制器状态"""
        self.state = MotorState()


# 导入 random（放在文件末尾避免循环导入问题）
import random
