#!/usr/bin/env python3
"""
真实感官编码器 - 把桌面环境的真实感官编码成神经电流

与原来的鼠标/音量人工编码不同，这里的感官输入来自果蝇在桌面环境中的
真实交互：
  - 视觉：前方窗口障碍物距离 → 视觉逼近刺激（LC4/LPLC2）
  - 嗅觉：食物源（终端窗口）的气味梯度 → ALPN 激活（触角叶）
  - 触觉：碰撞窗口边缘 → 机械感受 + DAN 惩罚（PPL1）
  - 捕食者：鼠标靠近/快速移动 → 视觉逼近 + 逃跑反应

这些感官输入直接输入蘑菇体 LIF 仿真，和真实果蝇的感官工作方式一致。
"""

import os
import sys
import math
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from encoder import NeuralActivation
from desktop_environment import EnvironmentState
from metabolism import MetabolismState


class RealSensoryEncoder:
    """
    真实感官编码器 - 将桌面环境状态转换为神经激活

    用法:
        encoder = RealSensoryEncoder(circuit)
        activation = encoder.encode(env_state, meta_state)
    """

    # 感官参数
    VISUAL_OBSTACLE_THRESHOLD = 200   # 障碍物距离小于此值触发视觉反应
    VISUAL_OBSTACLE_CRITICAL = 80     # 临界距离（触发强烈反应）
    OLFACTORY_THRESHOLD = 0.1          # 气味强度阈值
    TACTILE_COLLISION_INTENSITY = 0.8  # 碰撞触觉强度
    PREDATOR_VISUAL_STRENGTH = 0.9     # 捕食者视觉强度
    PREDATOR_PUNISHMENT_INTENSITY = 0.5  # 捕食者惩罚强度

    def __init__(self, circuit=None):
        """
        Args:
            circuit: 蘑菇体回路数据（用于获取 ALPN/DAN 神经元 ID）
        """
        self.circuit = circuit
        self.alpn_ids = []
        self.ppl1_ids = []
        self.pam_ids = []

        if circuit:
            self._load_neuron_ids()

    def _load_neuron_ids(self):
        """从回路数据加载神经元 ID"""
        neurons = self.circuit.neurons
        for n in neurons:
            role = n.get('role', '')
            nid = n.get('id', 0)
            if 'ALPN' in role or 'alpn' in role:
                self.alpn_ids.append(nid)
            elif 'PPL1' in role or 'ppl1' in role:
                self.ppl1_ids.append(nid)
            elif 'PAM' in role or 'pam' in role:
                self.pam_ids.append(nid)

        # 如果回路数据没有明确分组，用启发式
        if not self.alpn_ids:
            # 假设前 685 个非 KC 神经元是 ALPN
            self.alpn_ids = [n['id'] for n in neurons[:685]
                           if 'KC' not in n.get('role', '')]

    def encode(self, env: EnvironmentState, meta: MetabolismState) -> NeuralActivation:
        """
        将环境状态和代谢状态编码为神经激活

        Args:
            env: 桌面环境状态
            meta: 代谢状态

        Returns:
            NeuralActivation 神经激活模式
        """
        activation = NeuralActivation()

        # 1. 视觉编码（障碍物 + 捕食者）
        self._encode_vision(env, activation)

        # 2. 嗅觉编码（食物源气味）
        self._encode_olfaction(env, activation)

        # 3. 触觉编码（碰撞）
        self._encode_tactile(env, activation)

        # 4. 捕食者编码（鼠标）
        self._encode_predator(env, activation)

        # 5. 代谢状态调制（全局兴奋性）
        self._encode_metabolism(meta, activation)

        return activation

    def _encode_vision(self, env: EnvironmentState, activation: NeuralActivation):
        """视觉编码：前方障碍物 → 逼近刺激"""
        dist = env.visual_obstacle_distance
        bearing = env.visual_obstacle_bearing

        if dist < self.VISUAL_OBSTACLE_THRESHOLD:
            # 障碍物在视野内，计算逼近强度
            # 距离越近，强度越高
            strength = 1.0 - (dist / self.VISUAL_OBSTACLE_THRESHOLD)
            strength = min(1.0, strength * 1.5)  # 增强

            # 根据方位分配到左右眼
            if bearing < 0:
                # 障碍物在左侧
                activation.loom_left = strength * (1.0 + abs(bearing) / math.pi)
                activation.loom_right = strength * 0.3
            else:
                # 障碍物在右侧
                activation.loom_right = strength * (1.0 + abs(bearing) / math.pi)
                activation.loom_left = strength * 0.3

            activation.looming_strength = strength
            activation.visual_label = f"obstacle_{dist:.0f}px"

            # 极近的障碍物触发气流反应（像撞到东西）
            if dist < self.VISUAL_OBSTACLE_CRITICAL:
                activation.airpuff_strength = 0.5

    def _encode_olfaction(self, env: EnvironmentState, activation: NeuralActivation):
        """嗅觉编码：食物源气味 → ALPN 激活"""
        intensity = env.olfactory_intensity
        odor_type = env.olfactory_type

        if intensity > self.OLFACTORY_THRESHOLD and odor_type == "food":
            # 食物气味激活 ALPN 神经元
            # 选择一部分 ALPN 神经元激活（模拟特定肾小球）
            num_alpn = len(self.alpn_ids)
            if num_alpn > 0:
                # 激活前 30% 的 ALPN 神经元（模拟食物气味的编码模式）
                num_active = max(1, int(num_alpn * 0.3))
                for i in range(num_active):
                    nid = self.alpn_ids[i]
                    # 电流强度随气味强度变化
                    current = 8.0 + intensity * 12.0  # 8-20 nA
                    activation.alpn_activation[nid] = current

            activation.odor_label = "food"
            activation.global_excitability = 1.0 + intensity * 0.2  # 气味增强兴奋性

            # 如果在食物源上，激活 PAM 奖励神经元（多巴胺奖励）
            if env.on_food:
                num_pam = len(self.pam_ids)
                if num_pam > 0:
                    num_active = max(1, int(num_pam * 0.2))
                    for i in range(num_active):
                        nid = self.pam_ids[i]
                        activation.dan_activation[nid] = 5.0  # 奖励电流

    def _encode_tactile(self, env: EnvironmentState, activation: NeuralActivation):
        """触觉编码：碰撞 → 机械感受 + 惩罚"""
        if env.is_colliding:
            # 碰撞激活感觉神经元
            activation.sensory_activation[0] = self.TACTILE_COLLISION_INTENSITY

            # 碰撞触发 PPL1 惩罚神经元（多巴胺惩罚）
            num_ppl1 = len(self.ppl1_ids)
            if num_ppl1 > 0:
                # 激活所有 PPL1 神经元
                for nid in self.ppl1_ids:
                    activation.dan_activation[nid] = 10.0  # 强惩罚电流

            activation.punishment = True
            activation.punishment_intensity = 0.6
            activation.airpuff_strength = max(activation.airpuff_strength, 0.8)
            activation.visual_label = "collision"

    def _encode_predator(self, env: EnvironmentState, activation: NeuralActivation):
        """捕食者编码：鼠标靠近 → 视觉逼近 + 逃跑 + 惩罚"""
        if env.predator_escape:
            # 鼠标非常近，强烈视觉逼近
            activation.loom_left = max(activation.loom_left, self.PREDATOR_VISUAL_STRENGTH)
            activation.loom_right = max(activation.loom_right, self.PREDATOR_VISUAL_STRENGTH)
            activation.looming_strength = max(activation.looming_strength, self.PREDATOR_VISUAL_STRENGTH)

            # 触发惩罚（害怕）
            activation.punishment = True
            activation.punishment_intensity = max(
                activation.punishment_intensity, self.PREDATOR_PUNISHMENT_INTENSITY)

            # 激活 PPL1 惩罚神经元
            num_ppl1 = len(self.ppl1_ids)
            if num_ppl1 > 0:
                for nid in self.ppl1_ids:
                    if nid not in activation.dan_activation:
                        activation.dan_activation[nid] = 8.0

            activation.visual_label = "predator_escape"
            activation.airpuff_strength = max(activation.airpuff_strength, 0.6)

        elif env.predator_alert:
            # 鼠标较近或移动快，中等视觉反应
            alert_strength = 0.4 + (1.0 - min(1.0, env.mouse_distance / 200)) * 0.4
            activation.loom_left = max(activation.loom_left, alert_strength * 0.7)
            activation.loom_right = max(activation.loom_right, alert_strength * 0.7)
            activation.looming_strength = max(activation.looming_strength, alert_strength)
            activation.visual_label = "predator_alert"

        # 记录鼠标信息
        activation.mouse_distance = env.mouse_distance
        activation.mouse_bearing = env.mouse_bearing

    def _encode_metabolism(self, meta: MetabolismState, activation: NeuralActivation):
        """代谢状态调制：全局兴奋性"""
        # 低能量降低兴奋性（饿了没力气）
        if meta.energy < 0.3:
            activation.global_excitability *= 0.7 + meta.energy

        # 高疲劳降低兴奋性
        if meta.fatigue > 0.7:
            activation.global_excitability *= 1.0 - (meta.fatigue - 0.7) * 1.5

        # 夜间降低兴奋性
        if meta.is_night:
            activation.global_excitability *= 0.8

        # 垂死状态大幅降低
        if meta.is_dying:
            activation.global_excitability *= 0.3

        # 需要食物时，增强对食物气味的敏感性
        if meta.need_food:
            activation.global_excitability *= 1.2


if __name__ == "__main__":
    # 测试
    from desktop_environment import DesktopEnvironment
    from metabolism import Metabolism
    from window_detector import WindowDetector

    detector = WindowDetector()
    detector.update(force=True)

    env = DesktopEnvironment()
    env.update(detector)

    meta = Metabolism()
    meta.update(dt=0.033, walk_intensity=0.5)

    encoder = RealSensoryEncoder()
    activation = encoder.encode(env.state, meta.state)

    print("=== 真实感官编码测试 ===")
    print(f"气味: {activation.odor_label}")
    print(f"ALPN 激活: {len(activation.alpn_activation)} 个神经元")
    print(f"DAN 激活: {len(activation.dan_activation)} 个神经元")
    print(f"惩罚: {activation.punishment} (强度: {activation.punishment_intensity:.2f})")
    print(f"视觉: {activation.visual_label}")
    print(f"左眼逼近: {activation.loom_left:.2f}")
    print(f"右眼逼近: {activation.loom_right:.2f}")
    print(f"气流: {activation.airpuff_strength:.2f}")
    print(f"全局兴奋性: {activation.global_excitability:.2f}")
