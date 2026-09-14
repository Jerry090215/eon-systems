#!/usr/bin/env python3
"""
神经编码模块 (Neural Encoder)

将桌面物理变量编码为蘑菇体回路的神经激活模式：
  - 鼠标运动学特征 → ALPN 激活模式（嗅觉编码）
  - 音量突变 → DAN 激活（惩罚/电击信号，驱动 STDP）
  - 键盘频率 → 感觉神经元激活（背景刺激）
  - 屏幕亮度 → 全局兴奋性调制（昼夜节律，可选）

编码原理：
  真实果蝇的不同气味激活触角叶不同的肾小球（glomerulus），
  进而激活不同的投射神经元（PN）集群。这里用鼠标的运动学
  特征（速度、曲率、方向变化、轨迹熵）模拟不同气味分子
  对触角叶的激活模式。
"""

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from sensors import SensorData, VisionState


# ============== 数据结构 ==============

@dataclass
class NeuralActivation:
    """神经激活模式"""
    alpn_activation: Dict[int, float] = field(default_factory=dict)  # ALPN 神经元 ID → 输入电流
    dan_activation: Dict[int, float] = field(default_factory=dict)   # DAN 神经元 ID → 输入电流
    sensory_activation: Dict[int, float] = field(default_factory=dict)  # 感觉神经元 → 输入电流
    global_excitability: float = 1.0  # 全局兴奋性缩放因子
    odor_label: str = "none"  # 当前气味标签（用于调试）
    punishment: bool = False  # 是否处于惩罚状态
    punishment_intensity: float = 0.0  # 惩罚强度
    # 视觉相关
    visual_label: str = "none"  # 视觉刺激标签
    looming_strength: float = 0.0  # 逼近强度 0-1
    airpuff_strength: float = 0.0  # 气流强度
    loom_left: float = 0.0  # 左眼逼近
    loom_right: float = 0.0  # 右眼逼近
    mouse_distance: float = 9999.0  # 鼠标距离
    mouse_bearing: float = 0.0  # 鼠标方位
    punishment_signal: float = 0.0  # 视觉触发的惩罚信号
    ppl1_ids: List[int] = field(default_factory=list)  # PPL1 神经元 ID 列表


@dataclass
class OdorPrototype:
    """气味原型（鼠标运动学特征 → 气味）"""
    name: str
    label: str
    # 特征权重：[速度, 曲率, 方向变化率, 轨迹熵, 加速度]
    feature_weights: np.ndarray
    # 目标特征值（原型的"理想"状态）
    target_features: np.ndarray
    # 激活强度缩放
    scale: float = 1.0


# ============== 回路加载 ==============

class MushroomBodyCircuit:
    """加载蘑菇体回路数据，提供神经元分组"""

    def __init__(self, circuit_path: str):
        with open(circuit_path, 'r') as f:
            self.data = json.load(f)

        self.neurons = self.data['neurons']
        self.edges = self.data['edges']

        # 按角色分组
        self.kc_ids: List[int] = []
        self.alpn_ids: List[int] = []
        self.dan_ids: List[int] = []
        self.mbon_ids: List[int] = []

        for n in self.neurons:
            role = n['role']
            nid = n['id']
            if role == 'KC':
                self.kc_ids.append(nid)
            elif role == 'ALPN':
                self.alpn_ids.append(nid)
            elif role == 'DAN':
                self.dan_ids.append(nid)
            elif role == 'MBON':
                self.mbon_ids.append(nid)

        # DAN 子群（PAM = 奖励，PPL1 = 惩罚）
        self.pam_ids: List[int] = []
        self.ppl1_ids: List[int] = []
        for n in self.neurons:
            if n['role'] == 'DAN':
                ntype = n.get('type', '')
                if ntype.startswith('PAM'):
                    self.pam_ids.append(n['id'])
                elif ntype.startswith('PPL'):
                    self.ppl1_ids.append(n['id'])

        print(f"蘑菇体回路加载完成:")
        print(f"  KC (Kenyon Cell): {len(self.kc_ids)}")
        print(f"  ALPN (投射神经元): {len(self.alpn_ids)}")
        print(f"  DAN (多巴胺神经元): {len(self.dan_ids)}")
        print(f"    PAM (奖励群): {len(self.pam_ids)}")
        print(f"    PPL1 (惩罚群): {len(self.ppl1_ids)}")
        print(f"  MBON (输出神经元): {len(self.mbon_ids)}")
        print(f"  连接: {len(self.edges)}")


# ============== 视觉编码器 ==============

class VisualEncoder:
    """将 DesktopFly 的视觉状态（looming/airPuff）编码为蘑菇体输入

    视觉信号作为第二模态条件刺激，可以和嗅觉一样参与 STDP 联想学习。
    强 looming 同时触发惩罚（PPL1 多巴胺激活），模拟果蝇对逼近物体的先天回避。
    """

    def __init__(self, alpn_ids: List[int], visual_alpn_count: int = 80):
        # 使用最后 N 个 ALPN 作为视觉输入通道（与嗅觉通道分开）
        self.visual_alpn_ids = alpn_ids[-visual_alpn_count:] if len(alpn_ids) > visual_alpn_count else alpn_ids
        self.looming_cluster = self.visual_alpn_ids[:len(self.visual_alpn_ids)//2]   # 逼近刺激
        self.airpuff_cluster = self.visual_alpn_ids[len(self.visual_alpn_ids)//2:]    # 气流刺激
        self.looming_detected = False
        self._looming_start_time = 0.0

    def encode(self, vision: VisionState, activation: NeuralActivation):
        """编码视觉状态"""
        if not vision.available:
            return

        # --- Looming（逼近）作为视觉条件刺激 ---
        loom_strength = min(1.0, vision.loom_total)
        if loom_strength > 0.15:
            # 激活 looming 集群的 ALPN，强度与逼近程度成正比
            base_current = 8.0 + loom_strength * 12.0  # 8-20nA
            for i, nid in enumerate(self.looming_cluster):
                spatial_mod = 0.8 + 0.4 * math.sin(i * 0.4)
                activation.alpn_activation[nid] = base_current * spatial_mod
            activation.visual_label = "looming"
            activation.looming_strength = loom_strength

            # 强 looming 直接触发先天惩罚（PPL1 多巴胺激活）
            if loom_strength > 0.5:
                punish_current = (loom_strength - 0.5) * 30.0  # 0-15nA
                for nid in activation.dan_activation:
                    if nid in getattr(activation, 'ppl1_ids', []):
                        activation.dan_activation[nid] = max(
                            activation.dan_activation.get(nid, 0),
                            punish_current
                        )
                activation.punishment_signal = max(activation.punishment_signal, loom_strength * 0.5)
        else:
            activation.visual_label = "none"
            activation.looming_strength = 0.0

        # --- AirPuff（气流）作为触觉刺激 ---
        if vision.air_puff > 0.1:
            puff_current = vision.air_puff * 10.0  # 0-10nA
            for i, nid in enumerate(self.airpuff_cluster):
                spatial_mod = 0.8 + 0.3 * math.sin(i * 0.5)
                activation.alpn_activation[nid] = max(
                    activation.alpn_activation.get(nid, 0),
                    puff_current * spatial_mod
                )
            activation.airpuff_strength = vision.air_puff

        # 左右眼差异编码为空间信息（用于转向）
        activation.loom_left = vision.loom_left
        activation.loom_right = vision.loom_right
        activation.mouse_distance = vision.mouse_distance
        activation.mouse_bearing = vision.mouse_bearing


# ============== 触角叶（Antennal Lobe）真实模拟 ==============

class AntennalLobe:
    """
    果蝇触角叶真实模拟

    包含：
    - 50 个肾小球（glomerulus），每个对应一种气味受体类型
    - 嗅觉受体神经元（ORN）输入
    - 局部中间神经元（LN）侧抑制：活跃肾小球抑制周围肾小球，增强对比度
    - 投射神经元（PN/ALPN）输出到蘑菇体

    真实果蝇触角叶有约 50 个肾小球，约 1300 个 ORN，约 200 个 LN，约 150 个 PN。
    这里用简化的计算模型实现核心功能：侧抑制增强气味对比度。
    """

    def __init__(self, num_glomeruli: int = 50, inhibition_gain: float = 0.15, alpn_ids: List[int] = None):
        self.num_glomeruli = num_glomeruli
        self.inhibition_gain = inhibition_gain  # 侧抑制强度

        # 肾小球状态
        self.orn_input = np.zeros(num_glomeruli)      # ORN 输入
        self.glomerulus_activation = np.zeros(num_glomeruli)  # 肾小球激活（侧抑制后）
        self.ln_activation = np.zeros(num_glomeruli)   # 局部中间神经元激活

        # 肾小球 → ALPN 映射（每个肾小球对应约 N 个 ALPN）
        self.alpn_ids = alpn_ids or []
        self.glomerulus_to_alpn = {}
        if self.alpn_ids:
            alpn_per_glomerulus = max(1, len(self.alpn_ids) // num_glomeruli)
            for g in range(num_glomeruli):
                start = g * alpn_per_glomerulus
                end = min(start + alpn_per_glomerulus, len(self.alpn_ids))
                self.glomerulus_to_alpn[g] = self.alpn_ids[start:end]

    def set_odor_input(self, glomerulus_pattern: np.ndarray):
        """
        设置气味输入（肾小球激活模式）

        Args:
            glomerulus_pattern: 长度为 num_glomeruli 的数组，每个元素 0-1 表示该肾小球的激活强度
        """
        if len(glomerulus_pattern) != self.num_glomeruli:
            # 自动调整大小
            resized = np.zeros(self.num_glomeruli)
            for i in range(min(len(glomerulus_pattern), self.num_glomeruli)):
                resized[i] = glomerulus_pattern[i]
            self.orn_input = resized
        else:
            self.orn_input = np.array(glomerulus_pattern, dtype=float)

    def step(self, dt_ms: float = 1.0):
        """
        运行触角叶仿真一步：侧抑制处理

        侧抑制模型：
        - 每个肾小球的激活 = ORN输入 - 抑制增益 × 所有其他肾小球的平均激活
        - LN（局部中间神经元）被活跃肾小球激活，然后抑制所有肾小球
        - 这实现了"赢家通吃"效应，增强气味对比度
        """
        # LN 激活：与总输入成正比
        total_input = np.sum(self.orn_input)
        self.ln_activation = self.inhibition_gain * total_input / max(1, self.num_glomeruli)

        # 侧抑制：每个肾小球的激活 = 输入 - 全局抑制
        # 加上一些空间结构（相邻肾小球抑制更强）
        activation = np.zeros(self.num_glomeruli)
        for i in range(self.num_glomeruli):
            # 全局抑制
            inhibition = self.ln_activation * 2.0
            # 相邻肾小球抑制更强（环形距离）
            for j in range(self.num_glomeruli):
                if i != j:
                    ring_dist = min(abs(i - j), self.num_glomeruli - abs(i - j))
                    if ring_dist <= 3:
                        inhibition += self.orn_input[j] * 0.05 * (4 - ring_dist)
            activation[i] = max(0.0, self.orn_input[i] - inhibition)

        # 平滑（时间常数 50ms）
        tau = 50.0
        alpha = min(1.0, dt_ms / tau)
        self.glomerulus_activation = (1 - alpha) * self.glomerulus_activation + alpha * activation

    def get_alpn_output(self) -> Dict[int, float]:
        """
        获取 ALPN 输出电流（从肾小球激活映射到 ALPN）

        Returns:
            ALPN 神经元 ID → 输出电流（nA）
        """
        alpn_currents = {}
        if not self.glomerulus_to_alpn:
            return alpn_currents

        for g_idx, alpn_list in self.glomerulus_to_alpn.items():
            g_activation = self.glomerulus_activation[g_idx]
            if g_activation > 0.01:
                # 肾小球激活 → ALPN 电流（0-20nA）
                base_current = g_activation * 20.0
                for i, alpn_id in enumerate(alpn_list):
                    # 空间调制（同一肾小球内不同 ALPN 激活略有差异）
                    spatial_mod = 0.8 + 0.4 * math.sin(i * 0.5 + g_idx)
                    alpn_currents[alpn_id] = base_current * spatial_mod

        return alpn_currents

    def get_active_glomeruli_count(self) -> int:
        """获取活跃肾小球数量（激活 > 0.1）"""
        return int(np.sum(self.glomerulus_activation > 0.1))

    def reset(self):
        """重置触角叶状态"""
        self.orn_input = np.zeros(self.num_glomeruli)
        self.glomerulus_activation = np.zeros(self.num_glomeruli)
        self.ln_activation = np.zeros(self.num_glomeruli)


# ============== 嗅觉编码器 ==============

class OdorEncoder:
    """将鼠标运动学特征编码为 ALPN 激活模式（嗅觉）"""

    def __init__(self, alpn_ids: List[int]):
        self.alpn_ids = alpn_ids
        self.n_alpn = len(alpn_ids)
        # 触角叶侧抑制参数（模拟局部中间神经元 LN 的抑制作用）
        self.lateral_inhibition_gain = 0.15  # 侧抑制强度
        self.antennal_lobe = AntennalLobe(num_glomeruli=50, inhibition_gain=0.15, alpn_ids=alpn_ids)

        # 定义气味原型
        # 特征向量: [归一化速度, 归一化曲率, 归一化方向变化率, 归一化轨迹熵, 归一化加速度]
        self.prototypes = [
            OdorPrototype(
                name="快速直线（危险气味 A）",
                label="odor_A_fast",
                feature_weights=np.array([1.0, 0.3, 0.2, 0.1, 0.5]),
                target_features=np.array([0.9, 0.1, 0.1, 0.1, 0.7]),
                scale=1.0
            ),
            OdorPrototype(
                name="缓慢画圈（安全气味 B）",
                label="odor_B_slow",
                feature_weights=np.array([0.2, 0.9, 0.8, 0.9, 0.1]),
                target_features=np.array([0.2, 0.8, 0.7, 0.8, 0.1]),
                scale=1.0
            ),
            OdorPrototype(
                name="中速随机（中性气味 C）",
                label="odor_C_random",
                feature_weights=np.array([0.5, 0.5, 0.5, 0.6, 0.4]),
                target_features=np.array([0.5, 0.5, 0.5, 0.6, 0.4]),
                scale=0.8
            ),
        ]

        # 将 ALPN 分成 3 个重叠集群，分别对应 3 种气味
        # 重叠编码更接近真实果蝇的嗅觉表征，同时确保每个气味激活足够多的 ALPN
        half = self.n_alpn // 2
        self.alpn_clusters = {
            'odor_A_fast': alpn_ids[:half],                    # 前半
            'odor_B_slow': alpn_ids[self.n_alpn // 4: self.n_alpn // 4 + half],  # 中间
            'odor_C_random': alpn_ids[self.n_alpn - half:],    # 后半
        }

        # 特征归一化范围
        self.speed_max = 2000.0      # px/s
        self.curvature_max = 0.5      # 1/px
        self.dir_change_max = 10.0    # rad/s
        self.entropy_max = 1.0        # 0-1
        self.accel_max = 5000.0       # px/s²

    def extract_features(self, sensor_data: SensorData) -> np.ndarray:
        """从感官数据提取归一化的运动学特征向量"""
        m = sensor_data.mouse

        # 归一化到 [0, 1]
        speed_norm = min(m.speed / self.speed_max, 1.0)
        curvature_norm = min(m.curvature / self.curvature_max, 1.0)
        dir_change_norm = min(m.direction_change / self.dir_change_max, 1.0)
        entropy_norm = min(m.trajectory_entropy / self.entropy_max, 1.0)
        accel_norm = min(m.acceleration / self.accel_max, 1.0)

        return np.array([speed_norm, curvature_norm, dir_change_norm, entropy_norm, accel_norm])

    def compute_similarity(self, features: np.ndarray, prototype: OdorPrototype) -> float:
        """计算当前特征与气味原型的相似度（加权欧氏距离的倒数）"""
        diff = features - prototype.target_features
        weighted_diff = diff * prototype.feature_weights
        distance = np.sqrt(np.sum(weighted_diff ** 2))
        # 转换为相似度（距离越小，相似度越高）
        similarity = math.exp(-distance * 3.0) * prototype.scale
        return similarity

    def encode(self, sensor_data: SensorData, activation: NeuralActivation):
        """将鼠标运动学特征编码为 ALPN 激活模式（简单规则分类，高灵敏度）"""
        m = sensor_data.mouse

        # 静止超过 0.8 秒或速度极低 → 无气味
        if m.stationary_time > 0.8 or m.speed < 10:
            activation.odor_label = "none"
            for nid in self.alpn_ids:
                activation.alpn_activation[nid] = 0.1
            return

        # 简单规则分类（高灵敏度，确保用户移动就能识别）
        speed = m.speed
        curvature = m.curvature

        if speed > 400 and curvature < 0.15:
            best_odor = "odor_A_fast"      # 快速直线 = 气味A
        elif speed < 350 and curvature > 0.15:
            best_odor = "odor_B_slow"      # 缓慢画圈 = 气味B
        else:
            best_odor = "odor_C_random"    # 其他移动 = 气味C

        activation.odor_label = best_odor

        # 强激活（固定 16nA，足够驱动 KC 放电）
        base_current = 16.0

        # 激活对应集群，其他集群给弱基础激活
        for label, cluster_ids in self.alpn_clusters.items():
            if label == best_odor:
                for i, nid in enumerate(cluster_ids):
                    spatial_mod = 0.85 + 0.3 * math.sin(i * 0.5)
                    activation.alpn_activation[nid] = base_current * spatial_mod
            else:
                for nid in cluster_ids:
                    activation.alpn_activation[nid] = 0.3

        # === 触角叶侧抑制处理（增强气味对比度）===
        # 模拟局部中间神经元（LN）的侧抑制：活跃的 ALPN 抑制周围的 ALPN
        # 这使得气味表征更稀疏、更有特异性，类似真实果蝇的触角叶处理
        alpn_values = list(activation.alpn_activation.values())
        if alpn_values:
            mean_activation = sum(alpn_values) / len(alpn_values)
            max_activation = max(alpn_values)
            if max_activation > 1.0:  # 只有有强气味输入时才应用侧抑制
                for nid in list(activation.alpn_activation.keys()):
                    raw = activation.alpn_activation[nid]
                    # 侧抑制：每个 ALPN 的激活 = max(0, 原始 - 抑制增益 * 平均激活)
                    # 加上空间结构：同一集群内的 ALPN 互相抑制更强
                    inhibited = max(0.1, raw - self.lateral_inhibition_gain * mean_activation * 2.0)
                    activation.alpn_activation[nid] = inhibited


# ============== 惩罚编码器 ==============

class PunishmentEncoder:
    """将音量突变编码为 DAN 激活（惩罚/电击信号）"""

    def __init__(self, pam_ids: List[int], ppl1_ids: List[int]):
        self.pam_ids = pam_ids      # PAM 群 = 奖励信号
        self.ppl1_ids = ppl1_ids    # PPL1 群 = 惩罚信号

    def encode(self, sensor_data: SensorData, activation: NeuralActivation):
        """根据音量状态编码 DAN 激活"""
        v = sensor_data.volume

        # 基础激活（与音量大小成正比）
        base_current = (v.volume / 100.0) * 2.0

        # PAM 群（奖励）：柔和/稳定的音量给弱激活
        for i, nid in enumerate(self.pam_ids):
            reward_mod = 0.5 + 0.5 * math.sin(i * 0.3)
            activation.dan_activation[nid] = base_current * 0.3 * reward_mod

        # PPL1 群（惩罚）：音量突变给强激活
        if v.spike:
            activation.punishment = True
            activation.punishment_intensity = min(v.spike_magnitude / 50.0, 1.0)
            # 强惩罚电流（最大 15nA）
            punish_current = 8.0 + activation.punishment_intensity * 7.0
            for i, nid in enumerate(self.ppl1_ids):
                spatial_mod = 0.9 + 0.2 * math.sin(i * 0.7)
                activation.dan_activation[nid] = punish_current * spatial_mod
        else:
            activation.punishment = False
            activation.punishment_intensity = 0.0
            # 弱基础激活
            for nid in self.ppl1_ids:
                activation.dan_activation[nid] = base_current * 0.2


# ============== 触觉刺激编码器（键盘） ==============

class SensoryEncoder:
    """将键盘事件编码为感觉神经元激活（触觉/机械刺激）"""

    def __init__(self, alpn_ids: List[int]):
        # 使用一部分 ALPN 作为触觉/机械感觉输入
        self.sensory_ids = alpn_ids[-50:] if len(alpn_ids) > 50 else alpn_ids

    def encode(self, sensor_data: SensorData, activation: NeuralActivation):
        """根据键盘频率和打字爆发编码感觉神经元激活"""
        k = sensor_data.keyboard

        # 基础按键频率转换为感觉输入
        base_current = min(k.key_press_rate * 0.5, 3.0)

        # 修饰键给特定的激活模式
        if k.shift:
            base_current += 1.0
        if k.command:
            base_current += 1.5
        if k.control:
            base_current += 0.5
        if k.option:
            base_current += 0.5

        # 打字爆发 = 强触觉刺激（如快速敲击桌面）
        if k.typing_burst:
            burst_current = 5.0 + k.burst_intensity * 5.0  # 5-10nA
            for i, nid in enumerate(self.sensory_ids):
                spatial_mod = 0.7 + 0.6 * math.sin(i * 0.4 + time.monotonic() * 10)
                activation.sensory_activation[nid] = burst_current * spatial_mod
        else:
            for i, nid in enumerate(self.sensory_ids):
                spatial_mod = 0.7 + 0.6 * math.sin(i * 0.4)
                activation.sensory_activation[nid] = base_current * spatial_mod


# ============== 新奇刺激编码器（窗口切换） ==============

class NoveltyEncoder:
    """将窗口切换和应用变化编码为新奇刺激（PAM 多巴胺神经元激活）

    真实果蝇中，新奇环境和新刺激会激活多巴胺能神经元，
    驱动探索行为和学习。这里用窗口切换模拟新奇刺激。
    """

    def __init__(self, pam_ids: List[int]):
        self.pam_ids = pam_ids  # PAM 群 = 奖励/新奇信号
        self._novelty_decay = 0.0  # 新奇度衰减

    def encode(self, sensor_data: SensorData, activation: NeuralActivation):
        """根据窗口状态编码新奇刺激"""
        w = sensor_data.window

        # 应用切换 = 强新奇刺激
        if w.app_switched:
            novelty_current = 8.0 + w.novelty * 4.0  # 8-12nA
            self._novelty_decay = 1.0  # 设置衰减起始
        # 窗口数量变化 = 中等新奇刺激
        elif w.window_count_changed:
            novelty_current = 3.0 + w.novelty * 3.0  # 3-6nA
            self._novelty_decay = max(self._novelty_decay, 0.5)
        else:
            # 新奇度随时间衰减
            self._novelty_decay *= 0.95
            novelty_current = self._novelty_decay * 2.0  # 衰减中的弱新奇信号

        # 激活 PAM 多巴胺神经元（新奇/奖励信号）
        if novelty_current > 0.1:
            for i, nid in enumerate(self.pam_ids):
                # 空间调制：不同 PAM 子群激活强度不同
                spatial_mod = 0.6 + 0.8 * math.sin(i * 0.2 + time.monotonic() * 5)
                activation.dan_activation[nid] = max(
                    activation.dan_activation.get(nid, 0.0),
                    novelty_current * spatial_mod
                )


# ============== 全局兴奋性调制 ==============

class GlobalModulation:
    """全局兴奋性调制（昼夜节律，基于屏幕亮度）"""

    def __init__(self):
        self.default_excitability = 1.0

    def encode(self, sensor_data: SensorData, activation: NeuralActivation):
        """根据屏幕亮度调制全局兴奋性"""
        b = sensor_data.brightness
        if b.available:
            # 亮度 0-1 映射到兴奋性 0.6-1.2
            activation.global_excitability = 0.6 + b.brightness * 0.6
        else:
            # 亮度不可用时，使用默认值
            activation.global_excitability = self.default_excitability


# ============== 神经编码聚合器 ==============

class NeuralEncoder:
    """神经编码聚合器，将所有感官通道编码为统一的神经激活模式"""

    def __init__(self, circuit: MushroomBodyCircuit):
        self.circuit = circuit
        self.odor_encoder = OdorEncoder(circuit.alpn_ids)
        self.visual_encoder = VisualEncoder(circuit.alpn_ids)
        self.punishment_encoder = PunishmentEncoder(circuit.pam_ids, circuit.ppl1_ids)
        self.sensory_encoder = SensoryEncoder(circuit.alpn_ids)
        self.novelty_encoder = NoveltyEncoder(circuit.pam_ids)
        self.global_modulation = GlobalModulation()
        self.activation = NeuralActivation()
        self.activation.ppl1_ids = list(circuit.ppl1_ids)

    def encode(self, sensor_data: SensorData) -> NeuralActivation:
        """将感官数据编码为神经激活模式"""
        # 重置激活
        self.activation = NeuralActivation()

        # 各通道编码
        self.odor_encoder.encode(sensor_data, self.activation)
        self.visual_encoder.encode(sensor_data.vision, self.activation)
        self.punishment_encoder.encode(sensor_data, self.activation)
        self.sensory_encoder.encode(sensor_data, self.activation)
        self.novelty_encoder.encode(sensor_data, self.activation)
        self.global_modulation.encode(sensor_data, self.activation)

        return self.activation

    def get_input_current_vector(self) -> Dict[int, float]:
        """获取所有神经元的输入电流向量（用于 LIF 仿真）"""
        currents = {}

        # ALPN 激活（嗅觉 + 感觉）
        for nid, current in self.activation.alpn_activation.items():
            currents[nid] = currents.get(nid, 0.0) + current
        for nid, current in self.activation.sensory_activation.items():
            currents[nid] = currents.get(nid, 0.0) + current

        # DAN 激活（惩罚/奖励）
        for nid, current in self.activation.dan_activation.items():
            currents[nid] = currents.get(nid, 0.0) + current

        # 全局兴奋性缩放
        for nid in currents:
            currents[nid] *= self.activation.global_excitability

        return currents

    def print_status(self):
        """打印编码状态（用于调试）"""
        a = self.activation
        sensory_total = sum(a.sensory_activation.values())
        novelty_active = self.novelty_encoder._novelty_decay > 0.1
        print(f"\r[嗅觉] {a.odor_label:15s} "
              f"ALPN={sum(a.alpn_activation.values()):6.1f}nA | "
              f"[惩罚] {'⚡' if a.punishment else '○'} "
              f"强度={a.punishment_intensity:.2f} "
              f"DAN={sum(a.dan_activation.values()):6.1f}nA | "
              f"[触觉] {sensory_total:5.1f}nA | "
              f"[新奇] {'✨' if novelty_active else '○'} | "
              f"[昼夜] 兴奋={a.global_excitability:.2f}",
              end='', flush=True)


# ============== 命令行测试 ==============

if __name__ == "__main__":
    import sys
    import time

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from sensors import SensorHub

    print("=" * 80)
    print("神经编码模块测试")
    print("=" * 80)

    # 加载回路
    circuit_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "mushroom_body", "circuit.json"
    )
    circuit = MushroomBodyCircuit(circuit_path)

    # 创建编码器
    encoder = NeuralEncoder(circuit)
    hub = SensorHub()

    print("\n开始编码测试（运行 5 秒）...")
    print("移动鼠标改变嗅觉，调节音量触发惩罚\n")

    start = time.monotonic()
    last_print = 0

    try:
        while time.monotonic() - start < 5.0:
            sensor_data = hub.update()
            activation = encoder.encode(sensor_data)
            currents = encoder.get_input_current_vector()

            now = time.monotonic()
            if now - last_print > 0.3:
                encoder.print_status()
                print(f" | 激活神经元={len(currents)}", end='', flush=True)
                print()
                last_print = now

            time.sleep(1.0 / 30.0)
    except KeyboardInterrupt:
        pass

    print("\n\n测试完成！")
