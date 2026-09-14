#!/usr/bin/env python3
"""
视叶 LIF 仿真网络 (Visual Lobe LIF Network)

基于 FlyWire v783 真实连接组的视叶核心通路：
- 视网膜输入 (Retina Input) — 模拟光感受器阵列
- 髓质中间神经元 (Tm1-Tm21) — On/Off 通道处理
- T4/T5 运动检测神经元 — 4个方向的运动检测
- LC 输出神经元 — 视觉特征输出（looming、小物体、大范围运动等）

输出到：
- 蘑菇体 (ALPN 输入) — 视觉联想学习
- 中央复合体 — 视觉导航
- 逃逸反射 — 直接触发逃跑行为
"""

import json
import math
import os
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VisualInput:
    """视觉输入（模拟视网膜阵列）"""
    # 视网膜阵列（简化为 8x8 的光强阵列，值 0-1）
    retina: np.ndarray = None  # shape: (8, 8)
    # 上一帧的视网膜（用于运动检测）
    retina_prev: np.ndarray = None
    # 全局光强变化（用于昼夜节律、闪光刺激）
    light_intensity: float = 0.5
    # 逼近刺激（looming）— 模拟物体靠近
    looming: float = 0.0  # 0-1，逼近强度
    looming_direction: float = 0.0  # 逼近方向（弧度）
    # 小物体运动
    small_object: bool = False
    object_direction: float = 0.0
    object_speed: float = 0.0


@dataclass
class VisualOutput:
    """视叶输出（视觉特征）"""
    # 运动检测（4个方向，T4=On通道，T5=Off通道）
    motion_on: np.ndarray = None  # shape: (4,) — 前→后, 后→前, 上→下, 下→上
    motion_off: np.ndarray = None  # shape: (4,)
    # 总运动强度
    motion_total: float = 0.0
    motion_direction: float = 0.0  # 主导运动方向（弧度）
    # 逼近刺激（looming）
    looming: float = 0.0
    looming_direction: float = 0.0
    # 小物体检测
    small_object: float = 0.0
    # 大范围运动（光流）
    large_field_motion: float = 0.0
    # 逃逸触发（LC4 神经元激活）
    escape_trigger: float = 0.0
    # 输出神经元放电率（用于监控）
    lc_rates: dict = field(default_factory=dict)
    t4_rates: np.ndarray = None
    t5_rates: np.ndarray = None


class VisualLobeLIF:
    """
    视叶 LIF 神经元网络

    基于 FlyWire 真实连接组，包含：
    - 髓质中间神经元 (Tm)
    - T4/T5 运动检测神经元
    - LC 输出神经元
    """

    def __init__(self, circuit_path: Optional[str] = None):
        """
        初始化视叶网络

        Args:
            circuit_path: 视叶电路 JSON 文件路径
        """
        if circuit_path is None:
            circuit_path = "/Users/fujierui/Desktop/Eon Systems/visual_lobe/circuit.json"

        print(f"[VisualLobe] 加载视叶电路: {circuit_path}")

        with open(circuit_path, 'r') as f:
            self.circuit = json.load(f)

        self.n_neurons = self.circuit["neuron_count"]
        print(f"[VisualLobe] 神经元数: {self.n_neurons}")
        print(f"[VisualLobe] 连接数: {self.circuit['connection_count']}")
        print(f"[VisualLobe] 突触数: {self.circuit['synapse_count']}")

        # 神经元信息
        self.neurons = self.circuit["neurons"]
        self.cell_types = [n["cell_type"] for n in self.neurons]
        self.functional_classes = [n["functional_class"] for n in self.neurons]

        # 建立索引映射
        self.id_to_idx = {n["root_id"]: i for i, n in enumerate(self.neurons)}

        # 按功能类别分组
        self.group_indices = {}
        for i, fc in enumerate(self.functional_classes):
            if fc not in self.group_indices:
                self.group_indices[fc] = []
            self.group_indices[fc].append(i)

        print(f"[VisualLobe] 功能类别: {list(self.group_indices.keys())}")
        for fc, indices in self.group_indices.items():
            print(f"  {fc:20s}: {len(indices):4d} 个")

        # 连接矩阵（稀疏）
        conn = self.circuit["connections"]
        self.pre_indices = np.array(conn["pre_indices"], dtype=np.int32)
        self.post_indices = np.array(conn["post_indices"], dtype=np.int32)
        self.syn_weights = np.array(conn["weights"], dtype=np.float32)

        # 神经递质（决定兴奋性/抑制性）
        self.neurotransmitters = [n.get("main_neurotransmitter", "unknown") for n in self.neurons]

        # 连接权重缩放（基于突触数）
        self.weight_scale = 0.002  # 基础缩放
        self.excitatory_scale = 1.0  # 兴奋性（ach/glut）
        self.inhibitory_scale = -1.5  # 抑制性（gaba）

        # LIF 参数
        self.tau_m = 15.0  # 膜时间常数（ms），视叶神经元更快
        self.threshold = 1.0  # 放电阈值
        self.refractory = 1.5  # 不应期（ms）
        self.rest_potential = 0.0
        self.input_scale = 0.15  # 输入电流缩放

        # 状态变量
        self.V = np.zeros(self.n_neurons, dtype=np.float32)  # 膜电位
        self.refractory_time = np.zeros(self.n_neurons, dtype=np.float32)  # 不应期剩余
        self.spikes = np.zeros(self.n_neurons, dtype=np.float32)  # 当前帧放电
        self.spike_rates = np.zeros(self.n_neurons, dtype=np.float32)  # 放电率（滑动平均）
        self.rate_tau = 100.0  # 放电率滑动平均时间常数（ms）

        # 输入电流
        self.input_current = np.zeros(self.n_neurons, dtype=np.float32)

        # 视网膜状态（用于运动检测的时间差分）
        self._retina_prev = None
        self._retina_history = []  # 保存最近几帧视网膜，用于延迟线运动检测

        print(f"[VisualLobe] 初始化完成")

    def set_retina_input(self, visual_input: VisualInput):
        """
        设置视网膜输入，将视觉刺激编码为神经元输入电流

        Args:
            visual_input: 视觉输入
        """
        self.input_current[:] = 0.0

        if visual_input.retina is not None:
            retina = visual_input.retina
        else:
            # 默认视网膜（均匀光强）
            retina = np.ones((8, 8)) * visual_input.light_intensity

        # 保存视网膜历史（用于运动检测）
        self._retina_history.append(retina.copy())
        if len(self._retina_history) > 5:
            self._retina_history.pop(0)

        # === 髓质中间神经元输入（Tm 神经元）===
        # Tm 神经元接收视网膜输入，按空间位置映射
        # 简化：每个 Tm 神经元对应视网膜的一个区域
        tm_indices = self.group_indices.get("medulla_on", []) + self.group_indices.get("medulla_off", [])

        for i, idx in enumerate(tm_indices):
            # 空间位置映射（均匀分布在 8x8 视网膜上）
            row = (i // 8) % 8
            col = i % 8
            light = retina[row, col] if 0 <= row < 8 and 0 <= col < 8 else 0.5

            # On 通道神经元对光强增加敏感
            if self.functional_classes[idx] == "medulla_on":
                if self._retina_prev is not None:
                    delta = light - self._retina_prev[row, col]
                    self.input_current[idx] += max(0, delta) * 8.0 + light * 2.0
                else:
                    self.input_current[idx] += light * 3.0
            # Off 通道神经元对光强减少敏感
            else:  # medulla_off
                if self._retina_prev is not None:
                    delta = self._retina_prev[row, col] - light
                    self.input_current[idx] += max(0, delta) * 8.0 + (1.0 - light) * 2.0
                else:
                    self.input_current[idx] += (1.0 - light) * 3.0

        # === T4/T5 运动检测神经元输入 ===
        # T4 = On 通道运动检测，T5 = Off 通道运动检测
        # 使用延迟线相关（Hassenstein-Reichardt 检测器）
        if len(self._retina_history) >= 2:
            retina_curr = self._retina_history[-1]
            retina_prev = self._retina_history[-2]

            # 计算 4 个方向的运动能量
            # 方向: 0=前→后(右), 1=后→前(左), 2=上→下, 3=下→上
            motion_energy_on = np.zeros(4)
            motion_energy_off = np.zeros(4)

            for row in range(8):
                for col in range(8):
                    curr_on = max(0, retina_curr[row, col] - retina_prev[row, col])
                    curr_off = max(0, retina_prev[row, col] - retina_curr[row, col])

                    # 前→后（向右运动）
                    if col > 0:
                        motion_energy_on[0] += curr_on * retina_prev[row, col - 1]
                        motion_energy_off[0] += curr_off * (1.0 - retina_prev[row, col - 1])
                    # 后→前（向左运动）
                    if col < 7:
                        motion_energy_on[1] += curr_on * retina_prev[row, col + 1]
                        motion_energy_off[1] += curr_off * (1.0 - retina_prev[row, col + 1])
                    # 上→下
                    if row > 0:
                        motion_energy_on[2] += curr_on * retina_prev[row - 1, col]
                        motion_energy_off[2] += curr_off * (1.0 - retina_prev[row - 1, col])
                    # 下→上
                    if row < 7:
                        motion_energy_on[3] += curr_on * retina_prev[row + 1, col]
                        motion_energy_off[3] += curr_off * (1.0 - retina_prev[row + 1, col])

            # 归一化
            max_on = np.max(motion_energy_on)
            max_off = np.max(motion_energy_off)
            if max_on > 0:
                motion_energy_on /= max_on
            if max_off > 0:
                motion_energy_off /= max_off

            # 注入到 T4/T5 神经元
            t4_indices = self.group_indices.get("motion_on", [])
            t5_indices = self.group_indices.get("motion_off", [])

            # 每个方向的神经元数量
            t4_per_dir = len(t4_indices) // 4 if t4_indices else 0
            t5_per_dir = len(t5_indices) // 4 if t5_indices else 0

            for dir_idx in range(4):
                # T4 (On 通道)
                start = dir_idx * t4_per_dir
                end = start + t4_per_dir
                for idx in t4_indices[start:end]:
                    self.input_current[idx] += motion_energy_on[dir_idx] * 10.0

                # T5 (Off 通道)
                start = dir_idx * t5_per_dir
                end = start + t5_per_dir
                for idx in t5_indices[start:end]:
                    self.input_current[idx] += motion_energy_off[dir_idx] * 10.0

        # === LC 输出神经元直接输入（逼近、小物体等）===
        # LC11: looming 检测
        lc11_indices = [i for i, ct in enumerate(self.cell_types) if ct == "LC11"]
        for idx in lc11_indices:
            self.input_current[idx] += visual_input.looming * 12.0

        # LC12: 小物体运动
        lc12_indices = [i for i, ct in enumerate(self.cell_types) if ct == "LC12"]
        for idx in lc12_indices:
            if visual_input.small_object:
                self.input_current[idx] += 8.0

        # LC17: 大范围运动（光流）
        lc17_indices = [i for i, ct in enumerate(self.cell_types) if ct == "LC17"]
        for idx in lc17_indices:
            # 大范围运动 = 所有方向运动能量的平均值
            if len(self._retina_history) >= 2:
                avg_motion = np.mean([
                    np.mean(np.abs(self._retina_history[-1] - self._retina_history[-2]))
                ])
                self.input_current[idx] += avg_motion * 15.0

        # LC4: 逃逸触发（强逼近或快速运动）
        lc4_indices = [i for i, ct in enumerate(self.cell_types) if ct == "LC4"]
        for idx in lc4_indices:
            escape_signal = visual_input.looming * 0.7
            if len(self._retina_history) >= 2:
                rapid_change = np.mean(np.abs(self._retina_history[-1] - self._retina_history[-2]))
                escape_signal += rapid_change * 3.0
            self.input_current[idx] += escape_signal * 10.0

        # LC6: 运动检测
        lc6_indices = [i for i, ct in enumerate(self.cell_types) if ct == "LC6"]
        for idx in lc6_indices:
            if len(self._retina_history) >= 2:
                total_motion = np.mean(np.abs(self._retina_history[-1] - self._retina_history[-2]))
                self.input_current[idx] += total_motion * 12.0

        # 保存当前视网膜为上一帧
        self._retina_prev = retina.copy()

    def step(self, dt_ms: float = 1.0):
        """
        执行一步 LIF 仿真

        Args:
            dt_ms: 时间步长（毫秒）

        Returns:
            spikes: 当前帧放电的神经元数组
        """
        # 衰减膜电位
        decay = math.exp(-dt_ms / self.tau_m)
        self.V *= decay

        # 注入输入电流
        self.V += self.input_current * self.input_scale * dt_ms

        # 突触传递（上一帧放电的神经元影响突触后神经元）
        if np.any(self.spikes > 0):
            # 找到放电的神经元
            spiking_indices = np.where(self.spikes > 0)[0]

            for pre_idx in spiking_indices:
                # 找到这个神经元的所有输出连接
                mask = self.pre_indices == pre_idx
                if np.any(mask):
                    post_idxs = self.post_indices[mask]
                    weights = self.syn_weights[mask]

                    # 神经递质决定兴奋性/抑制性
                    nt = self.neurotransmitters[pre_idx]
                    if nt in ["ach_avg", "glut_avg"]:
                        sign = self.excitatory_scale
                    elif nt == "gaba_avg":
                        sign = self.inhibitory_scale
                    else:
                        sign = 0.5  # 未知，默认弱兴奋

                    # 注入突触后电流
                    for j, post_idx in enumerate(post_idxs):
                        if self.refractory_time[post_idx] <= 0:
                            self.V[post_idx] += weights[j] * self.weight_scale * sign * dt_ms

        # 不应期衰减
        self.refractory_time = np.maximum(0, self.refractory_time - dt_ms)

        # 检测放电
        self.spikes[:] = 0.0
        fire_mask = (self.V >= self.threshold) & (self.refractory_time <= 0)
        self.spikes[fire_mask] = 1.0

        # 重置放电神经元
        self.V[fire_mask] = self.rest_potential
        self.refractory_time[fire_mask] = self.refractory

        # 更新放电率（滑动平均）
        rate_decay = math.exp(-dt_ms / self.rate_tau)
        self.spike_rates *= rate_decay
        self.spike_rates += self.spikes * (1000.0 / dt_ms)  # 转换为 Hz

        # 清空输入电流（下一帧重新设置）
        self.input_current[:] = 0.0

        return self.spikes

    def get_output(self) -> VisualOutput:
        """
        获取视叶输出（视觉特征）

        Returns:
            VisualOutput: 视觉特征输出
        """
        output = VisualOutput()

        # T4/T5 运动检测输出
        t4_indices = self.group_indices.get("motion_on", [])
        t5_indices = self.group_indices.get("motion_off", [])

        t4_per_dir = len(t4_indices) // 4 if t4_indices else 0
        t5_per_dir = len(t5_indices) // 4 if t5_indices else 0

        output.motion_on = np.zeros(4)
        output.motion_off = np.zeros(4)
        output.t4_rates = np.zeros(4)
        output.t5_rates = np.zeros(4)

        for dir_idx in range(4):
            if t4_per_dir > 0:
                start = dir_idx * t4_per_dir
                end = start + t4_per_dir
                output.motion_on[dir_idx] = np.mean(self.spike_rates[t4_indices[start:end]])
                output.t4_rates[dir_idx] = output.motion_on[dir_idx]

            if t5_per_dir > 0:
                start = dir_idx * t5_per_dir
                end = start + t5_per_dir
                output.motion_off[dir_idx] = np.mean(self.spike_rates[t5_indices[start:end]])
                output.t5_rates[dir_idx] = output.motion_off[dir_idx]

        # 总运动强度和方向
        all_motion = output.motion_on + output.motion_off
        output.motion_total = float(np.sum(all_motion))
        if output.motion_total > 0:
            # 方向映射: 0=右, 1=左, 2=下, 3=上
            dir_angles = [0, math.pi, math.pi / 2, -math.pi / 2]
            weighted_angle = sum(all_motion[i] * dir_angles[i] for i in range(4))
            output.motion_direction = weighted_angle / output.motion_total

        # LC 输出神经元
        lc_types = ["LC11", "LC12", "LC17", "LC4", "LC6"]
        for lc_type in lc_types:
            indices = [i for i, ct in enumerate(self.cell_types) if ct == lc_type]
            if indices:
                rate = float(np.mean(self.spike_rates[indices]))
                output.lc_rates[lc_type] = rate

                if lc_type == "LC11":
                    output.looming = min(1.0, rate / 50.0)
                elif lc_type == "LC12":
                    output.small_object = min(1.0, rate / 50.0)
                elif lc_type == "LC17":
                    output.large_field_motion = min(1.0, rate / 50.0)
                elif lc_type == "LC4":
                    output.escape_trigger = min(1.0, rate / 50.0)

        return output

    def get_alpn_input(self) -> np.ndarray:
        """
        获取输入到蘑菇体 ALPN 的视觉信号

        将视叶输出压缩为 ALPN 输入维度（685 个 ALPN 中的一部分）

        Returns:
            ALPN 输入电流数组
        """
        # 简化：将视觉特征映射为 ALPN 输入
        # 实际应该是视叶输出神经元 → ALPN 的特定连接
        output = self.get_output()

        # 创建一个 685 维的 ALPN 输入（实际使用时由主系统映射）
        # 这里返回一个压缩的视觉特征向量
        visual_features = np.zeros(64, dtype=np.float32)

        # 运动方向（4维）
        visual_features[0:4] = output.motion_on / 100.0
        visual_features[4:8] = output.motion_off / 100.0

        # 视觉特征（looming、小物体、大范围运动、逃逸）
        visual_features[8] = output.looming
        visual_features[9] = output.small_object
        visual_features[10] = output.large_field_motion
        visual_features[11] = output.escape_trigger

        # LC 神经元放电率
        for i, (lc_type, rate) in enumerate(output.lc_rates.items()):
            if 12 + i < 64:
                visual_features[12 + i] = rate / 100.0

        return visual_features

    def reset(self):
        """重置网络状态"""
        self.V[:] = 0.0
        self.refractory_time[:] = 0.0
        self.spikes[:] = 0.0
        self.spike_rates[:] = 0.0
        self.input_current[:] = 0.0
        self._retina_prev = None
        self._retina_history = []


def test_visual_lobe():
    """测试视叶网络"""
    print("=" * 60)
    print("视叶 LIF 网络测试")
    print("=" * 60)

    vl = VisualLobeLIF()

    # 测试 1: 静态光强
    print("\n--- 测试 1: 静态光强 ---")
    vis_input = VisualInput(light_intensity=0.8)
    vl.set_retina_input(vis_input)
    for i in range(50):
        vl.step(dt_ms=1.0)
    output = vl.get_output()
    print(f"  运动强度: {output.motion_total:.2f} (应为低)")
    print(f"  looming: {output.looming:.2f}")
    print(f"  T4 放电率: {output.t4_rates}")
    print(f"  LC 放电率: {output.lc_rates}")

    # 测试 2: 运动刺激（向右运动）
    print("\n--- 测试 2: 向右运动刺激 ---")
    vl.reset()
    for frame in range(30):
        # 创建一个向右移动的亮斑
        retina = np.zeros((8, 8))
        col = min(7, frame // 3)
        retina[3:5, col:col+2] = 1.0
        vis_input = VisualInput(retina=retina)
        vl.set_retina_input(vis_input)
        vl.step(dt_ms=1.0)

    output = vl.get_output()
    print(f"  运动强度: {output.motion_total:.2f} (应为高)")
    print(f"  运动方向: {output.motion_direction:.2f} rad (0=右)")
    print(f"  T4 On 通道: {output.t4_rates} (方向0=右应最高)")
    print(f"  T5 Off 通道: {output.t5_rates}")

    # 测试 3: looming 刺激
    print("\n--- 测试 3: Looming 逼近刺激 ---")
    vl.reset()
    for frame in range(20):
        looming = frame / 20.0
        vis_input = VisualInput(looming=looming, looming_direction=0.0)
        vl.set_retina_input(vis_input)
        vl.step(dt_ms=1.0)

    output = vl.get_output()
    print(f"  looming: {output.looming:.2f} (应为高)")
    print(f"  逃逸触发: {output.escape_trigger:.2f} (应为高)")
    print(f"  LC11 放电率: {output.lc_rates.get('LC11', 0):.1f} Hz")
    print(f"  LC4 放电率: {output.lc_rates.get('LC4', 0):.1f} Hz")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    test_visual_lobe()
