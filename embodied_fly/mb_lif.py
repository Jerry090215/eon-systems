#!/usr/bin/env python3
"""
蘑菇体 LIF 脉冲神经网络仿真 (Mushroom Body LIF Network)

基于 FlyWire FAFB v783 完整连接组提取的蘑菇体回路：
  - KC (Kenyon Cell): 5,177 个，蘑菇体主要神经元
  - ALPN (投射神经元): 685 个，嗅觉输入
  - DAN (多巴胺神经元): 339 个，PAM(奖励)/PPL1(惩罚)调制
  - MBON (输出神经元): 96 个，行为输出

关键通路:
  ALPN → KC: 嗅觉输入
  KC → MBON: 学习输出（可塑，STDP）
  DAN → KC/MBON: 多巴胺调制（奖励/惩罚）

学习机制:
  当 KC 放电 + DAN 激活时，KC→MBON 突触权重改变
  - PPL1 (惩罚): 突触抑制 → MBON 放电减少 → 回避行为
  - PAM (奖励): 突触增强 → MBON 放电增加 → 趋向行为

用法:
  from mb_lif import MushroomBodyLIF
  sim = MushroomBodyLIF("../mushroom_body/circuit.json")
  sim.set_odor_input(alpn_currents)   # 嗅觉输入
  sim.set_punishment(ppl1_current)     # 惩罚输入
  sim.step(1)                           # 运行 1ms
  behavior = sim.get_behavior()         # 读取行为输出
"""

import json
import os
import time
import numpy as np
from scipy import sparse
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class MBState:
    """蘑菇体仿真状态快照"""
    time_ms: int = 0
    kc_spikes: int = 0
    alpn_spikes: int = 0
    dan_spikes: int = 0
    mbon_spikes: int = 0
    total_spikes: int = 0
    mbon_rates: Dict[str, float] = field(default_factory=dict)
    punishment_active: bool = False
    reward_active: bool = False
    odor_label: str = "none"
    stdp_updates: int = 0


class MushroomBodyLIF:
    """
    蘑菇体 LIF 脉冲神经网络仿真

    神经元模型: Leaky Integrate-and-Fire (LIF)
      dV/dt = -(V - V_rest) / tau_m + I / C
      当 V >= V_threshold 时放电，V 重置为 V_reset，进入不应期

    时间步长: 1ms
    膜时间常数: 20ms
    阈值: 1.0
    不应期: 2ms
    """

    # LIF 参数
    TAU_M = 20.0        # 膜时间常数 (ms)
    THRESHOLD = 1.0      # 放电阈值
    V_RESET = 0.0        # 重置电位
    V_REST = 0.0         # 静息电位
    REFRACTORY_MS = 2    # 不应期 (ms)
    WEIGHT_SCALE = 0.0008  # 突触权重缩放（与 DesktopFly 一致）

    # STDP 参数
    STDP_TAU_PRE = 50.0   # 突触前时间窗口 (ms)，延长以覆盖气味呈现期
    STDP_TAU_POST = 20.0  # 突触后时间窗口 (ms)
    STDP_LR_PUNISH = 0.0008  # 惩罚学习率（绝对权重变化/次）
    STDP_LR_REWARD = 0.0005  # 奖励学习率
    STDP_WEIGHT_MIN = 0.0   # 权重下限
    STDP_WEIGHT_MAX = 5.0   # 权重上限

    def __init__(self, circuit_path: str, enable_stdp: bool = True):
        """
        加载蘑菇体回路并初始化仿真

        Args:
            circuit_path: circuit.json 文件路径
            enable_stdp: 是否启用 STDP 突触可塑性
        """
        self.enable_stdp = enable_stdp
        self._load_circuit(circuit_path)
        self._build_network()
        self._init_state()
        print(f"[蘑菇体 LIF] 初始化完成: {self.n} 神经元, {self.n_edges} 连接")
        print(f"  KC={len(self.kc_idx)}, ALPN={len(self.alpn_idx)}, "
              f"DAN={len(self.dan_idx)}, MBON={len(self.mbon_idx)}")

    def _load_circuit(self, path: str):
        """加载回路数据"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.neurons = data["neurons"]
        self.edges = data["edges"]
        self.n = len(self.neurons)
        self.n_edges = len(self.edges)

        # 建立 ID → 索引映射
        self.id_to_idx = {}
        self.idx_to_id = []
        for i, n in enumerate(self.neurons):
            nid = n["id"]
            self.id_to_idx[nid] = i
            self.idx_to_id.append(nid)

        # 按角色分组
        self.kc_idx = []
        self.alpn_idx = []
        self.dan_idx = []
        self.mbon_idx = []
        self.other_idx = []

        self.kc_types = defaultdict(list)
        self.dan_subgroups = defaultdict(list)  # PAM/PPL1/PPM
        self.mbon_types = defaultdict(list)

        for i, n in enumerate(self.neurons):
            role = n.get("role", "other")
            ntype = n.get("type", "unknown")
            if role == "KC":
                self.kc_idx.append(i)
                self.kc_types[ntype].append(i)
            elif role == "ALPN":
                self.alpn_idx.append(i)
            elif role == "DAN":
                self.dan_idx.append(i)
                if ntype.startswith("PAM"):
                    self.dan_subgroups["PAM"].append(i)
                elif ntype.startswith("PPL1"):
                    self.dan_subgroups["PPL1"].append(i)
                elif ntype.startswith("PPM"):
                    self.dan_subgroups["PPM"].append(i)
                else:
                    self.dan_subgroups["other"].append(i)
            elif role == "MBON":
                self.mbon_idx.append(i)
                # 提取 MBON 类型编号
                import re
                m = re.match(r"MBON(\d+)", ntype)
                if m:
                    self.mbon_types[f"MBON{m.group(1)}"].append(i)
                else:
                    self.mbon_types[ntype].append(i)
            else:
                self.other_idx.append(i)

        self.pam_idx = self.dan_subgroups.get("PAM", [])
        self.ppl1_idx = self.dan_subgroups.get("PPL1", [])

        print(f"[蘑菇体 LIF] DAN 子群: PAM={len(self.pam_idx)}, PPL1={len(self.ppl1_idx)}, "
              f"其他={len(self.dan_idx) - len(self.pam_idx) - len(self.ppl1_idx)}")

    def _build_network(self):
        """构建稀疏连接矩阵"""
        # 神经递质 → 符号映射
        nt_sign = {
            "acetylcholine": 1.0,   # 兴奋性
            "glutamate": 1.0,        # 兴奋性（果蝇中主要是兴奋性）
            "GABA": -1.0,            # 抑制性
            "gaba": -1.0,
            "dopamine": 0.3,         # 调制性（弱兴奋）
            "octopamine": 0.2,       # 调制性
            "serotonin": 0.1,        # 调制性
        }

        rows = []
        cols = []
        weights = []
        self.kc_to_mbon_edges = []  # 记录 KC→MBON 连接的索引，用于 STDP

        for e in self.edges:
            pre_id = e["pre"]
            post_id = e["post"]
            syn_count = e.get("syn", 1)
            nt = e.get("nt", "acetylcholine")
            base_weight = e.get("weight", 1.0)

            if pre_id not in self.id_to_idx or post_id not in self.id_to_idx:
                continue

            pre = self.id_to_idx[pre_id]
            post = self.id_to_idx[post_id]

            sign = nt_sign.get(nt, 0.5)
            w = syn_count * base_weight * sign * self.WEIGHT_SCALE

            # KC→KC 循环连接缩放：真实果蝇中 KC 通过 APL 中间神经元实现侧抑制，
            # 直接的 KC→KC 兴奋连接很弱。这里大幅降低以维持稀疏编码，
            # 防止激活扩散到整个 KC 群体。
            if pre in self.kc_idx_set and post in self.kc_idx_set:
                w *= 0.01

            # ALPN→KC 连接增强：确保嗅觉输入能有效驱动 KC
            if pre in self.alpn_idx_set and post in self.kc_idx_set:
                w *= 2.0

            rows.append(pre)
            cols.append(post)
            weights.append(w)

            # 记录 KC→MBON 连接
            if pre in self.kc_idx_set and post in self.mbon_idx_set:
                self.kc_to_mbon_edges.append((pre, post, len(weights) - 1))

        # 构建 CSR 稀疏矩阵
        self.W = sparse.csr_matrix(
            (weights, (rows, cols)),
            shape=(self.n, self.n)
        )

        # 为 STDP 单独存储 KC→MBON 连接的权重（可修改）
        # 即使 STDP 关闭也初始化权重（繁衍系统需要读取父母权重）
        if self.kc_to_mbon_edges:
            self.kc_mbon_pre = np.array([e[0] for e in self.kc_to_mbon_edges])
            self.kc_mbon_post = np.array([e[1] for e in self.kc_to_mbon_edges])
            self.kc_mbon_weight = np.array(
                [weights[e[2]] for e in self.kc_to_mbon_edges],
                dtype=np.float32
            )
            self.kc_mbon_base_weight = self.kc_mbon_weight.copy()

            # 预计算每条 KC→MBON 连接在 W.data 中的索引
            # 这样修改权重时可以直接更新 W 矩阵，不需要额外的 apply 步骤
            self._kc_mbon_data_idx = []
            for pre, post, _ in self.kc_to_mbon_edges:
                row_start = self.W.indptr[pre]
                row_end = self.W.indptr[pre + 1]
                found = -1
                for j in range(row_start, row_end):
                    if self.W.indices[j] == post:
                        found = j
                        break
                self._kc_mbon_data_idx.append(found)

            # 验证所有连接都找到了
            found_count = sum(1 for idx in self._kc_mbon_data_idx if idx >= 0)
            print(f"[蘑菇体 LIF] STDP 可塑连接: {len(self.kc_to_mbon_edges)} 条 KC→MBON "
                  f"(W矩阵匹配: {found_count}/{len(self.kc_to_mbon_edges)})")
        else:
            self.kc_mbon_pre = np.array([], dtype=np.int32)
            self.kc_mbon_post = np.array([], dtype=np.int32)
            self.kc_mbon_weight = np.array([], dtype=np.float32)
            self.kc_mbon_base_weight = np.array([], dtype=np.float32)
            self._kc_mbon_data_idx = []

    @property
    def kc_idx_set(self):
        if not hasattr(self, "_kc_idx_set"):
            self._kc_idx_set = set(self.kc_idx)
        return self._kc_idx_set

    @property
    def alpn_idx_set(self):
        if not hasattr(self, "_alpn_idx_set"):
            self._alpn_idx_set = set(self.alpn_idx)
        return self._alpn_idx_set

    @property
    def mbon_idx_set(self):
        if not hasattr(self, "_mbon_idx_set"):
            self._mbon_idx_set = set(self.mbon_idx)
        return self._mbon_idx_set

    def _init_state(self):
        """初始化神经元状态"""
        self.time_ms = 0
        self.V = np.zeros(self.n, dtype=np.float32)  # 膜电位
        self.refractory = np.zeros(self.n, dtype=np.int32)  # 不应期计数器
        self.last_spike = np.full(self.n, -10000, dtype=np.int32)  # 最近放电时间
        self.input_current = np.zeros(self.n, dtype=np.float32)  # 外部输入电流

        # Per-neuron 阈值：KC 阈值更高，实现稀疏编码
        # 真实果蝇中 KC 需要大量 ALPN 同步输入才能放电
        self.threshold = np.full(self.n, self.THRESHOLD, dtype=np.float32)
        for kc in self.kc_idx:
            self.threshold[kc] = 2.5  # KC 阈值更高，更难放电

        # === 内在可塑性调制器（运行时可调整）===
        # 全局兴奋性调制：>1 更容易放电，<1 更难放电
        # 由体细胞进化的基因型参数控制
        self.global_excitability = 1.0
        # 全局时间常数调制：>1 膜电位衰减更慢（反应慢但记忆长），<1 衰减更快
        self.global_tau_modulator = 1.0
        # 自发放电率调制：>1 更活跃，<1 更安静
        self.global_spontaneous_modulator = 1.0

        # APL 全局侧抑制参数（模拟果蝇 APL 中间神经元）
        # 当 KC 活跃时，APL 释放 GABA 抑制所有 KC，实现赢家通吃的稀疏编码
        self.apl_inhibition = 0.0  # 当前 APL 抑制强度
        self.apl_tau = 50.0  # APL 时间常数 (ms)
        self.apl_gain = 0.002  # APL 增益（每个活跃 KC 产生的抑制）

        # 放电率 EMA
        self.spike_rates = np.zeros(self.n, dtype=np.float32)
        self.rate_alpha = 0.001  # 1ms 步长，时间常数 ~1s

        # 异质基线电流（类似 DesktopFly）
        self.baseline = np.zeros(self.n, dtype=np.float32)
        for i in range(self.n):
            if i in self.kc_idx_set:
                self.baseline[i] = np.random.uniform(0.002, 0.008)
            elif i in self.alpn_idx:
                self.baseline[i] = 0.004
            elif i in self.dan_idx:
                self.baseline[i] = np.random.uniform(0.01, 0.03)
            elif i in self.mbon_idx:
                self.baseline[i] = np.random.uniform(0.02, 0.05)
            else:
                self.baseline[i] = np.random.uniform(0.01, 0.05)

        # 噪声
        self.noise_rate = 0.002
        self.noise_kick = 0.42

        # KC 自发放电（背景活动，泊松过程）
        # 真实果蝇 KC 自发放电率约 1-5 Hz，即使没有气味输入也会随机放电
        # 这是大脑"自发活动"的基础——不是外部刺激驱动，而是神经元自身的节律
        self.spontaneous_enabled = True
        self.spontaneous_rate = 1.2  # KC 自发放电率（Hz）
        self.spontaneous_current = 0.6  # 自发放电注入电流
        self.spontaneous_kc_count = 0

        self.mbon_spontaneous_baseline = 0.0  # 不额外加MBON基线，靠KC自发输入驱动

        # 统计
        self.total_spikes = 0
        self.stdp_update_count = 0

        # 当前感官状态
        self.current_odor = "none"
        self.punishment_strength = 0.0
        self.reward_strength = 0.0

    def set_odor_input(self, alpn_currents: Dict[int, float], odor_label: str = "unknown"):
        """
        设置嗅觉输入（ALPN 神经元电流）

        Args:
            alpn_currents: {alpn_index: current_nA}，ALPN 神经元的输入电流
            odor_label: 气味标签（用于记录）
        """
        self.input_current.fill(0)
        for idx, cur in alpn_currents.items():
            if 0 <= idx < self.n:
                self.input_current[idx] = cur
        self.current_odor = odor_label

    def set_punishment(self, strength: float = 1.0):
        """
        设置惩罚输入（PPL1 多巴胺神经元激活，模拟电击）

        Args:
            strength: 惩罚强度 0.0-1.0
        """
        self.punishment_strength = max(0.0, min(1.0, strength))
        # PPL1 神经元强激活
        ppl1_current = 8.0 + 7.0 * self.punishment_strength  # 8-15 nA
        for idx in self.ppl1_idx:
            self.input_current[idx] = max(self.input_current[idx], ppl1_current)

    def set_reward(self, strength: float = 1.0):
        """
        设置奖励输入（PAM 多巴胺神经元激活）

        Args:
            strength: 奖励强度 0.0-1.0
        """
        self.reward_strength = max(0.0, min(1.0, strength))
        pam_current = 2.0 + 3.0 * self.reward_strength  # 2-5 nA
        for idx in self.pam_idx:
            self.input_current[idx] = max(self.input_current[idx], pam_current)

    def clear_input(self):
        """清除所有外部输入"""
        self.input_current.fill(0)
        self.punishment_strength = 0.0
        self.reward_strength = 0.0
        self.current_odor = "none"

    def step(self, ms: int = 1):
        """
        运行仿真指定毫秒数

        Args:
            ms: 运行毫秒数
        """
        # 有效时间常数（受内在可塑性调制）
        effective_tau = self.TAU_M * self.global_tau_modulator
        decay = np.exp(-1.0 / effective_tau)

        for _ in range(ms):
            self.time_ms += 1

            # 1. 膜电位更新（泄漏 + 基线 + 输入 + 噪声）
            active = self.refractory <= 0
            self.V[active] = (
                self.V[active] * decay
                + self.baseline[active]
                + self.input_current[active] * 0.12  # 输入电流缩放
            )

            # 1.5 KC 自发放电（背景活动，泊松过程）
            # 真实大脑不是只有外部刺激才活动——神经元有自发放电节律
            # 这是"活的大脑"和"反射弧"的根本区别
            if self.spontaneous_enabled:
                spont_prob = (self.spontaneous_rate * self.global_spontaneous_modulator) / 1000.0  # 每ms放电概率
                kc_idx_arr = np.asarray(self.kc_idx)
                spont_mask = np.random.random(len(kc_idx_arr)) < spont_prob
                spont_kc = kc_idx_arr[spont_mask]
                if len(spont_kc) > 0:
                    self.V[spont_kc] += self.spontaneous_current
                    self.spontaneous_kc_count = len(spont_kc)

            # 1.6 MBON 额外基线（增强背景活动，让 MBON 对 KC 自发输入更敏感）
            self.V[self.mbon_idx] += self.mbon_spontaneous_baseline

            # APL 全局侧抑制：活跃 KC 触发 APL，APL 抑制所有 KC
            # 实现赢家通吃的稀疏编码，只有最强的 KC 能维持放电
            kc_active_count = int(np.sum(self.spike_rates[self.kc_idx] > 10.0))
            apl_target = kc_active_count * self.apl_gain
            apl_decay = np.exp(-1.0 / self.apl_tau)
            self.apl_inhibition = self.apl_inhibition * apl_decay + apl_target * (1 - apl_decay)
            # 对所有 KC 施加抑制
            self.V[self.kc_idx] -= self.apl_inhibition

            # 不应期递减
            self.refractory[self.refractory > 0] -= 1

            # 噪声
            noise_mask = np.random.random(self.n) < self.noise_rate
            self.V[noise_mask & active] += self.noise_kick

            # 2. 放电检测（使用 per-neuron 阈值，受全局兴奋性调制）
            # 有效阈值 = threshold / global_excitability
            # global_excitability > 1 → 有效阈值降低 → 更容易放电
            effective_threshold = self.threshold / self.global_excitability
            spiked = (self.V >= effective_threshold) & (self.refractory <= 0)
            spiked_idx = np.where(spiked)[0]

            if len(spiked_idx) > 0:
                self.V[spiked] = self.V_RESET
                self.refractory[spiked] = self.REFRACTORY_MS
                self.last_spike[spiked_idx] = self.time_ms
                self.total_spikes += len(spiked_idx)

                # 3. 突触传播（稀疏矩阵乘法）
                # 只传播放电神经元的输出
                spike_vec = np.zeros(self.n, dtype=np.float32)
                spike_vec[spiked_idx] = 1.0

                synaptic_input = self.W.T.dot(spike_vec)  # 后突触神经元收到的输入
                self.V += synaptic_input
                self.V = np.clip(self.V, -2.0, 5.0)  # 膜电位钳制

                # 4. STDP 更新
                if self.enable_stdp and len(self.kc_mbon_pre) > 0:
                    self._update_stdp(spiked_idx)

            # 5. 放电率 EMA 更新
            self.spike_rates *= (1 - self.rate_alpha)
            if len(spiked_idx) > 0:
                self.spike_rates[spiked_idx] += self.rate_alpha * 1000  # Hz

    def _update_stdp(self, spiked_idx: np.ndarray):
        """
        STDP 突触可塑性更新（果蝇蘑菇体学习规则）

        真实果蝇蘑菇体学习机制:
          - 气味激活特定 KC 子集
          - 电击激活 PPL1 DAN，释放多巴胺
          - 当 KC 放电 + 多巴胺同时存在时，KC→MBON 突触被抑制（LTD）
          - 不需要 MBON 放电，多巴胺直接作用于 KC 轴突末梢
          - 结果: 该气味激活的 KC 不再能激活 MBON → 回避行为

        奖励学习（PAM）:
          - KC 放电 + PAM 多巴胺 → KC→MBON 突触增强（LTP）→ 趋向行为
        """
        if self.punishment_strength <= 0 and self.reward_strength <= 0:
            return

        # 找到最近放电的 KC（时间窗口内，向量化操作）
        time_window = int(self.STDP_TAU_PRE)
        kc_last_spike = self.last_spike[self.kc_idx]
        kc_recent_mask = (
            (self.time_ms - kc_last_spike >= 0) &
            (self.time_ms - kc_last_spike <= time_window)
        )
        recent_kc = np.array(self.kc_idx)[kc_recent_mask]

        if len(recent_kc) == 0:
            return

        # 确定学习方向和学习率
        if self.punishment_strength > 0:
            lr = self.STDP_LR_PUNISH * self.punishment_strength
            direction = -1.0  # 惩罚 → 权重减小（抑制/LTD）
        else:
            lr = self.STDP_LR_REWARD * self.reward_strength
            direction = 1.0   # 奖励 → 权重增大（增强/LTP）

        if lr <= 0:
            return

        # 批量更新：所有最近放电 KC 的输出连接
        # 不需要 MBON 放电，多巴胺直接作用于 KC 轴突末梢
        update_mask = np.isin(self.kc_mbon_pre, recent_kc)

        if np.any(update_mask):
            # 绝对学习率：每次更新固定 delta
            delta = lr * direction
            self.kc_mbon_weight[update_mask] += delta
            self.kc_mbon_weight = np.clip(
                self.kc_mbon_weight,
                self.STDP_WEIGHT_MIN,
                self.STDP_WEIGHT_MAX
            )
            self.stdp_update_count += int(np.sum(update_mask))

            # 立即更新 W 矩阵中对应位置的值
            for i in np.where(update_mask)[0]:
                data_idx = self._kc_mbon_data_idx[i]
                if data_idx >= 0:
                    self.W.data[data_idx] = self.kc_mbon_weight[i]

    def get_mbon_rates(self) -> Dict[str, float]:
        """获取各 MBON 类型的平均放电率 (Hz)"""
        rates = {}
        for mtype, indices in self.mbon_types.items():
            if indices:
                avg_rate = float(np.mean(self.spike_rates[indices]))
                rates[mtype] = avg_rate
        return rates

    def get_behavior(self) -> Dict:
        """
        从 MBON 输出解码行为信号

        果蝇蘑菇体输出编码:
          - MBON 放电率高 → 趋向/接近行为（该气味预示安全/奖励）
          - MBON 放电率低 → 回避/逃跑行为（该气味预示惩罚/危险）
          - 左右 MBON 差异 → 转向偏向

        基线: 自发活动时 MBON 平均放电率 ~30-50 Hz
        强气味刺激时 MBON 可达 100-200 Hz
        学习抑制后 MBON 可降到 20-50 Hz
        """
        mbon_rates = self.get_mbon_rates()
        avg_mbon = float(np.mean(self.spike_rates[self.mbon_idx])) if self.mbon_idx else 0

        # 行为解码（基于 MBON 放电率的相对水平）
        # 中性区域: MBON < 15Hz = 静息/随机探索
        # 弱趋向: MBON 15-50Hz
        # 强趋向: MBON > 50Hz
        # 回避: 惩罚激活时 MBON 被抑制，表现为回避
        if avg_mbon < 15.0:
            # 静息 = 中性探索（既不趋向也不回避）
            approach = 0.35
            avoidance = 0.25
        elif avg_mbon < 50.0:
            # 低活动 = 弱趋向
            approach = 0.4 + (avg_mbon - 15) / 35 * 0.3
            avoidance = max(0.1, 0.3 - (avg_mbon - 15) / 35 * 0.2)
        else:
            # 高活动 = 强趋向
            approach = min(1.0, 0.7 + (avg_mbon - 50) / 50 * 0.3)
            avoidance = max(0.0, 0.1 - (avg_mbon - 50) / 50 * 0.1)

        # 惩罚激活时强化回避（即使 MBON 还没降下来，电击本身就触发回避）
        if self.punishment_strength > 0:
            avoidance = max(avoidance, 0.5 + self.punishment_strength * 0.5)
            approach = min(approach, 0.3)

        behavior = {
            "approach": approach,       # 趋向倾向 0-1
            "avoidance": avoidance,     # 回避倾向 0-1
            "arousal": float(np.mean(self.spike_rates)) * 0.05,  # 唤醒水平
            "mbon_avg_rate": avg_mbon,
            "mbon_rates": mbon_rates,
            "kc_active": int(np.sum(self.spike_rates[self.kc_idx] > 1.0)) if self.kc_idx else 0,
            "punishment": self.punishment_strength,
            "reward": self.reward_strength,
            "odor": self.current_odor,
        }
        return behavior

    def get_state(self) -> MBState:
        """获取当前仿真状态快照"""
        return MBState(
            time_ms=self.time_ms,
            kc_spikes=int(np.sum(self.spike_rates[self.kc_idx] > 0.5)) if self.kc_idx else 0,
            alpn_spikes=int(np.sum(self.spike_rates[self.alpn_idx] > 0.5)) if self.alpn_idx else 0,
            dan_spikes=int(np.sum(self.spike_rates[self.dan_idx] > 0.5)) if self.dan_idx else 0,
            mbon_spikes=int(np.sum(self.spike_rates[self.mbon_idx] > 0.5)) if self.mbon_idx else 0,
            total_spikes=self.total_spikes,
            mbon_rates=self.get_mbon_rates(),
            punishment_active=self.punishment_strength > 0,
            reward_active=self.reward_strength > 0,
            odor_label=self.current_odor,
            stdp_updates=self.stdp_update_count,
        )

    def reset_learning(self):
        """重置所有 STDP 学习（恢复初始权重）"""
        if self.enable_stdp:
            self.kc_mbon_weight = self.kc_mbon_base_weight.copy()
            self.stdp_update_count = 0
            print("[蘑菇体 LIF] 学习已重置，权重恢复初始值")

    def get_weight_changes(self) -> Dict:
        """获取 STDP 权重变化统计"""
        if len(self.kc_mbon_weight) == 0:
            return {"total": 0, "changed": 0, "avg_change": 0.0}

        changes = self.kc_mbon_weight - self.kc_mbon_base_weight
        changed = int(np.sum(np.abs(changes) > 0.0001))
        return {
            "total": len(self.kc_mbon_weight),
            "changed": changed,
            "avg_change": float(np.mean(changes)),
            "max_increase": float(np.max(changes)),
            "max_decrease": float(np.min(changes)),
        }

    def save_weights(self, path: str) -> bool:
        """
        保存 STDP 权重到文件（.npz 格式）

        Args:
            path: 保存路径（.npz 扩展名）

        Returns:
            是否保存成功
        """
        if not self.enable_stdp or len(self.kc_mbon_weight) == 0:
            print("[蘑菇体 LIF] STDP 未启用或无可塑连接，跳过保存")
            return False

        try:
            np.savez(
                path,
                kc_mbon_weight=self.kc_mbon_weight,
                kc_mbon_pre=self.kc_mbon_pre,
                kc_mbon_post=self.kc_mbon_post,
                stdp_update_count=np.array([self.stdp_update_count]),
                save_time=np.array([time.time()]),
            )
            changes = self.get_weight_changes()
            print(f"[蘑菇体 LIF] 权重已保存: {path} "
                  f"(可塑连接={changes['total']}, 已修改={changes['changed']}, "
                  f"STDP更新={self.stdp_update_count})")
            return True
        except Exception as e:
            print(f"[蘑菇体 LIF] 权重保存失败: {e}")
            return False

    def load_weights(self, path: str) -> bool:
        """
        从文件加载 STDP 权重

        Args:
            path: 权重文件路径（.npz 格式）

        Returns:
            是否加载成功
        """
        if not self.enable_stdp or len(self.kc_mbon_weight) == 0:
            print("[蘑菇体 LIF] STDP 未启用或无可塑连接，跳过加载")
            return False

        if not os.path.exists(path):
            print(f"[蘑菇体 LIF] 权重文件不存在: {path}")
            return False

        try:
            data = np.load(path, allow_pickle=True)

            # 验证连接数量匹配
            loaded_weight = data['kc_mbon_weight']
            if len(loaded_weight) != len(self.kc_mbon_weight):
                print(f"[蘑菇体 LIF] 权重文件连接数不匹配 "
                      f"(文件={len(loaded_weight)}, 当前={len(self.kc_mbon_weight)})，跳过加载")
                return False

            # 验证连接的前/后神经元匹配（防止回路版本不匹配）
            if 'kc_mbon_pre' in data and 'kc_mbon_post' in data:
                if not np.array_equal(data['kc_mbon_pre'], self.kc_mbon_pre) or \
                   not np.array_equal(data['kc_mbon_post'], self.kc_mbon_post):
                    print("[蘑菇体 LIF] 权重文件连接拓扑不匹配（回路版本可能不同），跳过加载")
                    return False

            # 加载权重
            self.kc_mbon_weight = loaded_weight.astype(np.float32)
            self.stdp_update_count = int(data['stdp_update_count'][0]) if 'stdp_update_count' in data else 0

            # 同步更新 W 矩阵
            for i in range(len(self.kc_mbon_weight)):
                data_idx = self._kc_mbon_data_idx[i]
                if data_idx >= 0:
                    self.W.data[data_idx] = self.kc_mbon_weight[i]

            changes = self.get_weight_changes()
            save_time = data['save_time'][0] if 'save_time' in data else 0
            time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(save_time)) if save_time else "未知"
            print(f"[蘑菇体 LIF] 权重已加载: {path} "
                  f"(保存时间={time_str}, 可塑连接={changes['total']}, "
                  f"已修改={changes['changed']}, STDP更新={self.stdp_update_count})")
            return True
        except Exception as e:
            print(f"[蘑菇体 LIF] 权重加载失败: {e}")
            return False


# ============== 快速测试 ==============
if __name__ == "__main__":
    import time

    circuit_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "mushroom_body", "circuit.json"
    )

    print("=" * 70)
    print("蘑菇体 LIF 仿真 - 快速测试")
    print("=" * 70)

    sim = MushroomBodyLIF(circuit_path, enable_stdp=True)

    # 测试 1: 自发活动 1 秒
    print("\n[测试 1] 自发活动 1 秒...")
    t0 = time.time()
    sim.step(1000)
    elapsed = time.time() - t0
    state = sim.get_state()
    print(f"  耗时: {elapsed:.2f}s ({1000/elapsed:.0f}x 实时速度)")
    print(f"  总放电: {state.total_spikes}")
    print(f"  KC 活跃: {state.kc_spikes}, ALPN: {state.alpn_spikes}, "
          f"DAN: {state.dan_spikes}, MBON: {state.mbon_spikes}")

    # 测试 2: 嗅觉输入
    print("\n[测试 2] 嗅觉输入（激活前 100 个 ALPN）...")
    alpn_input = {i: 10.0 for i in sim.alpn_idx[:100]}
    sim.set_odor_input(alpn_input, odor_label="test_odor")
    sim.step(500)
    behavior = sim.get_behavior()
    print(f"  气味: {behavior['odor']}")
    print(f"  MBON 平均放电率: {behavior['mbon_avg_rate']:.1f} Hz")
    print(f"  趋向倾向: {behavior['approach']:.2f}, 回避倾向: {behavior['avoidance']:.2f}")
    print(f"  KC 活跃数: {behavior['kc_active']}")

    # 测试 3: 惩罚输入
    print("\n[测试 3] 惩罚输入（电击，PPL1 激活）...")
    sim.set_punishment(1.0)
    sim.step(200)
    state = sim.get_state()
    print(f"  惩罚激活: {state.punishment_active}")
    print(f"  DAN 活跃: {state.dan_spikes}")
    print(f"  STDP 更新数: {state.stdp_updates}")

    # 测试 4: 学习后行为变化
    print("\n[测试 4] 学习后行为变化...")
    sim.clear_input()
    sim.step(500)  # 等待恢复
    sim.set_odor_input(alpn_input, odor_label="test_odor")
    sim.step(500)
    behavior_after = sim.get_behavior()
    print(f"  学习后 MBON 平均放电率: {behavior_after['mbon_avg_rate']:.1f} Hz")
    print(f"  学习后趋向倾向: {behavior_after['approach']:.2f}")
    print(f"  权重变化: {sim.get_weight_changes()}")

    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)
