#!/usr/bin/env python3
"""
内在可塑性 (Intrinsic Plasticity)

神经元层面的自调整机制：
- 放电阈值自调整（经常放电的神经元阈值降低，不常放电的升高）
- 膜时间常数自调整（反应快还是慢）
- 自发放电率自调整（好动还是好静）

这是个体内部进化的神经基础——
不是改变突触连接，而是改变神经元本身的响应特性。
对应真实生物的"内在可塑性"（Intrinsic Plasticity）。
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class IntrinsicPlasticityConfig:
    """内在可塑性配置"""
    # 阈值调整
    threshold_adaptation_rate: float = 0.001  # 阈值调整速率
    target_firing_rate: float = 5.0  # 目标放电率（Hz）
    min_threshold: float = 0.5  # 最小阈值
    max_threshold: float = 5.0  # 最大阈值

    # 时间常数调整
    tau_adaptation_rate: float = 0.0005  # 时间常数调整速率
    min_tau: float = 5.0  # 最小膜时间常数（ms）
    max_tau: float = 50.0  # 最大膜时间常数（ms）

    # 自发放电率调整
    spontaneous_adaptation_rate: float = 0.001  # 自发放电率调整速率
    min_spontaneous: float = 0.1  # 最小自发放电率（Hz）
    max_spontaneous: float = 5.0  # 最大自发放电率（Hz）

    # 全局调制（由基因型参数控制）
    global_excitability: float = 1.0  # 全局兴奋性调制
    adaptation_enabled: bool = True  # 是否启用内在可塑性


class IntrinsicPlasticity:
    """
    内在可塑性引擎

    监控每个神经元的放电率，自动调整其阈值、时间常数和自发放电率，
    使网络维持在合适的工作点（既不太沉默也不太亢奋）。

    这是一种稳态可塑性（Homeostatic Plasticity），
    对应真实生物神经系统中的"内在可塑性"机制。
    """

    def __init__(self, n_neurons: int, config: Optional[IntrinsicPlasticityConfig] = None):
        """
        初始化内在可塑性引擎

        Args:
            n_neurons: 神经元数量
            config: 配置，None=使用默认配置
        """
        self.n_neurons = n_neurons
        self.config = config or IntrinsicPlasticityConfig()

        # 神经元参数（可调整）
        self.thresholds = np.ones(n_neurons, dtype=np.float32) * 1.0  # 放电阈值
        self.tau_m = np.ones(n_neurons, dtype=np.float32) * 20.0  # 膜时间常数（ms）
        self.spontaneous_rates = np.ones(n_neurons, dtype=np.float32) * 1.2  # 自发放电率（Hz）

        # 放电率跟踪（滑动平均）
        self.firing_rates = np.zeros(n_neurons, dtype=np.float32)
        self.rate_decay = 0.99  # 放电率滑动平均衰减

        # 统计
        self.total_adjustments = 0
        self.last_adjustment_time = 0.0

        print(f"[IntrinsicPlasticity] 初始化完成: {n_neurons} 个神经元")
        print(f"[IntrinsicPlasticity] 目标放电率: {self.config.target_firing_rate} Hz")
        print(f"[IntrinsicPlasticity] 调整速率: threshold={self.config.threshold_adaptation_rate}, "
              f"tau={self.config.tau_adaptation_rate}")

    def update(self, spikes: np.ndarray, dt_ms: float = 1.0):
        """
        更新内在可塑性（每帧调用）

        Args:
            spikes: 当前帧放电的神经元（0/1数组）
            dt_ms: 时间步长（ms）
        """
        if not self.config.adaptation_enabled:
            return

        # 更新放电率滑动平均
        instantaneous_rate = spikes * (1000.0 / dt_ms)  # 转换为 Hz
        self.firing_rates = self.firing_rates * self.rate_decay + instantaneous_rate * (1 - self.rate_decay)

        # 每 100ms 调整一次参数（避免每帧都调导致振荡）
        self.last_adjustment_time += dt_ms
        if self.last_adjustment_time < 100.0:
            return
        self.last_adjustment_time = 0.0

        self._adjust_thresholds()
        self._adjust_tau()
        self._adjust_spontaneous()
        self.total_adjustments += 1

    def _adjust_thresholds(self):
        """调整放电阈值：放电率太高→阈值升高；太低→阈值降低"""
        rate_diff = self.firing_rates - self.config.target_firing_rate

        # 阈值调整方向：放电率高→阈值升高（更难放电）
        threshold_delta = rate_diff * self.config.threshold_adaptation_rate * self.config.global_excitability

        self.thresholds += threshold_delta
        self.thresholds = np.clip(self.thresholds, self.config.min_threshold, self.config.max_threshold)

    def _adjust_tau(self):
        """调整膜时间常数：放电率太高→tau减小（反应更快，更难累积）；太低→tau增大"""
        rate_diff = self.firing_rates - self.config.target_firing_rate

        # tau调整方向：放电率高→tau减小（膜电位衰减更快，更难累积到阈值）
        tau_delta = -rate_diff * self.config.tau_adaptation_rate * self.config.global_excitability

        self.tau_m += tau_delta
        self.tau_m = np.clip(self.tau_m, self.config.min_tau, self.config.max_tau)

    def _adjust_spontaneous(self):
        """调整自发放电率：整体放电率太低→自发放电率升高；太高→降低"""
        avg_rate = np.mean(self.firing_rates)
        rate_diff = avg_rate - self.config.target_firing_rate

        # 自发放电率调整：整体太低→升高自发放电（增加背景活动）
        spontaneous_delta = -rate_diff * self.config.spontaneous_adaptation_rate * self.config.global_excitability

        self.spontaneous_rates += spontaneous_delta
        self.spontaneous_rates = np.clip(self.spontaneous_rates,
                                          self.config.min_spontaneous,
                                          self.config.max_spontaneous)

    def get_thresholds(self) -> np.ndarray:
        """获取当前阈值数组（应用全局兴奋性调制）"""
        return self.thresholds / self.config.global_excitability

    def get_tau(self) -> np.ndarray:
        """获取当前膜时间常数数组"""
        return self.tau_m

    def get_spontaneous_rates(self) -> np.ndarray:
        """获取当前自发放电率数组（应用全局兴奋性调制）"""
        return self.spontaneous_rates * self.config.global_excitability

    def set_global_excitability(self, value: float):
        """设置全局兴奋性（由基因型参数控制）"""
        self.config.global_excitability = max(0.3, min(2.0, value))

    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "n_neurons": self.n_neurons,
            "total_adjustments": self.total_adjustments,
            "avg_threshold": float(np.mean(self.thresholds)),
            "min_threshold": float(np.min(self.thresholds)),
            "max_threshold": float(np.max(self.thresholds)),
            "avg_tau": float(np.mean(self.tau_m)),
            "avg_spontaneous": float(np.mean(self.spontaneous_rates)),
            "avg_firing_rate": float(np.mean(self.firing_rates)),
            "global_excitability": self.config.global_excitability,
        }

    def reset(self):
        """重置所有参数到默认值"""
        self.thresholds[:] = 1.0
        self.tau_m[:] = 20.0
        self.spontaneous_rates[:] = 1.2
        self.firing_rates[:] = 0.0
        self.total_adjustments = 0
        self.last_adjustment_time = 0.0


def test_intrinsic_plasticity():
    """测试内在可塑性"""
    print("=" * 60)
    print("内在可塑性测试")
    print("=" * 60)

    n_neurons = 100
    ip = IntrinsicPlasticity(n_neurons)

    # 测试1：高放电率→阈值升高
    print("\n--- 测试1: 高放电率导致阈值升高 ---")
    high_spikes = np.ones(n_neurons, dtype=np.float32)  # 所有神经元都放电
    for i in range(500):
        ip.update(high_spikes, dt_ms=1.0)

    stats = ip.get_stats()
    print(f"  平均阈值: {stats['avg_threshold']:.3f} (初始1.0，应升高)")
    print(f"  平均放电率: {stats['avg_firing_rate']:.1f} Hz")
    print(f"  调整次数: {stats['total_adjustments']}")

    # 测试2：低放电率→阈值降低
    print("\n--- 测试2: 低放电率导致阈值降低 ---")
    ip.reset()
    low_spikes = np.zeros(n_neurons, dtype=np.float32)  # 没有神经元放电
    for i in range(500):
        ip.update(low_spikes, dt_ms=1.0)

    stats = ip.get_stats()
    print(f"  平均阈值: {stats['avg_threshold']:.3f} (初始1.0，应降低)")
    print(f"  平均自发放电率: {stats['avg_spontaneous']:.3f} (应升高)")

    # 测试3：全局兴奋性调制
    print("\n--- 测试3: 全局兴奋性调制 ---")
    ip.set_global_excitability(1.5)
    thresholds = ip.get_thresholds()
    print(f"  全局兴奋性=1.5, 有效阈值均值: {np.mean(thresholds):.3f} (应降低)")

    ip.set_global_excitability(0.5)
    thresholds = ip.get_thresholds()
    print(f"  全局兴奋性=0.5, 有效阈值均值: {np.mean(thresholds):.3f} (应升高)")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    test_intrinsic_plasticity()
