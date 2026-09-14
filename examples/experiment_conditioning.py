#!/usr/bin/env python3
"""
嗅觉条件反射实验 (Olfactory Conditioning Experiment)

验证果蝇蘑菇体的联想学习能力：
  训练前: 气味 A 不引起回避（MBON 正常放电，趋向倾向高）
  训练中: 气味 A + 电击（惩罚）同时呈现，多次重复
  训练后: 单独呈现气味 A → 回避反应（MBON 放电降低，趋向倾向降低）

这是经典的巴甫洛夫条件反射，果蝇蘑菇体是学习记忆的中心。

实验设计:
  - 气味 A (CS+): 激活 ALPN 子集 1（前 100 个）
  - 气味 B (CS-): 激活 ALPN 子集 2（100-200 个），对照
  - 电击 (US): PPL1 DAN 强激活
  - 训练: 气味 A + 电击，重复 N 次
  - 测试: 单独呈现气味 A 和 B，比较 MBON 反应

用法:
  python3 experiment_conditioning.py [训练次数]
"""

import os
import sys
import time
import json
import numpy as np
from mb_lif import MushroomBodyLIF


class ConditioningExperiment:
    """嗅觉条件反射实验"""

    def __init__(self, circuit_path: str, n_train: int = 10):
        """
        Args:
            circuit_path: 蘑菇体回路路径
            n_train: 训练次数
        """
        self.n_train = n_train
        self.sim = MushroomBodyLIF(circuit_path, enable_stdp=True)

        # 定义两种气味的 ALPN 子集
        # 气味 A (CS+): 前 100 个 ALPN
        # 气味 B (CS-): 100-200 个 ALPN
        self.odor_a_alpn = self.sim.alpn_idx[:100]
        self.odor_b_alpn = self.sim.alpn_idx[100:200]

        # 实验参数
        self.odor_duration_ms = 500      # 气味呈现时间
        self.shock_duration_ms = 200     # 电击时间
        self.shock_overlap_ms = 200      # 气味和电击重叠时间
        self.inter_trial_ms = 1000       # 试次间隔
        self.test_duration_ms = 500      # 测试时气味呈现时间
        self.rest_after_train_ms = 2000  # 训练后休息时间

        # 记录数据
        self.results = {
            "pre_train": {},
            "training": [],
            "post_train": {},
            "weight_changes": {},
        }

    def _present_odor(self, alpn_subset: list, label: str, duration_ms: int):
        """呈现气味，返回期间的平均行为"""
        alpn_input = {i: 12.0 for i in alpn_subset}
        self.sim.set_odor_input(alpn_input, odor_label=label)

        mbon_rates = []
        approach_vals = []
        avoidance_vals = []
        kc_active = []

        for _ in range(duration_ms):
            self.sim.step(1)
            behavior = self.sim.get_behavior()
            mbon_rates.append(behavior["mbon_avg_rate"])
            approach_vals.append(behavior["approach"])
            avoidance_vals.append(behavior["avoidance"])
            kc_active.append(behavior["kc_active"])

        self.sim.clear_input()

        return {
            "mbon_avg": float(np.mean(mbon_rates[-100:])),  # 最后 100ms 平均
            "mbon_peak": float(np.max(mbon_rates)),
            "approach": float(np.mean(approach_vals[-100:])),
            "avoidance": float(np.mean(avoidance_vals[-100:])),
            "kc_active_avg": float(np.mean(kc_active[-100:])),
        }

    def _present_shock(self, strength: float = 1.0, duration_ms: int = 200):
        """呈现电击（惩罚）"""
        self.sim.set_punishment(strength)
        self.sim.step(duration_ms)
        self.sim.punishment_strength = 0.0  # 清除惩罚标记，但电流会自然衰减

    def _reset_network_state(self):
        """
        重置网络状态（膜电位、不应期、APL抑制等），但保留学习到的权重。
        这确保训练前后的测试条件一致，只有突触权重不同。
        """
        sim = self.sim
        sim.V.fill(0)
        sim.refractory.fill(0)
        sim.last_spike.fill(-10000)
        sim.input_current.fill(0)
        sim.spike_rates.fill(0)
        sim.apl_inhibition = 0.0
        sim.time_ms = 0
        sim.total_spikes = 0
        # 注意：不重置 stdp_update_count 和 kc_mbon_weight（保留学习成果）

    def run_pre_train_test(self):
        """训练前测试：呈现气味 A 和 B，记录基线反应"""
        print("\n" + "=" * 70)
        print("【训练前测试】基线反应")
        print("=" * 70)

        # 重置网络状态，确保测试条件一致
        self._reset_network_state()
        self.sim.step(1000)  # 基线恢复

        # 测试气味 A
        print("\n  呈现气味 A (CS+) ...")
        result_a = self._present_odor(self.odor_a_alpn, "odor_A", self.test_duration_ms)
        print(f"    MBON 放电率: {result_a['mbon_avg']:.1f} Hz")
        print(f"    趋向倾向: {result_a['approach']:.2f}")
        print(f"    回避倾向: {result_a['avoidance']:.2f}")
        print(f"    KC 活跃数: {result_a['kc_active_avg']:.0f}")

        # 休息
        self.sim.step(self.inter_trial_ms)

        # 测试气味 B
        print("\n  呈现气味 B (CS-, 对照) ...")
        result_b = self._present_odor(self.odor_b_alpn, "odor_B", self.test_duration_ms)
        print(f"    MBON 放电率: {result_b['mbon_avg']:.1f} Hz")
        print(f"    趋向倾向: {result_b['approach']:.2f}")
        print(f"    回避倾向: {result_b['avoidance']:.2f}")
        print(f"    KC 活跃数: {result_b['kc_active_avg']:.0f}")

        self.results["pre_train"] = {"odor_A": result_a, "odor_B": result_b}

        # 学习指数（气味 A 和 B 的差异，训练前应该接近 0）
        learning_index_pre = result_a["avoidance"] - result_b["avoidance"]
        print(f"\n  训练前学习指数 (A回避 - B回避): {learning_index_pre:.3f} (接近 0 = 无偏好)")

        return result_a, result_b

    def run_training(self):
        """训练：气味 A + 电击，重复 N 次"""
        print("\n" + "=" * 70)
        print(f"【训练阶段】气味 A + 电击，重复 {self.n_train} 次")
        print("=" * 70)

        for trial in range(self.n_train):
            print(f"\n  试次 {trial + 1}/{self.n_train}:")

            # 呈现气味 A
            alpn_input = {i: 12.0 for i in self.odor_a_alpn}
            self.sim.set_odor_input(alpn_input, odor_label="odor_A+shock")

            # 气味先呈现 300ms，然后电击重叠 200ms
            self.sim.step(300)

            # 电击开始（与气味重叠）
            print(f"    气味 A 呈现中... 电击开始!")
            self.sim.set_punishment(1.0)
            self.sim.step(self.shock_duration_ms)

            # 电击结束，气味再持续 100ms
            self.sim.punishment_strength = 0.0
            self.sim.step(100)

            # 清除输入
            self.sim.clear_input()

            # 记录本次试次的状态
            state = self.sim.get_state()
            weight_changes = self.sim.get_weight_changes()
            trial_data = {
                "trial": trial + 1,
                "total_spikes": state.total_spikes,
                "stdp_updates": state.stdp_updates,
                "weights_changed": weight_changes["changed"],
                "avg_weight_change": weight_changes["avg_change"],
            }
            self.results["training"].append(trial_data)

            print(f"    STDP 更新累计: {state.stdp_updates}")
            print(f"    权重变化连接数: {weight_changes['changed']}/{weight_changes['total']}")
            print(f"    平均权重变化: {weight_changes['avg_change']:.4f}")

            # 试次间隔
            if trial < self.n_train - 1:
                self.sim.step(self.inter_trial_ms)

        # 训练后重置网络状态（保留学习到的权重），确保测试条件一致
        print(f"\n  训练完成，重置网络状态（保留学习权重）...")
        self._reset_network_state()
        self.sim.step(1000)  # 基线恢复

    def run_post_train_test(self):
        """训练后测试：呈现气味 A 和 B，比较学习效果"""
        print("\n" + "=" * 70)
        print("【训练后测试】学习效果验证")
        print("=" * 70)

        # 测试气味 A（应该表现出回避）
        print("\n  呈现气味 A (CS+, 训练过) ...")
        result_a = self._present_odor(self.odor_a_alpn, "odor_A_post", self.test_duration_ms)
        print(f"    MBON 放电率: {result_a['mbon_avg']:.1f} Hz")
        print(f"    趋向倾向: {result_a['approach']:.2f}")
        print(f"    回避倾向: {result_a['avoidance']:.2f}")
        print(f"    KC 活跃数: {result_a['kc_active_avg']:.0f}")

        # 休息
        self.sim.step(self.inter_trial_ms)

        # 测试气味 B（对照，应该没有回避）
        print("\n  呈现气味 B (CS-, 未训练) ...")
        result_b = self._present_odor(self.odor_b_alpn, "odor_B_post", self.test_duration_ms)
        print(f"    MBON 放电率: {result_b['mbon_avg']:.1f} Hz")
        print(f"    趋向倾向: {result_b['approach']:.2f}")
        print(f"    回避倾向: {result_b['avoidance']:.2f}")
        print(f"    KC 活跃数: {result_b['kc_active_avg']:.0f}")

        self.results["post_train"] = {"odor_A": result_a, "odor_B": result_b}

        # 学习指数
        learning_index_post = result_a["avoidance"] - result_b["avoidance"]
        print(f"\n  训练后学习指数 (A回避 - B回避): {learning_index_post:.3f}")

        # 比较训练前后
        pre_a = self.results["pre_train"]["odor_A"]
        pre_b = self.results["pre_train"]["odor_B"]

        print("\n  --- 学习效果对比 ---")
        print(f"  气味 A MBON 放电率: {pre_a['mbon_avg']:.1f} → {result_a['mbon_avg']:.1f} Hz "
              f"(变化: {result_a['mbon_avg'] - pre_a['mbon_avg']:+.1f})")
        print(f"  气味 B MBON 放电率: {pre_b['mbon_avg']:.1f} → {result_b['mbon_avg']:.1f} Hz "
              f"(变化: {result_b['mbon_avg'] - pre_b['mbon_avg']:+.1f})")
        print(f"  气味 A 回避倾向: {pre_a['avoidance']:.2f} → {result_a['avoidance']:.2f}")
        print(f"  气味 B 回避倾向: {pre_b['avoidance']:.2f} → {result_b['avoidance']:.2f}")

        # 权重变化
        weight_changes = self.sim.get_weight_changes()
        self.results["weight_changes"] = weight_changes
        print(f"\n  --- 突触权重变化 ---")
        print(f"  总 KC→MBON 连接: {weight_changes['total']}")
        print(f"  发生变化的连接: {weight_changes['changed']} ({weight_changes['changed']/weight_changes['total']*100:.1f}%)")
        print(f"  平均权重变化: {weight_changes['avg_change']:.4f}")
        print(f"  最大增强: {weight_changes['max_increase']:.4f}")
        print(f"  最大抑制: {weight_changes['max_decrease']:.4f}")

        # 判断是否学会（特异性学习：气味 A MBON 下降，气味 B 基本不变）
        a_change_pct = (result_a["mbon_avg"] - pre_a["mbon_avg"]) / max(1.0, pre_a["mbon_avg"]) * 100
        b_change_pct = (result_b["mbon_avg"] - pre_b["mbon_avg"]) / max(1.0, pre_b["mbon_avg"]) * 100

        learned = (
            a_change_pct < -5.0 and  # 气味 A MBON 下降 >5%
            b_change_pct > -3.0 and  # 气味 B 基本不变或下降 <3%
            a_change_pct < b_change_pct - 2.0  # 气味 A 下降比 B 多 >2%
        )

        print("\n" + "=" * 70)
        if learned:
            print("✅ 学习成功！果蝇学会了特异性回避气味 A（与电击关联的气味）")
            print(f"   气味 A MBON 放电率变化: {a_change_pct:+.1f}% (特异性抑制)")
            print(f"   气味 B MBON 放电率变化: {b_change_pct:+.1f}% (对照，基本不变)")
            print("   这证明蘑菇体联想学习机制在工作：KC→MBON 突触被多巴胺依赖性抑制")
        else:
            print("⚠️  学习效果不明显，可能需要更多训练次数或调整参数")
            print(f"   气味 A MBON 变化: {a_change_pct:+.1f}% (目标: <-5%)")
            print(f"   气味 B MBON 变化: {b_change_pct:+.1f}% (目标: >-3%)")
            print("   建议: 增加训练次数、提高学习率、或延长气味-电击重叠时间")
        print("=" * 70)

        return learned

    def run(self):
        """运行完整实验"""
        print("\n" + "█" * 70)
        print("█" + " " * 15 + "果蝇蘑菇体嗅觉条件反射实验" + " " * 20 + "█")
        print("█" + " " * 10 + "Olfactory Conditioning in Mushroom Body" + " " * 17 + "█")
        print("█" * 70)
        print(f"\n实验参数:")
        print(f"  训练次数: {self.n_train}")
        print(f"  气味呈现: {self.odor_duration_ms}ms")
        print(f"  电击持续: {self.shock_duration_ms}ms")
        print(f"  试次间隔: {self.inter_trial_ms}ms")
        print(f"  神经元: {self.sim.n} (KC={len(self.sim.kc_idx)}, "
              f"ALPN={len(self.sim.alpn_idx)}, DAN={len(self.sim.dan_idx)}, "
              f"MBON={len(self.sim.mbon_idx)})")

        t0 = time.time()

        # 1. 训练前测试
        self.run_pre_train_test()

        # 2. 训练
        self.run_training()

        # 3. 训练后测试
        learned = self.run_post_train_test()

        elapsed = time.time() - t0
        print(f"\n实验总耗时: {elapsed:.1f}s")
        print(f"仿真时间: {(self.sim.time_ms / 1000):.1f}s")
        print(f"实时速度比: {(self.sim.time_ms / 1000) / elapsed:.1f}x")

        # 保存结果
        result_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "conditioning_results.json"
        )
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存: {result_path}")

        return learned


def main():
    n_train = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    circuit_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "mushroom_body", "circuit.json"
    )

    exp = ConditioningExperiment(circuit_path, n_train=n_train)
    exp.run()


if __name__ == "__main__":
    main()
