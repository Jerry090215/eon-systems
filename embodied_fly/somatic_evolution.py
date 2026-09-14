#!/usr/bin/env python3
"""
体细胞进化引擎 (Somatic Evolution Engine)

单果蝇自主进化的核心引擎：
- 每 MUTATION_INTERVAL 秒触发一次体细胞突变
- 突变前保存当前基因型快照
- 突变后运行评估周期，收集适应度数据
- 评估适应度，决定保留还是回退
- 记录完整进化日志

这不是达尔文式的种群进化，而是个体内部的体细胞进化
——对应真实生物大脑的终身神经可塑性。
"""

import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from genome import PlasticGenome, FitnessMetrics


@dataclass
class MutationRecord:
    """单次突变记录"""
    generation: int
    timestamp: float
    mutations: Dict[str, Tuple[float, float]]  # {param: (old, new)}
    fitness_before: float
    fitness_after: float
    accepted: bool
    personality_before: str
    personality_after: str
    metrics_before: Dict
    metrics_after: Dict


@dataclass
class EvolutionState:
    """进化引擎状态"""
    phase: str = "baseline"  # baseline / mutation_running / evaluating
    phase_start_time: float = 0.0
    current_mutations: Optional[Dict[str, Tuple[float, float]]] = None
    snapshot_genome: Optional[PlasticGenome] = None
    baseline_fitness: float = 0.0
    mutation_count: int = 0
    last_mutation_time: float = 0.0


class SomaticEvolutionEngine:
    """
    体细胞进化引擎

    工作流程：
    1. 基线期（BASELINE_DURATION秒）：收集初始适应度数据
    2. 突变期：发生一次随机突变，立即应用
    3. 评估期（EVALUATION_DURATION秒）：收集突变后的适应度数据
    4. 决策：突变后适应度更高？→ 保留；更低？→ 回退
    5. 回到步骤2，循环

    这样单个果蝇在生命周期内不断进化，
    好的突变被保留，差的被回退，
    几小时后它的性格会和刚开始完全不同。
    """

    # 时间参数（秒）
    BASELINE_DURATION = 120.0      # 基线期 2 分钟（收集初始数据）
    MUTATION_INTERVAL = 300.0      # 突变间隔 5 分钟
    EVALUATION_DURATION = 240.0    # 评估期 4 分钟（突变后收集数据）
    MUTATION_TEMPERATURE = 1.0     # 突变温度（1.0=正常）

    # 适应度阈值：突变后适应度至少提升这么多才保留
    # 设为0意味着只要不变差就保留（中性漂变也允许）
    FITNESS_IMPROVEMENT_THRESHOLD = 0.0

    def __init__(self, genome: Optional[PlasticGenome] = None,
                 log_path: Optional[str] = None):
        """
        初始化进化引擎

        Args:
            genome: 初始基因型，None=创建默认基因型
            log_path: 进化日志文件路径
        """
        self.genome = genome if genome else PlasticGenome()
        self.state = EvolutionState()
        self.state.phase_start_time = time.monotonic()

        # 适应度指标
        self.baseline_metrics = FitnessMetrics()
        self.evaluation_metrics = FitnessMetrics()
        self.baseline_fitness = 0.0  # 基线适应度
        self._metrics_sample_count = 0
        self._metrics_energy_sum = 0.0
        self._metrics_arousal_sum = 0.0
        self._metrics_rest_time = 0.0
        self._metrics_total_time = 0.0
        self._position_history = []  # 用于计算探索范围

        # 进化日志
        self.log_path = log_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "evolution_log.json"
        )
        self.mutation_history: List[MutationRecord] = []

        # 回调函数（突变被接受/拒绝时调用）
        self.on_mutation_accepted = None
        self.on_mutation_rejected = None

        print(f"[SomaticEvolution] 引擎初始化完成")
        print(f"[SomaticEvolution] 突变间隔: {self.MUTATION_INTERVAL}秒")
        print(f"[SomaticEvolution] 评估期: {self.EVALUATION_DURATION}秒")
        print(f"[SomaticEvolution] 初始性格: {self.genome.get_personality_description()}")

    def update(self, dt: float, energy: float, arousal: float,
               punishment: bool = False, punishment_intensity: float = 0.0,
               on_food: bool = False, is_colliding: bool = False,
               is_resting: bool = False, position: Tuple[float, float] = (0, 0)):
        """
        每帧更新进化引擎

        Args:
            dt: 时间步长（秒）
            energy: 当前能量水平
            arousal: 当前唤醒度
            punishment: 是否受到惩罚
            punishment_intensity: 惩罚强度
            on_food: 是否在食物源上
            is_colliding: 是否在碰撞
            is_resting: 是否在休息
            position: 当前位置 (x, y)
        """
        now = time.monotonic()
        elapsed = now - self.state.phase_start_time

        # 收集当前周期的适应度数据
        self._collect_metrics(dt, energy, arousal, punishment,
                               punishment_intensity, on_food, is_colliding,
                               is_resting, position)

        if self.state.phase == "baseline":
            # 基线期：收集初始数据
            if elapsed >= self.BASELINE_DURATION:
                self._finish_baseline()

        elif self.state.phase == "mutation_running":
            # 突变已发生，正在运行评估期
            if elapsed >= self.EVALUATION_DURATION:
                self._evaluate_mutation()

        elif self.state.phase == "cooldown":
            # 冷却期（突变决策后，等待下一次突变）
            if elapsed >= self.MUTATION_INTERVAL:
                self._trigger_mutation()

    def _collect_metrics(self, dt, energy, arousal, punishment,
                          punishment_intensity, on_food, is_colliding,
                          is_resting, position):
        """收集适应度指标数据"""
        metrics = self._get_current_metrics()

        self._metrics_sample_count += 1
        self._metrics_energy_sum += energy
        self._metrics_arousal_sum += arousal
        self._metrics_total_time += dt

        if is_resting:
            self._metrics_rest_time += dt

        if punishment:
            metrics.punishment_count += 1
            metrics.punishment_intensity += punishment_intensity

        if on_food:
            metrics.food_visits += 1  # 简化：每帧在食物上就算一次

        if is_colliding:
            metrics.collision_count += 1

        # 记录位置历史（用于计算探索范围）
        self._position_history.append(position)
        if len(self._position_history) > 1000:
            self._position_history.pop(0)

    def _get_current_metrics(self) -> FitnessMetrics:
        """获取当前周期的适应度指标"""
        if self.state.phase == "baseline":
            return self.baseline_metrics
        else:
            return self.evaluation_metrics

    def _finish_baseline(self):
        """完成基线期，计算基线适应度"""
        self._finalize_metrics(self.baseline_metrics)
        self.baseline_fitness = self.baseline_metrics.compute_fitness()

        print(f"\n[SomaticEvolution] === 基线期完成 ===")
        print(f"  基线适应度: {self.baseline_fitness:.3f}")
        print(f"  平均能量: {self.baseline_metrics.energy_level:.3f}")
        print(f"  惩罚次数: {self.baseline_metrics.punishment_count}")
        print(f"  性格: {self.genome.get_personality_description()}")

        # 进入冷却期，等待第一次突变
        self.state.phase = "cooldown"
        self.state.phase_start_time = time.monotonic()

    def _trigger_mutation(self):
        """触发一次体细胞突变"""
        # 保存当前基因型快照
        self.state.snapshot_genome = self.genome.clone()
        self.state.last_mutation_time = time.monotonic()

        # 发生突变
        mutations = self.genome.mutate(temperature=self.MUTATION_TEMPERATURE)
        self.genome.apply_mutations(mutations)
        self.state.current_mutations = mutations

        # 重置评估指标
        self.evaluation_metrics = FitnessMetrics()
        self._reset_metrics_accumulators()

        # 进入突变运行期
        self.state.phase = "mutation_running"
        self.state.phase_start_time = time.monotonic()
        self.state.mutation_count += 1
        self.genome.generation += 1

        print(f"\n[SomaticEvolution] === 第 {self.state.mutation_count} 次突变 ===")
        print(f"  突变参数: {len(mutations)} 个")
        for name, (old, new) in mutations.items():
            delta = new - old
            print(f"    {name}: {old:.4f} → {new:.4f} (Δ={delta:+.4f})")
        print(f"  新性格: {self.genome.get_personality_description()}")
        print(f"  评估期开始 ({self.EVALUATION_DURATION}秒)...")

    def _evaluate_mutation(self):
        """评估突变效果，决定保留还是回退"""
        self._finalize_metrics(self.evaluation_metrics)
        new_fitness = self.evaluation_metrics.compute_fitness()

        # 计算适应度变化
        fitness_delta = new_fitness - self.baseline_fitness
        improvement = fitness_delta > self.FITNESS_IMPROVEMENT_THRESHOLD

        # 记录突变历史
        record = MutationRecord(
            generation=self.genome.generation,
            timestamp=time.monotonic(),
            mutations=self.state.current_mutations or {},
            fitness_before=self.baseline_fitness,
            fitness_after=new_fitness,
            accepted=improvement,
            personality_before=self.state.snapshot_genome.get_personality_description() if self.state.snapshot_genome else "",
            personality_after=self.genome.get_personality_description(),
            metrics_before=self.baseline_metrics.__dict__.copy(),
            metrics_after=self.evaluation_metrics.__dict__.copy(),
        )
        self.mutation_history.append(record)

        if improvement:
            # 保留突变
            self.genome.accepted_mutations += 1
            self.baseline_fitness = new_fitness  # 更新基线
            self.baseline_metrics = self.evaluation_metrics  # 更新基线指标

            print(f"\n[SomaticEvolution] === 突变被保留 ✓ ===")
            print(f"  适应度: {self.baseline_fitness:.3f} → {new_fitness:.3f} (Δ={fitness_delta:+.3f})")
            print(f"  性格: {record.personality_before} → {record.personality_after}")
            print(f"  总突变: {self.genome.total_mutations}, 接受: {self.genome.accepted_mutations}, 拒绝: {self.genome.rejected_mutations}")

            if self.on_mutation_accepted:
                self.on_mutation_accepted(record)
        else:
            # 回退突变
            self.genome.rejected_mutations += 1
            if self.state.snapshot_genome:
                # 恢复快照的参数值
                for name, p in self.state.snapshot_genome.params.items():
                    self.genome.params[name].value = p.value

            print(f"\n[SomaticEvolution] === 突变被回退 ✗ ===")
            print(f"  适应度: {self.baseline_fitness:.3f} → {new_fitness:.3f} (Δ={fitness_delta:+.3f})")
            print(f"  性格恢复为: {self.genome.get_personality_description()}")
            print(f"  总突变: {self.genome.total_mutations}, 接受: {self.genome.accepted_mutations}, 拒绝: {self.genome.rejected_mutations}")

            if self.on_mutation_rejected:
                self.on_mutation_rejected(record)

        # 保存进化日志
        self._save_log()

        # 保存基因型
        genome_path = self.log_path.replace("evolution_log", "current_genome")
        self.genome.save(genome_path)

        # 进入冷却期，等待下一次突变
        self.state.phase = "cooldown"
        self.state.phase_start_time = time.monotonic()
        self.state.current_mutations = None
        self.state.snapshot_genome = None

    def _finalize_metrics(self, metrics: FitnessMetrics):
        """计算最终的适应度指标"""
        if self._metrics_sample_count > 0:
            metrics.energy_level = self._metrics_energy_sum / self._metrics_sample_count
            metrics.avg_arousal = self._metrics_arousal_sum / self._metrics_sample_count

        if self._metrics_total_time > 0:
            metrics.rest_ratio = self._metrics_rest_time / self._metrics_total_time

        # 计算探索范围（位置历史的标准差）
        if len(self._position_history) > 10:
            xs = [p[0] for p in self._position_history]
            ys = [p[1] for p in self._position_history]
            x_range = max(xs) - min(xs)
            y_range = max(ys) - min(ys)
            metrics.exploration_range = x_range * y_range

    def _reset_metrics_accumulators(self):
        """重置指标累加器"""
        self._metrics_sample_count = 0
        self._metrics_energy_sum = 0.0
        self._metrics_arousal_sum = 0.0
        self._metrics_rest_time = 0.0
        self._metrics_total_time = 0.0
        self._position_history = []

    def _save_log(self):
        """保存进化日志到文件"""
        log_data = {
            "total_mutations": self.genome.total_mutations,
            "accepted_mutations": self.genome.accepted_mutations,
            "rejected_mutations": self.genome.rejected_mutations,
            "acceptance_rate": self.genome.accepted_mutations / max(1, self.genome.total_mutations),
            "current_generation": self.genome.generation,
            "current_personality": self.genome.get_personality_description(),
            "current_fitness": self.baseline_fitness,
            "genome_summary": self.genome.get_summary(),
            "mutation_history": [
                {
                    "generation": r.generation,
                    "timestamp": r.timestamp,
                    "mutations": {k: [v[0], v[1]] for k, v in r.mutations.items()},
                    "fitness_before": r.fitness_before,
                    "fitness_after": r.fitness_after,
                    "accepted": r.accepted,
                    "personality_before": r.personality_before,
                    "personality_after": r.personality_after,
                }
                for r in self.mutation_history[-50:]  # 只保留最近50条
            ],
        }
        try:
            with open(self.log_path, 'w') as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[SomaticEvolution] 保存日志失败: {e}")

    def get_status(self) -> Dict:
        """获取当前进化状态（用于显示）"""
        now = time.monotonic()
        elapsed = now - self.state.phase_start_time

        if self.state.phase == "baseline":
            phase_name = "基线期"
            phase_progress = elapsed / self.BASELINE_DURATION
            next_event = f"基线完成还需 {max(0, self.BASELINE_DURATION - elapsed):.0f}秒"
        elif self.state.phase == "mutation_running":
            phase_name = "突变评估期"
            phase_progress = elapsed / self.EVALUATION_DURATION
            next_event = f"评估完成还需 {max(0, self.EVALUATION_DURATION - elapsed):.0f}秒"
        elif self.state.phase == "cooldown":
            phase_name = "冷却期"
            phase_progress = elapsed / self.MUTATION_INTERVAL
            next_event = f"下次突变还需 {max(0, self.MUTATION_INTERVAL - elapsed):.0f}秒"
        else:
            phase_name = self.state.phase
            phase_progress = 0
            next_event = ""

        return {
            "phase": phase_name,
            "phase_progress": min(1.0, phase_progress),
            "next_event": next_event,
            "generation": self.genome.generation,
            "total_mutations": self.genome.total_mutations,
            "accepted_mutations": self.genome.accepted_mutations,
            "rejected_mutations": self.genome.rejected_mutations,
            "acceptance_rate": self.genome.accepted_mutations / max(1, self.genome.total_mutations),
            "current_fitness": self.baseline_fitness,
            "personality": self.genome.get_personality_description(),
            "age_seconds": now - self.genome.birth_time,
        }

    def force_mutation(self):
        """强制立即触发一次突变（跳过冷却期）"""
        if self.state.phase in ["cooldown", "baseline"]:
            self._trigger_mutation()
            print("[SomaticEvolution] 强制突变已触发")

    def reset(self):
        """重置进化引擎（恢复默认基因型，清空历史）"""
        self.genome = PlasticGenome()
        self.state = EvolutionState()
        self.state.phase_start_time = time.monotonic()
        self.baseline_metrics = FitnessMetrics()
        self.evaluation_metrics = FitnessMetrics()
        self.mutation_history = []
        self._reset_metrics_accumulators()
        print("[SomaticEvolution] 引擎已重置")


def test_evolution_engine():
    """测试进化引擎"""
    print("=" * 60)
    print("体细胞进化引擎测试")
    print("=" * 60)

    engine = SomaticEvolutionEngine()

    # 模拟运行（加速时间）
    print("\n--- 模拟运行 10 分钟（加速）---")
    dt = 0.1  # 100ms per frame
    total_time = 600  # 10 minutes
    steps = int(total_time / dt)

    for i in range(steps):
        # 模拟一些随机的生存数据
        energy = 0.5 + 0.3 * (i % 200) / 200  # 能量波动
        arousal = 0.3 + 0.2 * (i % 150) / 150
        punishment = (i % 300) == 0  # 偶尔被惩罚
        on_food = (i % 250) < 50  # 偶尔在食物上
        is_colliding = (i % 400) == 0  # 偶尔碰撞
        is_resting = (i % 100) < 20  # 偶尔休息
        position = (100 * (i % 50), 100 * ((i // 50) % 50))

        engine.update(dt, energy, arousal, punishment, 0.5,
                       on_food, is_colliding, is_resting, position)

        # 每60秒打印一次状态
        if i % int(60 / dt) == 0:
            status = engine.get_status()
            print(f"\n  [t={i*dt:.0f}s] {status['phase']} ({status['phase_progress']*100:.0f}%)")
            print(f"    代数: {status['generation']}, 突变: {status['total_mutations']}")
            print(f"    接受: {status['accepted_mutations']}, 拒绝: {status['rejected_mutations']}")
            print(f"    性格: {status['personality']}")
            print(f"    适应度: {status['current_fitness']:.3f}")

    print("\n" + "=" * 60)
    print("测试完成！")
    print(f"  总突变: {engine.genome.total_mutations}")
    print(f"  接受: {engine.genome.accepted_mutations}")
    print(f"  拒绝: {engine.genome.rejected_mutations}")
    print(f"  最终性格: {engine.genome.get_personality_description()}")
    print("=" * 60)


if __name__ == "__main__":
    test_evolution_engine()
