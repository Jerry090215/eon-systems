#!/usr/bin/env python3
"""
双果蝇管理系统 - 雌雄双蝇仿真与求偶交互

管理两只独立的果蝇（雌蝇 + 雄蝇），每只都有完整的蘑菇体大脑，
通过求偶协调器实现信息素感知、求偶之歌、交配等交互。

架构：
  - 雌蝇：FlyWire 真实连接组（雌蝇大脑），pC1 接纳中枢
  - 雄蝇：相同连接组（简化雄蝇），P1 求偶中枢
  - 求偶协调器：信息素传播、求偶状态机、交配检测
  - 双果蝇输出：两个大脑信号文件，DesktopFly 显示两只果蝇
"""

import os
import sys
import time
import json
import math
import tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from embodied_fly_live import EmbodiedFly
from courtship import (
    Gender, CourtshipCoordinator, CourtshipPhase, FemaleResponse
)
from genetics import GeneticsEngine, EvolutionTracker, FertilizedEgg


# 双果蝇大脑信号文件路径
BRAIN_SIGNAL_FEMALE = "/tmp/fly_brain_female.json"
BRAIN_SIGNAL_MALE = "/tmp/fly_brain_male.json"

# 受精卵保存目录
EGG_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eggs")

# 最大种群数量（防止无限繁衍导致内存溢出）
MAX_POPULATION = 10


class FlyEntity:
    """
    单只果蝇实体（位置 + 朝向 + 大脑）

    封装 EmbodiedFly，添加位置和朝向管理，以及求偶行为调制。
    """

    def __init__(self, circuit_path: str, gender: Gender,
                 initial_pos: tuple, initial_heading: float = 0.0,
                 enable_learning: bool = True, fly_id: str = None,
                 generation: int = 0, mother_id: str = None, father_id: str = None):
        """
        Args:
            circuit_path: 蘑菇体回路 JSON 路径
            gender: 性别
            initial_pos: 初始位置 (x, y)
            initial_heading: 初始朝向（弧度）
            enable_learning: 是否启用 STDP 学习
            fly_id: 唯一ID（None则自动生成）
            generation: 代数（0=初代，1=初代的孩子）
            mother_id: 母亲ID
            father_id: 父亲ID
        """
        import uuid
        self.fly_id = fly_id or str(uuid.uuid4())[:8]
        self.generation = generation
        self.mother_id = mother_id
        self.father_id = father_id
        self.gender = gender
        self.pos = list(initial_pos)
        self.heading = initial_heading
        self.circuit_path = circuit_path
        self.enable_learning = enable_learning
        self.brain = EmbodiedFly(circuit_path, enable_learning=enable_learning,
                                  write_signals=False)

        # 运动参数
        self.max_speed = 3.0  # 像素/帧（最大移动速度）
        self.turn_rate = 0.15  # 弧度/帧（最大转向速度）

        # 最近的行为数据
        self.last_behavior = None
        self.last_activation = None
        self.last_sensor = None

        # 存活时间
        self.birth_time = time.monotonic()
        self.alive = True

    def step(self, dt_ms: int = 4):
        """
        运行一步大脑仿真（不更新位置，位置由 MultiFlySystem 统一管理）

        Returns:
            (sensor_data, activation, behavior)
        """
        result = self.brain.step(dt_ms)
        self.last_sensor, self.last_activation, self.last_behavior = result
        return result

    def apply_movement(self, walk_drive: float, turn_bias: float, dt_ms: int):
        """
        根据行为输出更新位置和朝向

        Args:
            walk_drive: 行走驱动（0-1.5）
            turn_bias: 转向偏置（-1到1）
            dt_ms: 时间步长（ms）
        """
        # 转向
        self.heading += turn_bias * self.turn_rate
        # 归一化朝向
        self.heading = (self.heading + math.pi) % (2 * math.pi) - math.pi

        # 移动
        speed = walk_drive * self.max_speed
        self.pos[0] += math.cos(self.heading) * speed
        self.pos[1] += math.sin(self.heading) * speed

        # 边界限制（假设桌面范围 -600 到 600）
        self.pos[0] = max(-580, min(580, self.pos[0]))
        self.pos[1] = max(-380, min(380, self.pos[1]))

    def get_position(self) -> tuple:
        return (self.pos[0], self.pos[1])

    def reset(self):
        """重置大脑"""
        self.brain = EmbodiedFly(
            self.circuit_path,
            enable_learning=self.enable_learning,
            write_signals=False
        )


class MultiFlySystem:
    """
    多果蝇系统 - 管理种群的仿真、求偶与繁衍

    核心职责：
      1. 创建并管理动态果蝇种群（初代雌蝇+雄蝇，后代通过繁衍出生）
      2. 每帧更新所有果蝇的感官输入和蘑菇体仿真
      3. 求偶协调器计算信息素感知和求偶状态
      4. 交配完成后通过遗传算法生成受精卵
      5. 受精卵孵化后出生新果蝇（加入种群）
      6. 输出所有果蝇的大脑信号到共享文件
    """

    def __init__(self, circuit_path: str, enable_learning: bool = True,
                 incubation_period: float = 60.0):
        """
        Args:
            circuit_path: 蘑菇体回路 JSON 路径
            enable_learning: 是否启用 STDP 学习
            incubation_period: 受精卵孵化期（秒，默认60秒）
        """
        print("[MultiFly] 初始化多果蝇繁衍系统...")

        self.circuit_path = circuit_path
        self.enable_learning = enable_learning

        # 遗传算法引擎
        self.genetics = GeneticsEngine(
            mutation_rate=0.015,
            mutation_std=0.1,
            incubation_period=incubation_period,
        )

        # 进化追踪器
        self.evolution = EvolutionTracker()

        # 果蝇种群（fly_id -> FlyEntity）
        self.flies = {}

        # 未孵化的受精卵
        self.eggs = []

        # 创建初代雌蝇（右侧）和雄蝇（左侧）
        self.female = FlyEntity(
            circuit_path, Gender.FEMALE,
            initial_pos=(150, 0), initial_heading=math.pi,
            enable_learning=enable_learning,
            fly_id="founder_f", generation=0,
        )
        self.male = FlyEntity(
            circuit_path, Gender.MALE,
            initial_pos=(-150, 0), initial_heading=0.0,
            enable_learning=enable_learning,
            fly_id="founder_m", generation=0,
        )
        self.flies[self.female.fly_id] = self.female
        self.flies[self.male.fly_id] = self.male

        # 求偶协调器（只处理初代雌雄蝇的求偶）
        self.courtship = CourtshipCoordinator()
        self.courtship.on_mating_complete = self._on_mating_complete

        # 状态
        self.running = False
        self.frame_count = 0
        self.start_time = None
        self.mating_count = 0
        self.birth_count = 0

        # 创建受精卵目录
        os.makedirs(EGG_DIRECTORY, exist_ok=True)

        print(f"[MultiFly] 初始化完成: 初代雌蝇@{self.female.fly_id} + 雄蝇@{self.male.fly_id}")
        print(f"[MultiFly] 孵化期: {incubation_period:.0f}秒, 变异率: {self.genetics.mutation_rate*100:.1f}%")

    # ============================================================
    # 繁衍系统：交配回调、受精卵管理、孵化、出生
    # ============================================================

    def _get_fly_weights(self, fly: FlyEntity) -> np.ndarray:
        """获取果蝇的蘑菇体KC→MBON突触权重"""
        return fly.brain.sim.kc_mbon_weight.copy()

    def _on_mating_complete(self):
        """
        交配完成回调 - 生成受精卵

        流程：
          1. 获取父母的KC→MBON突触权重
          2. 遗传算法交叉互换 + 随机变异
          3. 创建受精卵，随机分配性别
          4. 保存到文件
          5. 记录到进化追踪器
        """
        self.mating_count += 1
        print(f"\n[MultiFly] 💑 交配完成！(第{self.mating_count}次)")

        # 检查种群数量上限
        alive_count = sum(1 for f in self.flies.values() if f.alive)
        if alive_count >= MAX_POPULATION:
            print(f"[MultiFly] 种群已达上限({MAX_POPULATION})，跳过繁衍")
            return

        # 获取父母权重
        try:
            mother_weights = self._get_fly_weights(self.female)
            father_weights = self._get_fly_weights(self.male)
        except Exception as e:
            print(f"[MultiFly] 获取父母权重失败: {e}")
            return

        # 获取母亲对父亲的最终评估分
        mother_eval = self.courtship.female_pc1.state.evaluation_score
        father_song_freq = self.courtship.male_p1.state.song_frequency

        # 出生位置（在父母中间附近）
        birth_x = (self.female.pos[0] + self.male.pos[0]) / 2
        birth_y = (self.female.pos[1] + self.male.pos[1]) / 2 + np.random.uniform(-20, 20)

        # 创建受精卵
        egg = self.genetics.create_egg(
            weights_mother=mother_weights,
            weights_father=father_weights,
            mother_id=self.female.fly_id,
            father_id=self.male.fly_id,
            birth_position=(birth_x, birth_y),
            mother_eval=mother_eval,
            father_song_freq=father_song_freq,
            generation=1,
        )

        # 保存到文件
        try:
            egg_path = self.genetics.save_egg(egg, EGG_DIRECTORY)
            print(f"[MultiFly] 🥚 受精卵已生成: id={egg.egg_id}, "
                  f"性别={egg.gender.value}, 孵化期={egg.incubation_period:.0f}秒")
            print(f"[MultiFly]    保存到: {egg_path}")
        except Exception as e:
            print(f"[MultiFly] 保存受精卵失败: {e}")

        # 加入未孵化列表
        self.eggs.append(egg)

        # 记录到进化追踪器
        self.evolution.record_mating(self.female.fly_id, self.male.fly_id, egg.egg_id)

    def _check_eggs(self):
        """检查所有未孵化的受精卵，孵化到期的卵"""
        now = time.monotonic()
        hatched = []

        for egg in self.eggs:
            if egg.is_ready_to_hatch(now):
                hatched.append(egg)

        for egg in hatched:
            self._hatch_egg(egg)
            self.eggs.remove(egg)

    def _hatch_egg(self, egg: FertilizedEgg):
        """
        孵化受精卵 - 创建新果蝇并加入种群

        Args:
            egg: 要孵化的受精卵
        """
        print(f"\n[MultiFly] 🐣 受精卵孵化！id={egg.egg_id}, "
              f"性别={egg.gender.value}, 代数={egg.generation}")

        # 创建新果蝇
        new_fly = FlyEntity(
            circuit_path=self.circuit_path,
            gender=egg.gender,
            initial_pos=egg.birth_position,
            initial_heading=np.random.uniform(-math.pi, math.pi),
            enable_learning=self.enable_learning,
            fly_id=f"gen{egg.generation}_{egg.egg_id}",
            generation=egg.generation,
            mother_id=egg.mother_id,
            father_id=egg.father_id,
        )

        # 加载遗传的突触权重
        try:
            sim = new_fly.brain.sim
            if len(egg.weights) == len(sim.kc_mbon_weight):
                sim.kc_mbon_weight[:] = egg.weights.astype(np.float32)
                # 同步更新W矩阵中的KC→MBON权重
                if hasattr(sim, 'kc_mbon_data_idx'):
                    for i in range(len(sim.kc_mbon_weight)):
                        data_idx = sim.kc_mbon_data_idx[i]
                        if data_idx >= 0:
                            sim.W.data[data_idx] = sim.kc_mbon_weight[i]
                print(f"[MultiFly]    已加载遗传权重: mean={egg.weights.mean():.6f}, "
                      f"std={egg.weights.std():.6f}")
            else:
                print(f"[MultiFly]    权重长度不匹配(文件={len(egg.weights)}, "
                      f"当前={len(sim.kc_mbon_weight)})，使用默认权重")
        except Exception as e:
            print(f"[MultiFly]    加载遗传权重失败(使用默认权重): {e}")

        # 加入种群
        self.flies[new_fly.fly_id] = new_fly
        self.birth_count += 1

        # 标记受精卵已孵化
        egg.hatched = True

        # 记录到进化追踪器
        self.evolution.record_birth(egg)

        print(f"[MultiFly]    新果蝇已加入种群: id={new_fly.fly_id}, "
              f"当前种群数量={len(self.flies)}")

    def step(self, dt_ms: int = 4):
        """
        运行一步双果蝇仿真

        Args:
            dt_ms: 仿真时间步长（ms）
        """
        # 1. 两只果蝇各自的大脑仿真（共享全局感官输入）
        female_sensor, female_act, female_beh = self.female.step(dt_ms)
        male_sensor, male_act, male_beh = self.male.step(dt_ms)

        # 2. 求偶协调（根据位置计算信息素和求偶状态）
        self.courtship.update(
            dt=dt_ms / 1000.0,
            male_pos=self.male.get_position(),
            female_pos=self.female.get_position(),
            male_heading=self.male.heading,
            female_heading=self.female.heading,
        )

        # 3. 获取求偶行为调制
        male_mods = self.courtship.get_male_modifiers()
        female_mods = self.courtship.get_female_modifiers()

        # 4. 计算最终行为输出（基础行为 + 求偶调制）
        male_walk, male_turn = self._compute_final_movement(
            male_beh, male_mods, Gender.MALE
        )
        female_walk, female_turn = self._compute_final_movement(
            female_beh, female_mods, Gender.FEMALE
        )

        # 5. 更新位置
        self.male.apply_movement(male_walk, male_turn, dt_ms)
        self.female.apply_movement(female_walk, female_turn, dt_ms)

        # 5.5 更新后代果蝇（简单的探索行为，不求偶）
        for fly_id, fly in self.flies.items():
            if fly_id in (self.female.fly_id, self.male.fly_id):
                continue  # 初代已更新
            if not fly.alive:
                continue
            # 后代果蝇：运行大脑仿真，用默认探索行为
            fly.step(dt_ms)
            base_walk = 0.25  # 后代默认缓慢探索
            base_turn = math.sin(time.monotonic() * 0.5 + hash(fly_id) % 10) * 0.3
            fly.apply_movement(base_walk, base_turn, dt_ms)

        # 6. 检查受精卵孵化
        self._check_eggs()

        # 7. 输出大脑信号（初代 + 所有后代）
        self._write_all_brain_signals(
            female_beh, male_beh,
            female_mods, male_mods,
            female_act, male_act
        )

        self.frame_count += 1
        return female_beh, male_beh

    def _compute_final_movement(self, behavior: dict, courtship_mods: dict,
                                 gender: Gender) -> tuple:
        """
        计算最终运动参数（基础行为 + 求偶调制）

        Args:
            behavior: 蘑菇体行为输出
            courtship_mods: 求偶行为调制
            gender: 性别

        Returns:
            (walk_drive, turn_bias)
        """
        # 基础行为（从 brain_signals 中提取，或用默认值）
        base_walk = 0.3  # 默认缓慢探索
        base_turn = 0.0

        if behavior:
            # 从行为数据中提取 walk 和 turn（如果有的话）
            base_walk = behavior.get('walk_drive', 0.3)
            base_turn = behavior.get('turn_bias', 0.0)

        # 求偶调制（求偶状态下，求偶行为主导）
        is_courting = courtship_mods.get('courtship_phase', 'idle') not in ('idle', 'post_mating')
        is_mating = courtship_mods.get('is_mating', False)

        if is_mating:
            # 交配中：完全不动
            return 0.0, 0.0
        elif is_courting:
            # 求偶中：求偶行为主导（70%求偶 + 30%基础）
            walk_mod = courtship_mods.get('walk_drive_mod', 1.0)
            turn_mod = courtship_mods.get('turn_bias_mod', 0.0)
            final_walk = base_walk * 0.3 + walk_mod * 0.7
            final_turn = base_turn * 0.3 + turn_mod * 0.7
            return final_walk, final_turn
        else:
            # 非求偶状态：基础行为主导
            return base_walk, base_turn

    def _write_all_brain_signals(self, female_beh: dict, male_beh: dict,
                                    female_mods: dict, male_mods: dict,
                                    female_act, male_act):
        """
        写入所有果蝇的大脑信号到共享文件

        初代雌雄蝇用固定文件名，后代用 fly_id 命名。
        """
        # 初代雌蝇
        female_signals = self._build_brain_signal(
            self.female, female_beh, female_mods, female_act, Gender.FEMALE
        )
        self._atomic_write(BRAIN_SIGNAL_FEMALE, female_signals)

        # 初代雄蝇
        male_signals = self._build_brain_signal(
            self.male, male_beh, male_mods, male_act, Gender.MALE
        )
        self._atomic_write(BRAIN_SIGNAL_MALE, male_signals)

        # 后代果蝇
        first_offspring_written = False
        for fly_id, fly in self.flies.items():
            if fly_id in (self.female.fly_id, self.male.fly_id):
                continue
            if not fly.alive:
                continue
            # 后代果蝇：简单的大脑信号
            offspring_signals = {
                "gender": fly.gender.value,
                "position": {"x": fly.pos[0], "y": fly.pos[1]},
                "heading": fly.heading,
                "walkDrive": 0.25,
                "turnBias": 0.0,
                "escape": False,
                "wingDrive": 0.0,
                "arousal": 0.3,
                "courtship_phase": "idle",
                "response": "unaware",
                "is_mating": False,
                "is_receptive": False,
                "song_frequency": 0,
                "song_intensity": 0,
                "kc_active": fly.last_behavior.get('kc_active', 0) if fly.last_behavior else 0,
                "mbon_rate": fly.last_behavior.get('mbon_avg_rate', 0) if fly.last_behavior else 0,
                "odor": fly.last_activation.odor_label if fly.last_activation else "none",
                "fly_id": fly_id,
                "generation": fly.generation,
                "mother_id": fly.mother_id,
                "father_id": fly.father_id,
                "timestamp": time.time(),
                "source": "multi_fly_offspring",
            }
            # 第一个后代写入固定文件（DesktopFly 读取）
            if not first_offspring_written:
                self._atomic_write("/tmp/fly_brain_offspring.json", offspring_signals)
                first_offspring_written = True
            # 所有后代都写入各自的文件
            path = f"/tmp/fly_brain_{fly_id}.json"
            self._atomic_write(path, offspring_signals)

    def _write_brain_signals(self, female_beh: dict, male_beh: dict,
                               female_mods: dict, male_mods: dict,
                               female_act, male_act):
        """
        写入双果蝇大脑信号到共享文件

        Args:
            female_beh: 雌蝇行为数据
            male_beh: 雄蝇行为数据
            female_mods: 雌蝇求偶调制
            male_mods: 雄蝇求偶调制
            female_act: 雌蝇神经激活
            male_act: 雄蝇神经激活
        """
        # 雌蝇信号
        female_signals = self._build_brain_signal(
            self.female, female_beh, female_mods, female_act, Gender.FEMALE
        )
        self._atomic_write(BRAIN_SIGNAL_FEMALE, female_signals)

        # 雄蝇信号
        male_signals = self._build_brain_signal(
            self.male, male_beh, male_mods, male_act, Gender.MALE
        )
        self._atomic_write(BRAIN_SIGNAL_MALE, male_signals)

    def _build_brain_signal(self, fly: FlyEntity, behavior: dict,
                             courtship_mods: dict, activation, gender: Gender) -> dict:
        """构建单只果蝇的大脑信号"""
        walk, turn = self._compute_final_movement(behavior, courtship_mods, gender)

        return {
            "gender": gender.value,
            "position": {"x": fly.pos[0], "y": fly.pos[1]},
            "heading": fly.heading,
            "walkDrive": round(walk, 3),
            "turnBias": round(turn, 3),
            "escape": courtship_mods.get('is_mating', False),
            "wingDrive": round(courtship_mods.get('wing_drive_mod', 0.0), 3),
            "arousal": round(courtship_mods.get('p1_activation', courtship_mods.get('pc1_activation', 0.3)), 3),
            "courtship_phase": courtship_mods.get('courtship_phase', 'idle'),
            "response": courtship_mods.get('response', 'unaware'),
            "is_mating": courtship_mods.get('is_mating', False),
            "is_receptive": courtship_mods.get('is_receptive', False),
            "song_frequency": courtship_mods.get('song_frequency', 0),
            "song_intensity": courtship_mods.get('song_intensity', 0),
            "kc_active": behavior.get('kc_active', 0) if behavior else 0,
            "mbon_rate": behavior.get('mbon_avg_rate', 0) if behavior else 0,
            "odor": activation.odor_label if activation else "none",
            "timestamp": time.time(),
            "source": "multi_fly",
        }

    def _atomic_write(self, path: str, data: dict):
        """原子写入 JSON 文件"""
        try:
            dir_name = os.path.dirname(path)
            fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
            with os.fdopen(fd, 'w') as f:
                json.dump(data, f)
            os.rename(tmp_path, path)
        except Exception:
            pass

    def get_status(self) -> dict:
        """获取系统状态摘要"""
        cs = self.courtship.get_status()
        now = time.monotonic()

        # 受精卵信息
        eggs_info = []
        for egg in self.eggs:
            eggs_info.append({
                "id": egg.egg_id,
                "gender": egg.gender.value,
                "progress": egg.get_progress(now),
                "remaining": egg.get_remaining_time(now),
                "generation": egg.generation,
            })

        # 种群信息
        alive_flies = [f for f in self.flies.values() if f.alive]
        offspring = [f for f in alive_flies if f.generation > 0]

        return {
            "frame": self.frame_count,
            "mating_count": self.mating_count,
            "birth_count": self.birth_count,
            "population": len(alive_flies),
            "offspring_count": len(offspring),
            "eggs_count": len(self.eggs),
            "eggs": eggs_info,
            "female_pos": self.female.get_position(),
            "male_pos": self.male.get_position(),
            "distance": cs['distance'],
            "male_phase": cs['male_phase'],
            "female_response": cs['female_response'],
            "male_p1": cs['male_p1'],
            "female_pc1": cs['female_pc1'],
            "female_eval": cs['female_eval'],
            "song_freq": cs['male_song_freq'],
            "song_intensity": cs['male_song_intensity'],
            "female_pref": cs['female_pref_freq'],
        }

    def print_status(self):
        """打印实时状态"""
        s = self.get_status()
        elapsed = time.monotonic() - self.start_time if self.start_time else 0

        # 求偶状态显示
        if s['male_phase'] == 'mating':
            courtship_str = f"❤️交配中(dist={s['distance']:.0f})"
        elif s['male_phase'] == 'singing':
            courtship_str = f"🎵唱歌@{s['song_freq']:.0f}Hz(雌pref={s['female_pref']:.0f}Hz,eval={s['female_eval']:.2f})"
        elif s['male_phase'] == 'chasing':
            courtship_str = f"🏃追逐(dist={s['distance']:.0f})"
        elif s['male_phase'] == 'post_mating':
            courtship_str = f"😌不应期(已交配{s['mating_count']}次)"
        elif s['male_phase'] == 'idle':
            courtship_str = f"○空闲(dist={s['distance']:.0f})"
        else:
            courtship_str = f"⏳{s['male_phase']}(dist={s['distance']:.0f})"

        status = (
            f"\r[{elapsed:6.1f}s] "
            f"雌@{s['female_pos'][0]:.0f},{s['female_pos'][1]:.0f} "
            f"雄@{s['male_pos'][0]:.0f},{s['male_pos'][1]:.0f} | "
            f"{courtship_str} | "
            f"P1={s['male_p1']:.2f} pC1={s['female_pc1']:.2f} | "
            f"种群={s['population']} 卵={s['eggs_count']} 出生={s['birth_count']}"
        )
        print(status, end='', flush=True)

    def run(self, duration_sec: float = None, print_interval: float = 0.1):
        """
        运行双果蝇系统

        Args:
            duration_sec: 运行时长（None = 无限）
            print_interval: 打印间隔（秒）
        """
        print("\n" + "=" * 80)
        print("🪰🪰 多果蝇繁衍系统 - 求偶 + 交配 + 遗传进化")
        print("=" * 80)
        print("\n操作说明:")
        print("  🖱️  移动鼠标 → 两只果蝇都能感知")
        print("  🔊 快速调大音量 → 惩罚（两只都能感知）")
        print("  💑 雄蝇会自动追逐、唱歌、求偶")
        print("  💕 如果雌蝇被打动，会接受交配")
        print("  🥚 交配完成后生成受精卵（遗传算法交叉互换+变异）")
        print("  🐣 孵化期后新果蝇出生（加入种群）")
        print("  Ctrl+C 退出\n")

        self.running = True
        self.start_time = time.monotonic()
        last_print = 0

        try:
            while self.running:
                self.step(dt_ms=4)

                now = time.monotonic()
                if now - last_print > print_interval:
                    self.print_status()
                    last_print = now

                if duration_sec and (now - self.start_time) > duration_sec:
                    break

                # 控制帧率（约 30fps）
                time.sleep(0.03)

        except KeyboardInterrupt:
            print("\n\n[MultiFly] 收到中断信号，正在保存并退出...")
        finally:
            self.running = False
            # 保存权重
            if self.female.brain.enable_learning:
                self.female.brain.sim.save_weights(
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fly_weights_female.npz')
                )
            if self.male.brain.enable_learning:
                self.male.brain.sim.save_weights(
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fly_weights_male.npz')
                )
            print(f"[MultiFly] 运行结束: {self.frame_count} 帧, 交配 {self.mating_count} 次")


def main():
    """主入口"""
    circuit_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', 'mushroom_body', 'circuit.json'
    )

    system = MultiFlySystem(circuit_path, enable_learning=True)
    system.run()


if __name__ == "__main__":
    main()
