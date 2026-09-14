#!/usr/bin/env python3
"""
桌面果蝇具身智能 - 实时交互系统

把桌面物理变量编码为蘑菇体神经激活，实现实时联想学习：
  - 鼠标移动模式 → 嗅觉（ALPN 激活）
  - 音量突变 → 电击/惩罚（PPL1 DAN 激活）
  - 蘑菇体 LIF 仿真 → STDP 学习 → MBON 输出 → 回避/趋向行为

用法:
  python3 embodied_fly_live.py
  - 移动鼠标改变气味（快速直线=气味A，缓慢画圈=气味B）
  - 快速调大音量 = 电击（惩罚）
  - 系统会实时学习：某种鼠标模式 + 电击 → 学会回避该模式
  - 按 Ctrl+C 退出
"""

import os
import sys
import time
import json
import math
import tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sensors import SensorHub
from encoder import NeuralEncoder, MushroomBodyCircuit
from mb_lif import MushroomBodyLIF
from central_complex import CentralComplex
from motor_control import MotorController
from exploration import ExplorationDriver
from thought_generator import ThoughtGenerator
from window_detector import WindowDetector
from desktop_environment import DesktopEnvironment
from metabolism import Metabolism
from real_senses import RealSensoryEncoder
from visual_lobe import VisualLobeLIF, VisualInput
from genome import PlasticGenome, FitnessMetrics
from somatic_evolution import SomaticEvolutionEngine
from intrinsic_plasticity import IntrinsicPlasticity, IntrinsicPlasticityConfig

# 共享大脑信号文件路径（DesktopFly 读取此文件获取外部大脑指令）
BRAIN_SIGNAL_PATH = "/tmp/fly_brain.json"

# 共享视觉/位置文件路径（Python 虚拟位置写入，DesktopFly 也可写入）
VISION_FILE = "/tmp/fly_vision.json"

# STDP 权重保存路径（学习结果持久化）
WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fly_weights.npz")

# 自动保存间隔（秒）
AUTOSAVE_INTERVAL = 30.0


class EmbodiedFly:
    """桌面果蝇具身智能主体"""

    def __init__(self, circuit_path: str, enable_learning: bool = True,
                 write_signals: bool = True):
        """
        Args:
            circuit_path: 蘑菇体回路 JSON 路径
            enable_learning: 是否启用 STDP 学习
            write_signals: 是否写入大脑信号到共享文件（多果蝇模式下设为False）
        """
        print("[EmbodiedFly] 初始化中...")
        self.write_signals = write_signals
        self.circuit_path = circuit_path

        # 加载蘑菇体回路（用于编码器的神经元分组）
        self.circuit = MushroomBodyCircuit(circuit_path)

        # 感官编码器
        self.encoder = NeuralEncoder(self.circuit)

        # 蘑菇体 LIF 仿真
        self.sim = MushroomBodyLIF(circuit_path, enable_stdp=enable_learning)

        # 中央复合体（导航中枢）
        self.cx = CentralComplex(grid_size=80.0, boundary_margin=120.0)

        # 运动控制器（行为精细化）
        self.motor = MotorController()

        # 探索驱动（内在动机系统 - 真实果蝇的本能）
        self.exploration = ExplorationDriver()

        # 内心独白生成器（基于神经状态，由本地 mimi 模型实时生成）
        # mimi在M2 Air上生成较慢（约3分钟/条），间隔设60秒
        self.thought_gen = ThoughtGenerator(model="mimi", interval=60.0)

        # === 桌面真实环境系统 ===
        # 窗口检测器（实时获取桌面所有窗口位置）
        self.window_detector = WindowDetector(refresh_interval=1.0)
        # 桌面环境（碰撞检测、食物源、鼠标捕食者）
        self.desktop_env = DesktopEnvironment()
        # 代谢系统（能量、疲劳、昼夜节律、死亡）
        self.metabolism = Metabolism()
        # 真实感官编码器（视觉/嗅觉/触觉 → 神经电流）
        self.real_encoder = RealSensoryEncoder(circuit=self.circuit)

        # 视叶 LIF 网络（基于 FlyWire 真实连接组的视觉处理）
        # 1,200 个神经元：髓质中间神经元 + T4/T5运动检测 + LC输出
        self.visual_lobe = VisualLobeLIF()
        self.enable_visual_lobe = True  # 视叶开关

        # === 体细胞进化引擎（单果蝇自主进化）===
        # 不繁衍后代，单个果蝇在生命周期内不断突变、试错、保留好的改变
        # 每5分钟一次突变，4分钟评估期，好的保留，差的回退
        self.genome = PlasticGenome()
        self.evolution = SomaticEvolutionEngine(genome=self.genome)
        self.enable_evolution = True  # 进化开关

        # 内在可塑性（神经元阈值/时间常数自调整）
        self.intrinsic_plasticity = IntrinsicPlasticity(
            n_neurons=self.sim.n,
            config=IntrinsicPlasticityConfig(
                target_firing_rate=5.0,
                threshold_adaptation_rate=0.001,
                tau_adaptation_rate=0.0005,
            )
        )
        self.enable_intrinsic_plasticity = True

        # 环境系统启用开关（可以通过配置关闭，回退到原来的鼠标/音量编码）
        self.enable_desktop_env = True

        # 虚拟果蝇位置跟踪器（Python端维护，不依赖DesktopFly的位置输出）
        # 坐标系统：DesktopFly 坐标（屏幕中心为原点，y向上）
        self._virt_fly_x = 0.0
        self._virt_fly_y = 0.0
        self._virt_fly_heading = 0.0
        self._virt_last_update = time.monotonic()

        # 感官 hub
        self.hub = SensorHub()

        # 状态
        self.running = False
        self.frame_count = 0
        self.start_time = None

        # 学习记录
        self.odor_history = []
        self.punishment_count = 0
        self.last_punishment_time = 0

        # 权重持久化
        self.enable_learning = enable_learning
        self.last_autosave_time = time.monotonic()
        if enable_learning and os.path.exists(WEIGHTS_PATH):
            print("[EmbodiedFly] 检测到已保存的学习权重，正在加载...")
            self.sim.load_weights(WEIGHTS_PATH)

        print(f"[EmbodiedFly] 初始化完成: {self.sim.n} 神经元, 学习={'开' if enable_learning else '关'}")

    def step(self, dt_ms: int = 4):
        """
        运行一步仿真（默认 4ms = 250fps 仿真速度，约 30fps 显示帧率）

        Args:
            dt_ms: 仿真时间步长（ms）
        """
        # 1. 读取感官输入
        sensor_data = self.hub.update()

        # 1.1 应用基因型参数（体细胞进化的突变会改变这些参数）
        if self.enable_evolution:
            self._apply_genome()

        # 1.5 桌面真实环境系统（窗口检测 + 碰撞 + 食物 + 捕食者）
        if self.enable_desktop_env:
            # 更新虚拟果蝇位置（根据上一帧的大脑信号推算）
            self._update_virtual_fly_position(dt_ms / 1000.0)

            # 更新窗口检测器（内部有刷新间隔控制）
            self.window_detector.update()
            # 更新桌面环境（读取果蝇位置、检测碰撞/食物/捕食者）
            self.desktop_env.update(self.window_detector, dt=dt_ms / 1000.0)

        # 2. 编码为神经激活
        activation = self.encoder.encode(sensor_data)

        # 2.5 真实感官编码叠加（桌面环境的视觉/嗅觉/触觉 → 神经电流）
        if self.enable_desktop_env:
            real_activation = self.real_encoder.encode(
                self.desktop_env.state, self.metabolism.state
            )
            # 叠加真实感官到主激活
            # ALPN 激活（食物气味）
            for nid, curr in real_activation.alpn_activation.items():
                if nid in activation.alpn_activation:
                    activation.alpn_activation[nid] = max(
                        activation.alpn_activation[nid], curr)
                else:
                    activation.alpn_activation[nid] = curr
            # DAN 激活（惩罚/奖励）
            for nid, curr in real_activation.dan_activation.items():
                if nid in activation.dan_activation:
                    activation.dan_activation[nid] = max(
                        activation.dan_activation[nid], curr)
                else:
                    activation.dan_activation[nid] = curr
            # 视觉逼近
            activation.loom_left = max(activation.loom_left, real_activation.loom_left)
            activation.loom_right = max(activation.loom_right, real_activation.loom_right)
            activation.looming_strength = max(
                activation.looming_strength, real_activation.looming_strength)
            activation.airpuff_strength = max(
                activation.airpuff_strength, real_activation.airpuff_strength)
            # 惩罚
            if real_activation.punishment:
                activation.punishment = True
                activation.punishment_intensity = max(
                    activation.punishment_intensity, real_activation.punishment_intensity)
            # 气味标签（优先真实环境的食物气味）
            if real_activation.odor_label != "none":
                activation.odor_label = real_activation.odor_label
            # 全局兴奋性（代谢调制）
            activation.global_excitability *= real_activation.global_excitability

        currents = self.encoder.get_input_current_vector()
        # 叠加真实感官的 ALPN/DAN 电流
        if self.enable_desktop_env:
            for nid, curr in real_activation.alpn_activation.items():
                if nid in currents:
                    currents[nid] = max(currents[nid], curr)
                else:
                    currents[nid] = curr
            for nid, curr in real_activation.dan_activation.items():
                if nid in currents:
                    currents[nid] = max(currents[nid], curr)
                else:
                    currents[nid] = curr

        # 2.8 视叶处理（基于 FlyWire 真实连接组的视觉网络）
        # 把桌面环境编码为视网膜输入，经过视叶 LIF 网络处理，
        # 输出运动方向、looming、小物体等视觉特征，叠加到主激活
        visual_output = None
        if self.enable_visual_lobe:
            # 编码视觉输入
            vis_input = self._encode_visual_input(sensor_data)
            # 设置视网膜输入
            self.visual_lobe.set_retina_input(vis_input)
            # 运行视叶仿真（dt_ms=4ms，分4步1ms仿真）
            for _ in range(max(1, int(dt_ms))):
                self.visual_lobe.step(dt_ms=1.0)
            # 获取视叶输出
            visual_output = self.visual_lobe.get_output()

            # 视叶输出叠加到主激活
            # looming 逼近
            if visual_output.looming > activation.looming_strength:
                activation.looming_strength = visual_output.looming
                activation.visual_label = "vl_looming"
            # 左右眼差异（用于转向）
            if visual_output.motion_total > 0:
                # 运动方向映射到左右眼
                dir_angle = visual_output.motion_direction
                if dir_angle > 0:  # 向右运动
                    activation.loom_right = max(activation.loom_right, visual_output.motion_total / 1000.0)
                else:  # 向左运动
                    activation.loom_left = max(activation.loom_left, visual_output.motion_total / 1000.0)
            # 逃逸触发（LC4 神经元激活）
            if visual_output.escape_trigger > 0.5:
                activation.punishment = True
                activation.punishment_intensity = max(
                    activation.punishment_intensity, visual_output.escape_trigger * 0.5)

        # 3. 输入到蘑菇体仿真
        # ALPN 输入（嗅觉）—— encoder 返回神经元 ID，需要转换为 sim 数组索引
        alpn_currents = {}
        for neuron_id, current in currents.items():
            if neuron_id in self.sim.id_to_idx:
                idx = self.sim.id_to_idx[neuron_id]
                if idx in self.sim.alpn_idx_set:
                    alpn_currents[idx] = current
        self.sim.set_odor_input(alpn_currents, odor_label=activation.odor_label)

        # 惩罚输入（电击 + 视觉强 looming）
        total_punishment = activation.punishment_intensity if activation.punishment else 0.0
        total_punishment += activation.punishment_signal  # 视觉触发的先天惩罚
        if total_punishment > 0.01:
            self.sim.set_punishment(total_punishment)
            if activation.punishment:
                self.punishment_count += 1
            self.last_punishment_time = time.monotonic()
        else:
            self.sim.punishment_strength = 0.0

        # 4. 运行仿真
        self.sim.step(dt_ms)

        # 5. 读取行为输出
        behavior = self.sim.get_behavior()

        # 5.1 代谢系统更新（能量、疲劳、昼夜、死亡）
        if self.enable_desktop_env:
            walk_intensity = behavior.get('walk_intensity', 0.5)
            env_state = self.desktop_env.state
            self.metabolism.update(
                dt=dt_ms / 1000.0,
                walk_intensity=walk_intensity,
                on_food=env_state.on_food,
                food_recovery_rate=env_state.food_recovery_rate,
                is_colliding=env_state.is_colliding,
                in_danger=env_state.in_danger,
                predator_caught=env_state.predator_escape,  # 假设逃跑时也会受伤
                is_resting=(not env_state.on_food and walk_intensity < 0.1),
            )

        # 5.3 探索驱动更新（内在动机系统）
        has_external = (
            activation.odor_label != "none"
            or activation.punishment
            or activation.looming_strength > 0.1
        )
        self.exploration.update(
            dt=dt_ms / 1000.0,
            has_external_input=has_external,
            novelty=sensor_data.window.novelty,
            punishment=activation.punishment,
            visual_looming=activation.looming_strength,
        )

        # 5.5 中央复合体导航更新
        vis = sensor_data.vision
        if vis.available:
            # 屏幕边界假设（DesktopFly 窗口大约 1200x800，果蝇在中心附近活动）
            screen_bounds = (-600, 600, -400, 400)
            self.cx.update(
                fly_position=(vis.fly_x, vis.fly_y),
                fly_heading=vis.fly_heading,
                screen_bounds=screen_bounds,
                approach=behavior.get("approach", 0.0),
                avoidance=behavior.get("avoidance", 0.0),
                looming_strength=activation.looming_strength,
                novelty=sensor_data.window.novelty,
            )

        # 5.8 桌面环境 + 代谢系统的行为调整
        if self.enable_desktop_env:
            env_state = self.desktop_env.state
            meta_state = self.metabolism.state

            # 死亡状态：停止所有运动
            if meta_state.state.value == "dead":
                behavior['walk_intensity'] = 0.0
                behavior['mbon_avg_rate'] = 0.0
                behavior['turn_bias'] = 0.0
            else:
                # 碰撞强制转向（叠加到 turn_bias）
                if env_state.force_turn != 0.0:
                    current_turn = behavior.get('turn_bias', 0.0)
                    behavior['turn_bias'] = max(-1.0, min(1.0,
                        current_turn + env_state.force_turn * 0.8))

                # 环境速度倍率（碰撞减速、逃跑加速）
                if 'walk_intensity' in behavior:
                    behavior['walk_intensity'] *= env_state.speed_multiplier

                # 代谢速度修正（疲劳/能量/夜间减速）
                if 'walk_intensity' in behavior:
                    behavior['walk_intensity'] *= meta_state.speed_modifier

                # 代谢唤醒度修正
                if 'arousal' in behavior:
                    behavior['arousal'] *= meta_state.arousal_modifier
                else:
                    behavior['arousal'] = 0.3 * meta_state.arousal_modifier

                # 环境额外唤醒（捕食者接近）
                if env_state.extra_arousal > 0:
                    behavior['arousal'] = min(1.0,
                        behavior.get('arousal', 0.3) + env_state.extra_arousal)

                # 需要食物时增强探索（内在动机）
                if meta_state.need_food:
                    behavior['walk_intensity'] = min(1.0,
                        behavior.get('walk_intensity', 0.3) * 1.3)

        # 6. 转换为 DesktopFly BrainSignals 并写入共享文件
        brain_signals = self.behavior_to_brain_signals(behavior, activation, sensor_data)

        # 6.5 环境行为调整直接写入 brain_signals（确保碰撞/逃跑一定生效）
        if self.enable_desktop_env:
            env_state = self.desktop_env.state
            # 只要有任何环境调整（转向/速度/唤醒）就应用
            has_adjustment = (
                env_state.force_turn != 0.0 or
                env_state.speed_multiplier != 1.0 or
                env_state.extra_arousal > 0.0 or
                env_state.extra_punishment > 0.0
            )
            if has_adjustment:
                # 强制转向（增强系数到 0.8，确保碰撞时明显转向）
                if env_state.force_turn != 0.0:
                    current_turn = brain_signals.get('turnBias', 0.0)
                    brain_signals['turnBias'] = max(-1.0, min(1.0,
                        current_turn + env_state.force_turn * 0.8))
                # 速度调整（碰撞减速、逃跑加速）
                brain_signals['walkDrive'] *= env_state.speed_multiplier
                brain_signals['walkDrive'] = max(0.0, min(1.5, brain_signals['walkDrive']))
                # 额外唤醒（捕食者接近）
                if env_state.extra_arousal > 0:
                    brain_signals['arousal'] = min(1.0,
                        brain_signals.get('arousal', 0.3) + env_state.extra_arousal)
                # 逃跑时振翅
                if env_state.predator_escape:
                    brain_signals['wingDrive'] = max(
                        brain_signals.get('wingDrive', 0.0), 0.8)
                    brain_signals['escape'] = True

        # 6.8 代谢状态写入 brain_signals（供 DesktopFly 显示）
        if self.enable_desktop_env:
            meta = self.metabolism.state
            brain_signals['energy'] = meta.energy
            brain_signals['fatigue'] = meta.fatigue
            brain_signals['health'] = meta.health
            brain_signals['fly_state'] = meta.state.value
            brain_signals['on_food'] = self.desktop_env.state.on_food
            brain_signals['is_colliding'] = self.desktop_env.state.is_colliding
            brain_signals['predator_alert'] = self.desktop_env.state.predator_alert

        if self.write_signals:
            self.write_brain_signals(brain_signals)

        # 7. 记录
        self.frame_count += 1
        if activation.odor_label != "none":
            self.odor_history.append((time.monotonic(), activation.odor_label))

        # 7.5 体细胞进化引擎更新（收集适应度数据，周期性突变）
        if self.enable_evolution:
            energy = self.metabolism.state.energy
            arousal = brain_signals.get('arousal', 0.3)
            punishment = activation.punishment
            punishment_intensity = activation.punishment_intensity if hasattr(activation, 'punishment_intensity') else 0.5
            on_food = self.desktop_env.state.on_food if self.enable_desktop_env else False
            is_colliding = self.desktop_env.state.is_colliding if self.enable_desktop_env else False
            is_resting = behavior.get('resting', False) or self.metabolism.state.state.value == 'sleeping'
            position = (self._virt_fly_x, self._virt_fly_y)

            self.evolution.update(
                dt=dt_ms / 1000.0,
                energy=energy,
                arousal=arousal,
                punishment=punishment,
                punishment_intensity=punishment_intensity,
                on_food=on_food,
                is_colliding=is_colliding,
                is_resting=is_resting,
                position=position,
            )

        # 7.6 内在可塑性更新（神经元阈值/时间常数自调整）
        if self.enable_intrinsic_plasticity:
            # 获取当前帧放电的神经元
            spikes = np.zeros(self.sim.n, dtype=np.float32)
            # 从 sim 中获取放电信息（如果有的话）
            if hasattr(self.sim, 'last_spiked'):
                spikes[self.sim.last_spiked] = 1.0
            self.intrinsic_plasticity.update(spikes, dt_ms=dt_ms)
            # 应用内在可塑性的调整到蘑菇体
            self.sim.global_excitability *= (
                self.intrinsic_plasticity.config.global_excitability
            )

        # 8. 内心独白生成（基于神经状态，由本地 mimi 模型实时生成）
        # 每隔 interval 秒生成一次，异步不阻塞主循环
        now = time.monotonic()
        if self.thought_gen.should_generate(now):
            neural_state = {
                'kc_active': behavior.get('kc_active', 0),
                'kc_total': len(self.sim.kc_idx),
                'mbon_rate': behavior.get('mbon_avg_rate', 0.0),
                'odor': activation.odor_label,
                'punishment': activation.punishment,
                'curiosity': self.exploration.state.curiosity,
                'fatigue': self.exploration.state.fatigue,
                'walk_drive': brain_signals.get('walkDrive', 0.0),
                'is_exploring': self.exploration.state.is_exploring,
            }
            self.thought_gen.generate_async(neural_state)

        return sensor_data, activation, behavior

    def _update_virtual_fly_position(self, dt: float):
        """
        更新虚拟果蝇位置（根据上一帧的大脑信号推算）

        不依赖 DesktopFly 的位置输出，Python 端自己维护位置，
        用于碰撞检测、食物检测等桌面环境交互。

        Args:
            dt: 时间步长（秒）
        """
        # 读取上一帧的大脑信号（如果有）
        try:
            with open(BRAIN_SIGNAL_PATH, 'r') as f:
                brain = json.load(f)
            walk_drive = brain.get('walkDrive', 0.3)
            turn_bias = brain.get('turnBias', 0.0)
        except (FileNotFoundError, json.JSONDecodeError):
            walk_drive = 0.3
            turn_bias = 0.0

        # 速度参数（像素/秒）
        base_speed = 80.0  # 基础速度
        speed = walk_drive * base_speed * 2.0

        # 更新朝向
        self._virt_fly_heading += turn_bias * dt * 3.0
        # 归一化到 -pi 到 pi
        while self._virt_fly_heading > math.pi:
            self._virt_fly_heading -= 2 * math.pi
        while self._virt_fly_heading < -math.pi:
            self._virt_fly_heading += 2 * math.pi

        # 更新位置
        self._virt_fly_x += math.cos(self._virt_fly_heading) * speed * dt
        self._virt_fly_y += math.sin(self._virt_fly_heading) * speed * dt

        # 屏幕边界限制（假设屏幕 2560x1664，DesktopFly 坐标范围）
        max_x = 1200.0
        max_y = 780.0
        self._virt_fly_x = max(-max_x, min(max_x, self._virt_fly_x))
        self._virt_fly_y = max(-max_y, min(max_y, self._virt_fly_y))

        # 写入共享文件（模拟 DesktopFly 的位置输出）
        vision_data = {
            "fly_x": self._virt_fly_x,
            "fly_y": self._virt_fly_y,
            "fly_heading": self._virt_fly_heading,
            "mouse_distance": 9999.0,  # 鼠标距离由 sensors.py 单独获取
            "mouse_bearing": 0.0,
            "loom_left": 0.0,
            "loom_right": 0.0,
            "loom_total": 0.0,
            "air_puff": 0.0,
            "timestamp": time.time(),
            "source": "python_virtual",  # 标记为 Python 虚拟位置
        }
        try:
            tmp_path = VISION_FILE + ".tmp"
            with open(tmp_path, 'w') as f:
                json.dump(vision_data, f)
            os.rename(tmp_path, VISION_FILE)
        except Exception:
            pass

    def _encode_visual_input(self, sensor_data) -> VisualInput:
        """
        将桌面环境信息编码为视叶的视觉输入（模拟视网膜）

        把果蝇周围的窗口、鼠标、屏幕亮度等编码为 8x8 视网膜阵列，
        以及 looming、小物体运动、大范围运动等视觉特征。

        Args:
            sensor_data: 感官数据

        Returns:
            VisualInput: 视觉输入
        """
        vis_input = VisualInput()

        # 屏幕亮度（全局光强）
        brightness = sensor_data.brightness.brightness if hasattr(sensor_data, 'brightness') else 0.5
        vis_input.light_intensity = brightness

        # 初始化视网膜阵列（8x8，默认均匀光强）
        retina = np.ones((8, 8)) * brightness

        if self.enable_desktop_env:
            env_state = self.desktop_env.state

            # 果蝇屏幕坐标和朝向
            fly_x, fly_y = env_state.fly_x, env_state.fly_y
            heading = env_state.fly_heading

            # 把窗口编码为视网膜上的暗区（障碍物）
            for win in self.window_detector.windows:
                # 窗口中心相对于果蝇的位置
                cx, cy = win.center
                dx = cx - fly_x
                dy = cy - fly_y

                # 转换到果蝇视角（相对朝向）
                dist = math.hypot(dx, dy)
                if dist > 500:  # 太远的不编码
                    continue

                bearing = math.atan2(dy, dx) - heading
                # 归一化到 -pi 到 pi
                while bearing > math.pi:
                    bearing -= 2 * math.pi
                while bearing < -math.pi:
                    bearing += 2 * math.pi

                # 只编码前方 120 度视野
                if abs(bearing) > math.pi * 2 / 3:
                    continue

                # 映射到视网膜列（-60度到+60度 → 0到7列）
                col = int((bearing + math.pi / 3) / (2 * math.pi / 3) * 7)
                col = max(0, min(7, col))

                # 距离越近，遮挡越大（视网膜上的暗区越大）
                occlusion = max(0, 1.0 - dist / 500.0)
                for row in range(8):
                    retina[row, col] = min(retina[row, col], brightness * (1 - occlusion * 0.8))

            # 鼠标编码为移动的小物体
            mouse_dist = env_state.mouse_distance
            if mouse_dist < 300:
                mouse_bearing = env_state.mouse_bearing - heading
                while mouse_bearing > math.pi:
                    mouse_bearing -= 2 * math.pi
                while mouse_bearing < -math.pi:
                    mouse_bearing += 2 * math.pi

                if abs(mouse_bearing) < math.pi * 2 / 3:
                    col = int((mouse_bearing + math.pi / 3) / (2 * math.pi / 3) * 7)
                    col = max(0, min(7, col))
                    # 鼠标是亮的小物体
                    for row in range(3, 6):
                        retina[row, col] = min(1.0, retina[row, col] + 0.3)

                    # 鼠标靠近 = looming
                    if mouse_dist < 150:
                        vis_input.looming = max(vis_input.looming, 1.0 - mouse_dist / 150.0)
                        vis_input.looming_direction = mouse_bearing
                        vis_input.small_object = True
                        vis_input.object_direction = mouse_bearing

            # 前方障碍物距离也作为 looming
            if env_state.visual_obstacle_distance < 200:
                vis_input.looming = max(vis_input.looming,
                    1.0 - env_state.visual_obstacle_distance / 200.0)

        # 窗口切换 = 大范围运动（光流）
        if hasattr(sensor_data, 'window') and sensor_data.window.novelty > 0.3:
            vis_input.large_field_motion = sensor_data.window.novelty

        vis_input.retina = retina
        return vis_input

    def _apply_genome(self):
        """
        应用基因型参数到各个模块

        把可塑基因型的21个超参数应用到：
        - 蘑菇体（阈值/学习率/自发放电）
        - 代谢系统（代谢率/能量消耗/食物吸收）
        - 探索驱动（好奇心/疲劳）
        - 行为参数（速度/转向/风险容忍）
        - 视叶（逃逸敏感度/运动增益）

        每帧调用，突变后立即生效。
        """
        g = self.genome

        # === 蘑菇体参数 ===
        # 全局兴奋性（影响阈值）
        self.sim.global_excitability = g.get("baseline_dopamine") * 1.5 + 0.25
        # 时间常数调制
        self.sim.global_tau_modulator = 1.0 + (g.get("memory_decay_rate") - 0.001) * 100
        # 自发放电率调制
        self.sim.global_spontaneous_modulator = g.get("exploration_rate") * 1.5 + 0.25
        # STDP 学习率
        self.sim.lr_punish = g.get("stdp_learning_rate") * g.get("punishment_sensitivity")
        self.sim.lr_reward = g.get("stdp_learning_rate") * g.get("dopamine_reward_sensitivity")

        # === 代谢系统参数 ===
        self.metabolism.base_energy_consumption = g.get("base_metabolism")
        self.metabolism.movement_energy_cost = g.get("movement_energy_cost")
        self.metabolism.food_absorption_rate = g.get("food_absorption_rate")
        self.metabolism.fatigue_accumulation = g.get("fatigue_accumulation_rate")

        # === 探索驱动参数 ===
        self.exploration.curiosity_gain = g.get("exploration_rate")
        self.exploration.spontaneous_turn_prob = g.get("spontaneous_turn_prob")

        # === 桌面环境参数 ===
        # 风险容忍度影响捕食者逃跑阈值
        risk = g.get("risk_tolerance")
        self.desktop_env.predator_alert_distance = 80 + (1 - risk) * 120  # 0.05→195, 1.0→80
        self.desktop_env.predator_escape_distance = 40 + (1 - risk) * 80

        # === 行为参数（在 behavior_to_brain_signals 中使用）===
        # 存在实例变量中，behavior_to_brain_signals 读取
        self._genome_max_speed = g.get("max_walk_speed")
        self._genome_turn_agility = g.get("turn_agility")
        self._genome_wing_threshold = g.get("wing_flap_threshold")
        self._genome_baseline_arousal = g.get("baseline_arousal")
        self._genome_stress_gain = g.get("stress_response_gain")
        self._genome_recovery_rate = g.get("recovery_rate")
        self._genome_escape_sensitivity = g.get("visual_escape_sensitivity")
        self._genome_motion_gain = g.get("visual_motion_gain")
        self._genome_looming_strength = g.get("looming_response_strength")

    def behavior_to_brain_signals(self, behavior, activation, sensor_data):
        """
        将蘑菇体行为输出转换为 DesktopFly BrainSignals 格式。

        映射逻辑：
          - MBON 活跃（趋向）→ walkDrive 向前行走
          - 回避倾向 → turnBias 转向 + nervous 紧张
          - 惩罚/电击 → escape 逃逸起飞 + wingDrive 振翅
          - 新奇刺激 → arousal 唤醒 + wingDrive 轻微振翅
          - 低亮度/夜晚 → sleep 睡眠状态
        """
        approach = behavior.get("approach", 0.0)
        avoidance = behavior.get("avoidance", 0.0)
        mbon_rate = behavior.get("mbon_avg_rate", 0.0)
        odor = activation.odor_label
        punishment = activation.punishment

        # === 行为映射：不同输入产生明显不同的行为 ===
        # 原则：让用户能清晰建立"我的行为 → 果蝇行为"的因果链

        t = time.monotonic()

        if punishment:
            # 电击/惩罚：逃逸起飞，振翅，快速转向
            walk_drive = 0.3
            turn_bias = math.sin(t * 3.0) * 1.0  # 快速左右摆动（恐慌）
            escape = True
            backward = False
            wing_drive = 1.3
            arousal = 1.0
        elif odor == "odor_A_fast" and avoidance < 0.5:
            # 气味A（快速鼠标）+ 未学习回避：快速向前趋向
            walk_drive = 1.3
            turn_bias = math.sin(t * 0.5) * 0.15  # 轻微偏航
            escape = False
            backward = False
            wing_drive = 0.0
            arousal = 0.6
        elif odor == "odor_A_fast" and avoidance >= 0.5:
            # 气味A + 已学习回避：急转回避，减速
            walk_drive = 0.25
            turn_bias = math.sin(t * 1.5) * 0.9  # 快速转向（回避）
            escape = False
            backward = False
            wing_drive = 0.3
            arousal = 0.7
        elif odor == "odor_B_slow":
            # 气味B（缓慢画圈）：缓慢转圈探索
            walk_drive = 0.45
            turn_bias = 0.5 + 0.2 * math.sin(t * 0.3)  # 持续转圈
            escape = False
            backward = False
            wing_drive = 0.0
            arousal = 0.35
        elif odor == "odor_C_random":
            # 气味C（中速随机）：中速探索
            walk_drive = 0.85
            turn_bias = math.sin(t * 0.8) * 0.4
            escape = False
            backward = False
            wing_drive = 0.0
            arousal = 0.5
        else:
            # 无外部输入：由神经自发放电 + 探索驱动共同决定行为
            # 这才是"活的大脑"——不是脚本写死的随机游走，而是神经元自身的节律在驱动行为
            exp_behavior = self.exploration.get_exploration_behavior()

            # MBON 自发放电率调制行为强度（真实神经活动参与决策）
            # MBON 放电率越高，果蝇越活跃；放电率低时更安静
            mbon_mod = 0.4 + min(1.0, mbon_rate / 40.0) * 0.8  # 0-40Hz 映射到 0.4-1.2
            walk_drive = exp_behavior['walk_drive'] * mbon_mod
            turn_bias = exp_behavior['turn_bias']
            escape = False
            backward = False
            wing_drive = 0.0
            # 唤醒度也由神经活动决定
            arousal = max(0.2, min(1.0, 0.2 + mbon_rate / 60.0))

        # 新奇刺激（窗口切换）：振翅 + 唤醒
        novelty = sensor_data.window.novelty if hasattr(sensor_data, 'window') else 0.0
        if novelty > 0.3 and not punishment:
            wing_drive = max(wing_drive, novelty * 0.6)
            arousal = max(arousal, novelty)

        # 紧张度初始化（视觉处理中会用到）
        nervous = 0.0

        # === 视觉输入覆盖：looming 逼近刺激 ===
        vis = sensor_data.vision if hasattr(sensor_data, 'vision') else None
        if vis and vis.available and vis.loom_total > 0.15:
            loom = vis.loom_total
            # 左右眼差异 → 转向方向（左眼逼近→向右转，右眼逼近→向左转）
            eye_diff = vis.loom_left - vis.loom_right  # 正=左眼更强
            turn_from_vision = -eye_diff * 1.5  # 反向转向（远离逼近源）

            if loom > 0.5:
                # 强逼近：先天逃逸反应
                escape = True
                walk_drive = max(walk_drive, 0.6)
                turn_bias = turn_from_vision + math.sin(t * 4.0) * 0.3  # 快速转向+抖动
                wing_drive = max(wing_drive, 1.0)
                arousal = max(arousal, 0.9)
                nervous = min(1.0, nervous + loom * 0.5)
            elif loom > 0.3:
                # 中等逼近：警觉+减速+转向
                walk_drive = min(walk_drive, 0.4)
                turn_bias = turn_from_vision * 0.8
                arousal = max(arousal, 0.6)
                nervous = min(1.0, nervous + loom * 0.3)
            else:
                # 弱逼近：轻微警觉
                arousal = max(arousal, 0.3 + loom)

        # 紧张度 = 回避倾向
        nervous = min(1.0, avoidance)

        # 睡眠：低亮度/夜晚
        brightness = sensor_data.brightness.brightness if hasattr(sensor_data, 'brightness') else 0.5
        sleep = brightness < 0.25

        # 梳理：暂时不映射
        groom_drive = 0.0

        # 温度/节奏：默认 1.0
        tempo = 1.0

        # === 中央复合体导航整合 ===
        cx_turn, cx_speed, cx_explore = 0.0, 1.0, 0.0  # 默认值
        cx_state = self.cx.state
        if cx_state.last_update_time > 0:
            cx_turn, cx_speed, cx_explore = self.cx.get_navigation_command()

            if cx_state.near_boundary and not escape:
                # 边界回避时，中央复合体主导转向
                turn_bias = turn_bias * 0.3 + cx_turn * 0.7
                walk_drive = walk_drive * cx_speed
            elif not escape and not punishment:
                # 正常探索时，中央复合体调整转向和速度
                turn_bias = turn_bias * 0.6 + cx_turn * 0.4
                walk_drive = walk_drive * cx_speed

            # 探索驱动力影响唤醒度
            arousal = max(arousal, cx_explore * 0.4)

        # === 运动控制精细化（步态/梳理/起飞降落/后退）===
        vis = sensor_data.vision if hasattr(sensor_data, 'vision') else None
        motor_state = self.motor.update(
            base_walk=walk_drive,
            base_turn=turn_bias,
            escape=escape,
            approach=approach,
            avoidance=avoidance,
            looming_strength=activation.looming_strength,
            novelty=sensor_data.window.novelty,
            brightness=sensor_data.brightness.brightness,
            near_boundary=self.cx.state.near_boundary,
            cx_turn=cx_turn,
            cx_speed_mod=cx_speed,
        )

        brain_signals = {
            "escape": motor_state.behavior_mode == "escape" or motor_state.flying,
            "nervous": round(motor_state.nervous, 3),
            "turnBias": round(motor_state.turn_bias, 3),
            "backward": motor_state.backward,
            "walkDrive": round(motor_state.walk_drive, 3),
            "groomDrive": round(motor_state.groom_drive, 3),
            "wingDrive": round(motor_state.wing_drive, 3),
            "arousal": round(motor_state.arousal, 3),
            "tempo": round(tempo, 3),
            "sleep": motor_state.sleep,
            "source": "mushroom_body",
            "timestamp": time.time(),
            "mbon_rate": round(mbon_rate, 1),
            "approach": round(approach, 3),
            "avoidance": round(avoidance, 3),
            "odor": activation.odor_label,
            "kc_active": behavior.get("kc_active", 0),
            "stdp_updates": self.sim.stdp_update_count,
            "punishment_count": self.punishment_count,
            "behavior_mode": motor_state.behavior_mode,
            "flying": motor_state.flying,
        }

        # === 基因型参数调制（体细胞进化的核心）===
        if self.enable_evolution:
            # 最大速度调制
            brain_signals["walkDrive"] *= getattr(self, '_genome_max_speed', 1.0)
            brain_signals["walkDrive"] = min(1.5, brain_signals["walkDrive"])
            # 转向敏捷度调制
            brain_signals["turnBias"] *= getattr(self, '_genome_turn_agility', 1.0)
            brain_signals["turnBias"] = max(-1.0, min(1.0, brain_signals["turnBias"]))
            # 振翅阈值：唤醒度超过阈值才振翅
            wing_threshold = getattr(self, '_genome_wing_threshold', 0.6)
            if brain_signals["arousal"] < wing_threshold:
                brain_signals["wingDrive"] *= 0.3
            # 基线唤醒度调制
            baseline_arousal = getattr(self, '_genome_baseline_arousal', 0.3)
            brain_signals["arousal"] = max(brain_signals["arousal"], baseline_arousal)
            # 逃逸敏感度调制
            escape_sens = getattr(self, '_genome_escape_sensitivity', 0.7)
            if brain_signals.get("nervous", 0) > (1 - escape_sens):
                brain_signals["escape"] = True

        return brain_signals

    def write_brain_signals(self, signals):
        """原子写入大脑信号到共享文件（先写临时文件再 rename）"""
        try:
            dir_name = os.path.dirname(BRAIN_SIGNAL_PATH)
            fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
            with os.fdopen(fd, 'w') as f:
                json.dump(signals, f)
            os.rename(tmp_path, BRAIN_SIGNAL_PATH)
        except Exception as e:
            # 写入失败不影响主循环
            pass

    def print_status(self, sensor_data, activation, behavior):
        """打印实时状态"""
        # 感官状态
        mouse = sensor_data.mouse
        vol = sensor_data.volume
        kb = sensor_data.keyboard
        win = sensor_data.window
        bright = sensor_data.brightness

        # 时间
        elapsed = time.monotonic() - self.start_time if self.start_time else 0

        # 行为状态
        if behavior["avoidance"] > 0.5:
            behavior_str = f"⚠️回避({behavior['avoidance']:.2f})"
        elif behavior["approach"] > 0.7:
            behavior_str = f"➡️趋向({behavior['approach']:.2f})"
        else:
            behavior_str = f"○中立(趋={behavior['approach']:.2f}避={behavior['avoidance']:.2f})"

        # 惩罚状态
        punish_str = "⚡" if activation.punishment else " "

        # 新奇状态
        novelty_str = "✨" if win.novelty > 0.3 else " "

        # 触觉状态
        tactile_str = "⌨️" if kb.typing_burst else " "

        # 视觉状态
        vis = sensor_data.vision
        visual_str = f"👁️{activation.visual_label}:{activation.looming_strength:.2f}" if vis.available else "👁️--"

        # 中央复合体状态
        cx = self.cx.state
        boundary_str = "⚠️边界" if cx.near_boundary else "  "
        nav_str = f"🧭turn={cx.turn_bias:+.2f} spd={cx.speed_modulation:.2f} nov={cx.novelty_at_position:.2f}{boundary_str}"

        # 运动控制状态
        motor = self.motor.state
        mode_emoji = {
            "rest": "💤", "walk": "🚶", "turn": "🔄", "escape": "⚡",
            "groom": "🧹", "fly": "✈️", "sleep": "😴", "avoid": "↩️"
        }.get(motor.behavior_mode, "❓")
        motor_str = f"{mode_emoji}{motor.behavior_mode:6s}"
        if motor.flying:
            motor_str += "✈️"
        if motor.groom_drive > 0.1:
            motor_str += "🧹"

        # 探索驱动状态（内在动机）
        exp = self.exploration.state
        if exp.is_exploring:
            exp_str = f"🔍探索(curi={exp.curiosity:.2f} fat={exp.fatigue:.2f} drive={exp.exploration_drive:.2f})"
        elif time.monotonic() < exp.rest_until:
            exp_str = f"😴休息中(fat={exp.fatigue:.2f})"
        else:
            exp_str = f"🧘静息(curi={exp.curiosity:.2f} fat={exp.fatigue:.2f})"

        # 桌面环境 + 代谢状态
        if self.enable_desktop_env:
            env = self.desktop_env.state
            meta = self.metabolism.state

            # 环境状态
            env_parts = []
            if env.is_colliding:
                env_parts.append(f"💥撞{env.collision_window[:6]}")
            if env.on_food:
                env_parts.append(f"🍎吃{env.food_window[:6]}")
            if env.predator_escape:
                env_parts.append("⚡逃!")
            elif env.predator_alert:
                env_parts.append(f"👀鼠{env.mouse_distance:.0f}")
            env_str = " ".join(env_parts) if env_parts else "🌿平静"

            # 代谢状态
            state_emoji = {
                "alive": "🟢", "sleeping": "😴", "dying": "🟡", "dead": "💀"
            }.get(meta.state.value, "❓")
            meta_str = (f"{state_emoji}{meta.state.value[:4]} "
                       f"⚡{meta.energy:.2f} "
                       f"😫{meta.fatigue:.2f} "
                       f"❤️{meta.health:.2f}"
                       f"{'🌙' if meta.is_night else '☀️'}")
        else:
            env_str = "🌿环境关"
            meta_str = "🟢代谢关"

        # 体细胞进化状态
        if self.enable_evolution:
            evo_status = self.evolution.get_status()
            evo_phase_emoji = {
                "基线期": "📊", "突变评估期": "🧬", "冷却期": "⏳"
            }.get(evo_status["phase"], "🔄")
            evo_str = (f"{evo_phase_emoji}{evo_status['phase']}"
                      f"({evo_status['phase_progress']*100:.0f}%) "
                      f"🧬代{evo_status['generation']} "
                      f"✓{evo_status['accepted_mutations']}"
                      f"✗{evo_status['rejected_mutations']} "
                      f"🎯{evo_status['personality'][:8]}")
        else:
            evo_str = "🧬进化关"

        status = (
            f"\r[{elapsed:6.1f}s] "
            f"🖱️v={mouse.speed:6.1f} 曲率={mouse.curvature:.3f} | "
            f"🔊{vol.volume:5.1f}%{punish_str} | "
            f"⌨️{kb.key_press_rate:.1f}/s{tactile_str} | "
            f"🪟{win.frontmost_app[:8]:8s}{novelty_str} | "
            f"💡{bright.brightness:.2f}({bright.source[:4]}) | "
            f"👃{activation.odor_label:12s} | "
            f"{visual_str:18s} | "
            f"🧠MBON={behavior['mbon_avg_rate']:6.1f}Hz | "
            f"🎯{behavior_str} | "
            f"KC={behavior['kc_active']:4d} | "
            f"STDP={self.sim.stdp_update_count:8d} | "
            f"{nav_str} | "
            f"{motor_str} | "
            f"{exp_str} | "
            f"{env_str} | "
            f"{meta_str} | "
            f"{evo_str}"
        )
        print(status, end='', flush=True)

    def run(self, duration_sec: float = None, print_interval: float = 0.1):
        """
        运行实时交互

        Args:
            duration_sec: 运行时长（None = 无限，直到 Ctrl+C）
            print_interval: 打印间隔（秒）
        """
        print("\n" + "=" * 80)
        print("🪰 桌面果蝇具身智能 - 实时交互系统")
        print("=" * 80)
        print("\n操作说明:")
        print("  🖱️  快速直线移动鼠标 → 气味 A（危险气味）")
        print("  🖱️  缓慢画圈移动鼠标 → 气味 B（安全气味）")
        print("  🔊 快速调大系统音量 → 电击/惩罚（触发 STDP 学习）")
        print("  ⌨️  快速打字（修饰键+高频）→ 触觉刺激")
        print("  🪟 切换应用/窗口 → 新奇刺激（PAM 多巴胺激活，驱动探索）")
        print("  💡 屏幕亮度/时间 → 昼夜节律（全局兴奋性调制）")
        print("  🧠 系统会实时学习：某种鼠标模式 + 电击 → 学会回避该模式")
        print("  按 Ctrl+C 退出\n")
        print("-" * 80)

        self.running = True
        self.start_time = time.monotonic()
        last_print = 0

        try:
            while self.running:
                if duration_sec and (time.monotonic() - self.start_time) > duration_sec:
                    break

                sensor_data, activation, behavior = self.step(dt_ms=4)

                now = time.monotonic()
                if now - last_print > print_interval:
                    self.print_status(sensor_data, activation, behavior)
                    last_print = now

                # 60fps 限制
                elapsed = time.monotonic() - now
                sleep_time = max(0, 1.0/60 - elapsed)
                time.sleep(sleep_time)

                # 定期自动保存权重
                if self.enable_learning and (now - self.last_autosave_time) > AUTOSAVE_INTERVAL:
                    if self.sim.stdp_update_count > 0:
                        self.sim.save_weights(WEIGHTS_PATH)
                    self.last_autosave_time = now

        except KeyboardInterrupt:
            print("\n\n" + "-" * 80)
            print("用户中断，正在停止...")

        self.running = False

        # 退出时保存权重
        if self.enable_learning and self.sim.stdp_update_count > 0:
            print("\n💾 保存学习权重...")
            self.sim.save_weights(WEIGHTS_PATH)

        self._print_summary()

    def _print_summary(self):
        """打印运行总结"""
        elapsed = time.monotonic() - self.start_time if self.start_time else 0

        print("\n" + "=" * 80)
        print("📊 运行总结")
        print("=" * 80)
        print(f"  运行时长: {elapsed:.1f} 秒")
        print(f"  总帧数: {self.frame_count}")
        print(f"  平均帧率: {self.frame_count/max(1,elapsed):.1f} fps")
        print(f"  惩罚（电击）次数: {self.punishment_count}")
        print(f"  STDP 总更新数: {self.sim.stdp_update_count}")

        # 权重变化
        weight_changes = self.sim.get_weight_changes()
        print(f"\n  突触权重变化:")
        print(f"    总 KC→MBON 连接: {weight_changes['total']}")
        print(f"    发生变化的连接: {weight_changes['changed']} ({weight_changes['changed']/weight_changes['total']*100:.1f}%)")
        print(f"    平均权重变化: {weight_changes['avg_change']:.6f}")

        # 学习状态
        print(f"\n  当前学习状态:")
        print(f"    学习功能: {'开启' if self.sim.enable_stdp else '关闭'}")
        print(f"    蘑菇体神经元: {self.sim.n} (KC={len(self.sim.kc_idx)}, "
              f"ALPN={len(self.sim.alpn_idx)}, DAN={len(self.sim.dan_idx)}, "
              f"MBON={len(self.sim.mbon_idx)})")

        print("\n" + "=" * 80)
        print("💡 提示：再次运行时，果蝇会保留之前学到的关联（如果不重置权重）")
        print("   要重置学习，调用 sim.reset_learning()")
        print("=" * 80 + "\n")


def main():
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else None

    circuit_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "mushroom_body", "circuit.json"
    )

    fly = EmbodiedFly(circuit_path, enable_learning=True)
    fly.run(duration_sec=duration)


if __name__ == "__main__":
    main()
