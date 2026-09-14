#!/usr/bin/env python3
"""
求偶神经回路模块 - 果蝇性别二型性与求偶行为

基于真实果蝇神经科学：
  - 雄蝇：P1 神经元簇（求偶指挥官），闻到雌蝇信息素后激活
  - 雌蝇：pC1 神经元（接纳考核中枢），评估雄蝇求偶之歌频率
  - 求偶之歌：雄蝇单侧翅膀振动，160-300Hz 正弦波
  - 信息素：性信息素随距离衰减，异性接收

参考：FlyWire（雌蝇连接组）、MaleCNS（雄蝇连接组）、
      Drosophila courtship circuitry (P1/pC1 neurons)
"""

import math
import time
import random
from enum import Enum
from dataclasses import dataclass, field


# ============================================================
# 性别枚举
# ============================================================

class Gender(Enum):
    """果蝇性别"""
    MALE = "male"
    FEMALE = "female"


# ============================================================
# 求偶状态枚举
# ============================================================

class CourtshipPhase(Enum):
    """求偶阶段（雄蝇视角）"""
    IDLE = "idle"                    # 空闲，无求偶对象
    ORIENTING = "orienting"          # 定位雌蝇，转向
    CHASING = "chasing"              # 追逐雌蝇
    TAPPING = "tapping"              # 触碰雌蝇（前足敲击）
    SINGING = "singing"              # 展翅唱歌（求偶之歌）
    LICKING = "licking"              # 舔舐雌蝇腹部
    ATTEMPTING_MOUNT = "attempting"  # 尝试交配
    MATING = "mating"                # 交配中
    POST_MATING = "post_mating"      # 交配后（不应期）


class FemaleResponse(Enum):
    """雌蝇回应状态"""
    UNAWARE = "unaware"              # 未察觉雄蝇
    DETECTED = "detected"            # 检测到雄蝇信息素
    EVALUATING = "evaluating"        # 评估雄蝇求偶之歌
    RECEPTIVE = "receptive"          # 接受交配
    REJECTING = "rejecting"          # 拒绝（踢开、飞走）
    POST_MATING = "post_mating"      # 交配后（不应期）


# ============================================================
# 信息素传播场
# ============================================================

@dataclass
class PheromoneSource:
    """信息素源"""
    x: float = 0.0
    y: float = 0.0
    gender: Gender = Gender.MALE
    intensity: float = 1.0           # 信息素发射强度
    emission_rate: float = 0.8       # 发射速率（0-1）


class PheromoneField:
    """
    信息素传播场

    模拟果蝇性信息素的空间传播：
      - 每只果蝇持续发射信息素
      - 信息素浓度随距离指数衰减
      - 只有异性能检测到对方的信息素
      - 信息素浓度影响 P1/pC1 神经元激活
    """

    def __init__(self, decay_rate: float = 0.015, detection_threshold: float = 0.05):
        """
        Args:
            decay_rate: 信息素衰减率（每像素）
            detection_threshold: 检测阈值（低于此值检测不到）
        """
        self.decay_rate = decay_rate
        self.detection_threshold = detection_threshold
        self.sources: list[PheromoneSource] = []

    def add_source(self, source: PheromoneSource):
        """添加信息素源"""
        self.sources.append(source)

    def remove_source(self, source: PheromoneSource):
        """移除信息素源"""
        if source in self.sources:
            self.sources.remove(source)

    def get_concentration(self, x: float, y: float, receiver_gender: Gender) -> float:
        """
        获取某点的异性信息素浓度

        Args:
            x, y: 接收者位置
            receiver_gender: 接收者性别（只检测异性）

        Returns:
            信息素浓度（0-1）
        """
        total = 0.0
        for src in self.sources:
            if src.gender == receiver_gender:
                continue  # 只检测异性
            dist = math.hypot(src.x - x, src.y - y)
            # 指数衰减 + 源强度
            concentration = src.intensity * src.emission_rate * math.exp(-self.decay_rate * dist)
            total += concentration
        return min(1.0, total)

    def get_strongest_source_direction(self, x: float, y: float, receiver_gender: Gender) -> tuple[float, float]:
        """
        获取最强异性信息素源的方向（用于追逐）

        Returns:
            (direction_angle, concentration): 方向角（弧度）和浓度
        """
        best_conc = 0.0
        best_angle = 0.0
        for src in self.sources:
            if src.gender == receiver_gender:
                continue
            dist = math.hypot(src.x - x, src.y - y)
            conc = src.intensity * src.emission_rate * math.exp(-self.decay_rate * dist)
            if conc > best_conc:
                best_conc = conc
                best_angle = math.atan2(src.y - y, src.x - x)
        return best_angle, best_conc


# ============================================================
# 雄蝇 P1 求偶中枢
# ============================================================

@dataclass
class P1State:
    """P1 神经元状态"""
    activation: float = 0.0          # P1 激活度（0-1）
    target_detected: bool = False     # 是否检测到雌蝇
    target_direction: float = 0.0     # 雌蝇方向（弧度）
    target_distance: float = 999.0    # 雌蝇距离
    courtship_phase: CourtshipPhase = CourtshipPhase.IDLE
    song_intensity: float = 0.0       # 求偶之歌强度
    song_frequency: float = 220.0     # 求偶之歌频率（Hz，160-300）
    mating_progress: float = 0.0      # 交配进度（0-1）
    refractory_until: float = 0.0     # 不应期结束时间


class P1CourtshipCenter:
    """
    雄蝇 P1 求偶指挥官神经元

    真实果蝇中，P1 神经元是雄蝇求偶行为的核心中枢：
      - 接收雌蝇信息素输入（触角叶→蘑菇体→P1）
      - 激活后触发完整的求偶序列：定位→追逐→触碰→唱歌→舔舐→交配
      - 单侧翅膀振动产生求偶之歌（160-300Hz）
    """

    def __init__(self):
        self.state = P1State()
        self._rng = random.Random()
        self._sim_time = 0.0  # 模拟时间（dt 累加）
        # 求偶序列的时间参数（秒，模拟时间）
        self.phase_durations = {
            CourtshipPhase.ORIENTING: 0.5,
            CourtshipPhase.CHASING: 2.0,
            CourtshipPhase.TAPPING: 0.8,
            CourtshipPhase.SINGING: 3.0,
            CourtshipPhase.LICKING: 1.0,
            CourtshipPhase.ATTEMPTING_MOUNT: 1.5,
            CourtshipPhase.MATING: 20.0,
            CourtshipPhase.POST_MATING: 30.0,  # 不应期
        }
        self._phase_start_time = 0.0

    def update(self, dt: float, pheromone_conc: float, pheromone_direction: float,
               target_distance: float, female_receptive: bool, female_position: tuple = None):
        """
        更新 P1 求偶中枢

        Args:
            dt: 时间步长（秒）
            pheromone_conc: 雌蝇信息素浓度（0-1）
            pheromone_direction: 雌蝇方向（弧度）
            target_distance: 雌蝇距离（像素）
            female_receptive: 雌蝇是否接受交配
            female_position: 雌蝇位置（可选，用于精确追逐）
        """
        s = self.state
        self._sim_time += dt
        now = self._sim_time

        # 不应期检查
        if now < s.refractory_until:
            s.courtship_phase = CourtshipPhase.POST_MATING
            s.activation = 0.0
            s.song_intensity = 0.0
            return

        # P1 激活度由信息素浓度决定
        s.activation = min(1.0, pheromone_conc * 1.5)
        # 带滞后的目标检测（防止阈值附近反复跳变）
        if s.target_detected:
            s.target_detected = pheromone_conc > 0.06  # 退出阈值更低
        else:
            s.target_detected = pheromone_conc > 0.12  # 进入阈值更高
        s.target_direction = pheromone_direction
        s.target_distance = target_distance

        if not s.target_detected:
            # 没有检测到雌蝇，回到空闲
            if s.courtship_phase not in (CourtshipPhase.IDLE, CourtshipPhase.POST_MATING):
                s.courtship_phase = CourtshipPhase.IDLE
                self._phase_start_time = now
            s.song_intensity = 0.0
            return

        # 求偶状态机
        if s.courtship_phase == CourtshipPhase.IDLE:
            # 开始求偶：定位
            s.courtship_phase = CourtshipPhase.ORIENTING
            self._phase_start_time = now

        elif s.courtship_phase == CourtshipPhase.ORIENTING:
            # 定位阶段：转向雌蝇
            if now - self._phase_start_time > self.phase_durations[CourtshipPhase.ORIENTING]:
                s.courtship_phase = CourtshipPhase.CHASING
                self._phase_start_time = now

        elif s.courtship_phase == CourtshipPhase.CHASING:
            # 追逐阶段：靠近雌蝇
            if target_distance < 50:
                # 足够近了，进入触碰
                s.courtship_phase = CourtshipPhase.TAPPING
                self._phase_start_time = now
            elif now - self._phase_start_time > self.phase_durations[CourtshipPhase.CHASING]:
                # 追了太久还没追上，重新定位
                s.courtship_phase = CourtshipPhase.ORIENTING
                self._phase_start_time = now

        elif s.courtship_phase == CourtshipPhase.TAPPING:
            # 触碰阶段：前足敲击雌蝇
            s.song_intensity = 0.0
            if now - self._phase_start_time > self.phase_durations[CourtshipPhase.TAPPING]:
                s.courtship_phase = CourtshipPhase.SINGING
                self._phase_start_time = now
                # 随机生成求偶之歌频率（160-300Hz，个体差异）
                s.song_frequency = self._rng.uniform(180, 280)

        elif s.courtship_phase == CourtshipPhase.SINGING:
            # 唱歌阶段：单侧翅膀振动
            s.song_intensity = min(1.0, s.activation * 1.2)
            if now - self._phase_start_time > self.phase_durations[CourtshipPhase.SINGING]:
                if female_receptive:
                    # 雌蝇被打动，进入舔舐
                    s.courtship_phase = CourtshipPhase.LICKING
                    self._phase_start_time = now
                else:
                    # 雌蝇没被打动，重新唱歌（换频率）
                    s.song_frequency = self._rng.uniform(180, 280)
                    self._phase_start_time = now

        elif s.courtship_phase == CourtshipPhase.LICKING:
            # 舔舐阶段
            s.song_intensity = 0.3
            if now - self._phase_start_time > self.phase_durations[CourtshipPhase.LICKING]:
                s.courtship_phase = CourtshipPhase.ATTEMPTING_MOUNT
                self._phase_start_time = now

        elif s.courtship_phase == CourtshipPhase.ATTEMPTING_MOUNT:
            # 尝试交配
            if female_receptive and target_distance < 30:
                s.courtship_phase = CourtshipPhase.MATING
                self._phase_start_time = now
                s.mating_progress = 0.0
            elif now - self._phase_start_time > self.phase_durations[CourtshipPhase.ATTEMPTING_MOUNT]:
                # 尝试失败，回到唱歌
                s.courtship_phase = CourtshipPhase.SINGING
                self._phase_start_time = now

        elif s.courtship_phase == CourtshipPhase.MATING:
            # 交配中
            s.song_intensity = 0.0
            s.mating_progress = min(1.0, (now - self._phase_start_time) / self.phase_durations[CourtshipPhase.MATING])
            if s.mating_progress >= 1.0:
                # 交配完成，进入不应期
                s.courtship_phase = CourtshipPhase.POST_MATING
                s.refractory_until = now + self.phase_durations[CourtshipPhase.POST_MATING]
                s.mating_progress = 0.0

    def get_behavior_modifiers(self) -> dict:
        """
        获取求偶行为对运动的调制

        Returns:
            dict: {
                'walk_drive_mod': 行走速度调制,
                'turn_bias_mod': 转向偏置调制,
                'wing_drive_mod': 翅膀振动调制,
                'is_mating': 是否在交配,
                'courtship_phase': 求偶阶段,
            }
        """
        s = self.state
        modifiers = {
            'walk_drive_mod': 1.0,
            'turn_bias_mod': 0.0,
            'wing_drive_mod': 0.0,
            'is_mating': False,
            'courtship_phase': s.courtship_phase.value,
            'song_frequency': s.song_frequency,
            'song_intensity': s.song_intensity,
            'p1_activation': s.activation,
        }

        if s.courtship_phase == CourtshipPhase.ORIENTING:
            # 定位：转向雌蝇，缓慢移动
            modifiers['walk_drive_mod'] = 0.3
            modifiers['turn_bias_mod'] = self._angle_to_turn(s.target_direction)
        elif s.courtship_phase == CourtshipPhase.CHASING:
            # 追逐：快速冲向雌蝇
            modifiers['walk_drive_mod'] = 1.5
            modifiers['turn_bias_mod'] = self._angle_to_turn(s.target_direction) * 0.8
        elif s.courtship_phase == CourtshipPhase.TAPPING:
            # 触碰：停在雌蝇旁边，轻微振翅
            modifiers['walk_drive_mod'] = 0.1
            modifiers['wing_drive_mod'] = 0.3
        elif s.courtship_phase == CourtshipPhase.SINGING:
            # 唱歌：单侧翅膀振动，缓慢侧移
            modifiers['walk_drive_mod'] = 0.2
            modifiers['wing_drive_mod'] = s.song_intensity * 1.5
            modifiers['turn_bias_mod'] = 0.2  # 侧身展示
        elif s.courtship_phase == CourtshipPhase.LICKING:
            # 舔舐：停在雌蝇腹部
            modifiers['walk_drive_mod'] = 0.05
            modifiers['wing_drive_mod'] = 0.2
        elif s.courtship_phase == CourtshipPhase.ATTEMPTING_MOUNT:
            # 尝试交配：靠近
            modifiers['walk_drive_mod'] = 0.8
            modifiers['turn_bias_mod'] = self._angle_to_turn(s.target_direction)
        elif s.courtship_phase == CourtshipPhase.MATING:
            # 交配中：完全不动
            modifiers['walk_drive_mod'] = 0.0
            modifiers['turn_bias_mod'] = 0.0
            modifiers['is_mating'] = True

        return modifiers

    def _angle_to_turn(self, angle: float) -> float:
        """将方向角转换为 turn_bias（-1到1）"""
        # 简化：角度差映射到转向
        normalized = (angle + math.pi) % (2 * math.pi) - math.pi
        return max(-1.0, min(1.0, normalized / 1.5))

    def reset(self):
        """重置 P1 状态"""
        self.state = P1State()
        self._sim_time = 0.0


# ============================================================
# 雌蝇 pC1 接纳中枢
# ============================================================

@dataclass
class PC1State:
    """pC1 神经元状态"""
    activation: float = 0.0          # pC1 激活度（0-1）
    male_detected: bool = False       # 是否检测到雄蝇
    response: FemaleResponse = FemaleResponse.UNAWARE
    evaluation_score: float = 0.0     # 对雄蝇的评估分（0-1）
    song_quality: float = 0.0         # 求偶之歌质量（0-1）
    mating_progress: float = 0.0      # 交配进度
    refractory_until: float = 0.0     # 不应期结束时间
    last_song_time: float = 0.0       # 上次听到求偶之歌的时间


class PC1ReceptivityCenter:
    """
    雌蝇 pC1 接纳与考核中枢

    真实果蝇中，pC1 神经元是雌蝇性行为的核心中枢：
      - 接收雄蝇求偶之歌的听觉输入（触角 Johnston 器官→pC1）
      - 评估求偶之歌的频率和节奏
      - 激活后触发接纳行为（停止移动、允许交配）
      - 交配后进入不应期（雌蝇交配后会拒绝其他雄蝇）
    """

    def __init__(self):
        self.state = PC1State()
        self._rng = random.Random()
        self._sim_time = 0.0  # 模拟时间
        # 雌蝇偏好的求偶之歌频率（个体差异，200-260Hz）
        self.preferred_frequency = self._rng.uniform(200, 260)
        self.frequency_tolerance = 30.0  # 频率容差（Hz）

    def update(self, dt: float, pheromone_conc: float, male_song_intensity: float,
               male_song_frequency: float, male_distance: float, is_mating: bool = False):
        """
        更新 pC1 接纳中枢

        Args:
            dt: 时间步长（秒）
            pheromone_conc: 雄蝇信息素浓度（0-1）
            male_song_intensity: 雄蝇求偶之歌强度（0-1）
            male_song_frequency: 雄蝇求偶之歌频率（Hz）
            male_distance: 雄蝇距离（像素）
            is_mating: 是否正在交配（由雄蝇状态同步）
        """
        s = self.state
        self._sim_time += dt
        now = self._sim_time

        # 不应期检查
        if now < s.refractory_until:
            s.response = FemaleResponse.POST_MATING
            s.activation = 0.0
            return

        # 检测雄蝇（带滞后）
        if s.male_detected:
            s.male_detected = pheromone_conc > 0.06 or male_song_intensity > 0.06
        else:
            s.male_detected = pheromone_conc > 0.12 or male_song_intensity > 0.12

        if not s.male_detected:
            if s.response not in (FemaleResponse.UNAWARE, FemaleResponse.POST_MATING):
                s.response = FemaleResponse.UNAWARE
            s.activation = 0.0
            s.evaluation_score = 0.0
            return

        # 评估求偶之歌
        if male_song_intensity > 0.1:
            s.last_song_time = now
            # 频率匹配度（高斯函数）
            freq_diff = abs(male_song_frequency - self.preferred_frequency)
            freq_match = math.exp(-0.5 * (freq_diff / self.frequency_tolerance) ** 2)
            # 歌曲质量 = 强度 × 频率匹配度
            s.song_quality = male_song_intensity * freq_match
            # 评估分随时间积累（听越久越了解）
            s.evaluation_score = min(1.0, s.evaluation_score + s.song_quality * dt * 0.3)
        else:
            # 没在唱歌，评估分缓慢衰减
            s.evaluation_score = max(0.0, s.evaluation_score - dt * 0.05)

        # pC1 激活度 = 评估分 × 信息素浓度
        s.activation = min(1.0, s.evaluation_score * pheromone_conc * 1.5)

        # 雌蝇回应状态机
        if s.response == FemaleResponse.UNAWARE:
            if s.male_detected:
                s.response = FemaleResponse.DETECTED

        elif s.response == FemaleResponse.DETECTED:
            if male_song_intensity > 0.2:
                s.response = FemaleResponse.EVALUATING

        elif s.response == FemaleResponse.EVALUATING:
            if s.evaluation_score > 0.6 and male_distance < 50:
                # 被打动了，接受交配
                s.response = FemaleResponse.RECEPTIVE
            elif now - s.last_song_time > 5.0:
                # 雄蝇不唱了，回到检测状态
                s.response = FemaleResponse.DETECTED
                s.evaluation_score *= 0.5

        elif s.response == FemaleResponse.RECEPTIVE:
            # 接受状态：如果雄蝇开始交配，进入交配
            if is_mating:
                s.response = FemaleResponse.POST_MATING
                s.mating_progress = 0.0
                s.refractory_until = now + 60.0  # 雌蝇不应期更长（60秒）
            elif s.evaluation_score < 0.3:
                # 评估分下降，回到评估
                s.response = FemaleResponse.EVALUATING

        # 交配进度（如果在交配）
        if is_mating and s.response == FemaleResponse.POST_MATING:
            s.mating_progress = min(1.0, s.mating_progress + dt / 20.0)

    def is_receptive(self) -> bool:
        """是否接受交配"""
        return self.state.response == FemaleResponse.RECEPTIVE

    def get_behavior_modifiers(self) -> dict:
        """
        获取雌蝇回应行为对运动的调制

        Returns:
            dict: 行为调制参数
        """
        s = self.state
        modifiers = {
            'walk_drive_mod': 1.0,
            'turn_bias_mod': 0.0,
            'wing_drive_mod': 0.0,
            'is_receptive': self.is_receptive(),
            'response': s.response.value,
            'pc1_activation': s.activation,
            'evaluation_score': s.evaluation_score,
            'song_quality': s.song_quality,
        }

        if s.response == FemaleResponse.EVALUATING:
            # 评估中：缓慢移动，倾听
            modifiers['walk_drive_mod'] = 0.4
        elif s.response == FemaleResponse.RECEPTIVE:
            # 接受：停止移动，等待交配
            modifiers['walk_drive_mod'] = 0.05
            modifiers['wing_drive_mod'] = 0.1  # 轻微振翅表示接受
        elif s.response == FemaleResponse.REJECTING:
            # 拒绝：快速离开，踢开
            modifiers['walk_drive_mod'] = 1.5
            modifiers['turn_bias_mod'] = 1.0  # 急转弯离开

        return modifiers

    def reset(self):
        """重置 pC1 状态"""
        self.state = PC1State()
        self._sim_time = 0.0
        self.preferred_frequency = self._rng.uniform(200, 260)


# ============================================================
# 求偶协调器（双果蝇交互）
# ============================================================

class CourtshipCoordinator:
    """
    求偶协调器 - 管理雌雄双蝇的求偶交互

    职责：
      - 维护信息素场
      - 同步雌雄蝇的求偶状态
      - 检测交配完成事件
      - 提供繁衍触发接口
    """

    def __init__(self):
        self.pheromone_field = PheromoneField()
        self.male_source = PheromoneSource(gender=Gender.MALE)
        self.female_source = PheromoneSource(gender=Gender.FEMALE)
        self.pheromone_field.add_source(self.male_source)
        self.pheromone_field.add_source(self.female_source)

        self.male_p1 = P1CourtshipCenter()
        self.female_pc1 = PC1ReceptivityCenter()

        self.mating_completed = False
        self.on_mating_complete = None  # 回调函数

    def update(self, dt: float,
               male_pos: tuple, female_pos: tuple,
               male_heading: float = 0.0, female_heading: float = 0.0):
        """
        更新求偶协调器

        Args:
            dt: 时间步长（秒）
            male_pos: 雄蝇位置 (x, y)
            female_pos: 雌蝇位置 (x, y)
            male_heading: 雄蝇朝向（弧度）
            female_heading: 雌蝇朝向（弧度）
        """
        # 更新信息素源位置
        self.male_source.x, self.male_source.y = male_pos
        self.female_source.x, self.female_source.y = female_pos

        # 计算距离
        dist = math.hypot(male_pos[0] - female_pos[0], male_pos[1] - female_pos[1])

        # 雄蝇检测雌蝇信息素
        female_pheromone = self.pheromone_field.get_concentration(
            male_pos[0], male_pos[1], Gender.MALE
        )
        female_direction, female_conc = self.pheromone_field.get_strongest_source_direction(
            male_pos[0], male_pos[1], Gender.MALE
        )

        # 雌蝇检测雄蝇信息素
        male_pheromone = self.pheromone_field.get_concentration(
            female_pos[0], female_pos[1], Gender.FEMALE
        )

        # 更新雄蝇 P1
        self.male_p1.update(
            dt=dt,
            pheromone_conc=female_pheromone,
            pheromone_direction=female_direction,
            target_distance=dist,
            female_receptive=self.female_pc1.is_receptive(),
            female_position=female_pos,
        )

        # 获取雄蝇求偶之歌状态
        male_mods = self.male_p1.get_behavior_modifiers()
        song_intensity = male_mods['song_intensity']
        song_frequency = male_mods['song_frequency']

        # 更新雌蝇 pC1
        is_mating = male_mods['is_mating']
        self.female_pc1.update(
            dt=dt,
            pheromone_conc=male_pheromone,
            male_song_intensity=song_intensity,
            male_song_frequency=song_frequency,
            male_distance=dist,
            is_mating=is_mating,
        )

        # 检测交配完成
        if (self.male_p1.state.courtship_phase == CourtshipPhase.POST_MATING and
                self.female_pc1.state.response == FemaleResponse.POST_MATING and
                not self.mating_completed):
            self.mating_completed = True
            if self.on_mating_complete:
                self.on_mating_complete()

    def get_male_modifiers(self) -> dict:
        """获取雄蝇行为调制"""
        return self.male_p1.get_behavior_modifiers()

    def get_female_modifiers(self) -> dict:
        """获取雌蝇行为调制"""
        return self.female_pc1.get_behavior_modifiers()

    def get_status(self) -> dict:
        """获取求偶状态摘要"""
        return {
            'distance': math.hypot(
                self.male_source.x - self.female_source.x,
                self.male_source.y - self.female_source.y
            ),
            'male_phase': self.male_p1.state.courtship_phase.value,
            'male_p1': self.male_p1.state.activation,
            'male_song_freq': self.male_p1.state.song_frequency,
            'male_song_intensity': self.male_p1.state.song_intensity,
            'female_response': self.female_pc1.state.response.value,
            'female_pc1': self.female_pc1.state.activation,
            'female_eval': self.female_pc1.state.evaluation_score,
            'female_pref_freq': self.female_pc1.preferred_frequency,
            'mating_completed': self.mating_completed,
        }

    def reset(self):
        """重置求偶协调器"""
        self.male_p1.reset()
        self.female_pc1.reset()
        self.mating_completed = False
