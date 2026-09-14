#!/usr/bin/env python3
"""
桌面环境模块 - 让果蝇和真实桌面环境发生物理交互

包含：
  - 碰撞检测：果蝇撞到窗口边缘会反弹、减速
  - 食物源：特定窗口（终端）是食物，爬到上面恢复能量
  - 鼠标捕食者：鼠标靠近触发逃跑反应
  - 感官输入：视觉（窗口边缘距离）、嗅觉（窗口气味）、触觉（碰撞）

坐标系统：
  - 屏幕坐标：左上角为原点，y 向下（WindowDetector 使用）
  - DesktopFly 坐标：屏幕中心为原点，y 向上（fly_vision.json 使用）
  - 本模块内部统一使用屏幕坐标计算，输出时转换
"""

import os
import sys
import time
import json
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

from window_detector import WindowDetector, WindowInfo, fly_to_screen, screen_to_fly


# 共享文件路径
VISION_FILE = "/tmp/fly_vision.json"

# 食物源应用（这些窗口是果蝇的"食物"，爬到上面恢复能量）
FOOD_APPS = {"Terminal", "iTerm2", "WezTerm", "Alacritty"}

# 危险应用（这些窗口对果蝇有"威胁"）
DANGER_APPS = {"Activity Monitor", "Task Manager"}

# 鼠标捕食者参数
PREDATOR_DANGER_DISTANCE = 150   # 鼠标距离小于这个值触发警觉
PREDATOR_ESCAPE_DISTANCE = 80    # 鼠标距离小于这个值触发逃跑
PREDATOR_SPEED_THRESHOLD = 500    # 鼠标移动速度大于这个值触发警觉


@dataclass
class EnvironmentState:
    """环境状态 - 每帧更新"""
    # 果蝇位置（屏幕坐标）
    fly_x: int = 0
    fly_y: int = 0
    fly_heading: float = 0.0  # 朝向（弧度）

    # 碰撞
    is_colliding: bool = False
    collision_normal_x: float = 0.0  # 碰撞法线（用于反弹）
    collision_normal_y: float = 0.0
    collision_window: Optional[str] = None

    # 食物
    on_food: bool = False
    food_window: Optional[str] = None
    food_recovery_rate: float = 0.0  # 能量恢复速率

    # 危险
    in_danger: bool = False
    danger_window: Optional[str] = None

    # 鼠标捕食者
    mouse_distance: float = 9999.0
    mouse_bearing: float = 0.0
    mouse_speed: float = 0.0
    predator_alert: bool = False      # 警觉（鼠标较近或移动快）
    predator_escape: bool = False     # 逃跑（鼠标很近）

    # 感官输入（用于编码成神经电流）
    visual_obstacle_distance: float = 9999.0  # 前方障碍物距离
    visual_obstacle_bearing: float = 0.0       # 障碍物方位
    olfactory_intensity: float = 0.0            # 气味强度（食物源）
    olfactory_type: str = "none"                # 气味类型
    tactile_force: float = 0.0                  # 触觉强度（碰撞）

    # 行为调整（输出给主循环）
    force_turn: float = 0.0      # 强制转向（-1 到 1，碰撞时使用）
    speed_multiplier: float = 1.0  # 速度倍率（碰撞减速、逃跑加速）
    extra_punishment: float = 0.0  # 额外惩罚（碰撞、危险）
    extra_arousal: float = 0.0     # 额外唤醒（捕食者接近）


class DesktopEnvironment:
    """
    桌面环境 - 管理果蝇和真实桌面的交互

    用法:
        env = DesktopEnvironment()
        env.update(window_detector)
        state = env.state
        # 使用 state.force_turn, state.speed_multiplier 等调整行为
    """

    def __init__(self, vision_file: str = VISION_FILE):
        self.vision_file = vision_file
        self.state = EnvironmentState()
        self.screen_width = 2560
        self.screen_height = 1664
        self._last_mouse_x = 0
        self._last_mouse_y = 0
        self._last_mouse_time = 0.0
        self._last_vision_read = 0.0

    def update(self, detector: WindowDetector, dt: float = 0.033):
        """
        更新环境状态

        Args:
            detector: 窗口检测器（已更新窗口列表）
            dt: 时间步长（秒）
        """
        self.screen_width = detector.screen_width
        self.screen_height = detector.screen_height

        # 1. 读取果蝇位置
        self._read_fly_position()

        # 2. 碰撞检测
        self._check_collision(detector.windows)

        # 3. 食物源检测
        self._check_food(detector.windows)

        # 4. 危险区域检测
        self._check_danger(detector.windows)

        # 5. 鼠标捕食者检测
        self._check_predator()

        # 6. 感官输入计算
        self._compute_senses(detector.windows)

        # 7. 行为调整计算
        self._compute_behavior_adjustments(dt)

    def _read_fly_position(self):
        """从共享文件读取果蝇位置"""
        try:
            with open(self.vision_file, 'r') as f:
                data = json.load(f)
            fly_x = data.get('fly_x', 0.0)
            fly_y = data.get('fly_y', 0.0)
            fly_heading = data.get('fly_heading', 0.0)

            # 转换为屏幕坐标
            sx, sy = fly_to_screen(fly_x, fly_y, self.screen_width, self.screen_height)
            self.state.fly_x = sx
            self.state.fly_y = sy
            self.state.fly_heading = fly_heading

            # 鼠标信息
            self.state.mouse_distance = data.get('mouse_distance', 9999.0)
            self.state.mouse_bearing = data.get('mouse_bearing', 0.0)

            # 计算鼠标速度
            now = time.monotonic()
            if self._last_mouse_time > 0:
                # 从 mouse_distance 和 bearing 反推鼠标位置（近似）
                # 这里简化：用距离变化率近似速度
                dt = now - self._last_mouse_time
                if dt > 0:
                    self.state.mouse_speed = abs(
                        self.state.mouse_distance - self._last_mouse_x
                    ) / dt
            self._last_mouse_x = self.state.mouse_distance
            self._last_mouse_time = now

        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            pass

    def _check_collision(self, windows: List[WindowInfo]):
        """检测果蝇是否撞到窗口边缘"""
        self.state.is_colliding = False
        self.state.collision_normal_x = 0.0
        self.state.collision_normal_y = 0.0
        self.state.collision_window = None

        fly_x, fly_y = self.state.fly_x, self.state.fly_y
        collision_radius = 30  # 果蝇的碰撞半径（像素，增大让碰撞更容易触发）

        for w in windows:
            # 扩展窗口边界（加上果蝇半径）
            left = w.x - collision_radius
            right = w.x + w.width + collision_radius
            top = w.y - collision_radius
            bottom = w.y + w.height + collision_radius

            # 检查果蝇是否在扩展边界内（即撞到窗口边缘）
            if left <= fly_x <= right and top <= fly_y <= bottom:
                # 计算到最近边缘的距离，确定碰撞法线
                dist_left = fly_x - left
                dist_right = right - fly_x
                dist_top = fly_y - top
                dist_bottom = bottom - fly_y

                min_dist = min(dist_left, dist_right, dist_top, dist_bottom)

                if min_dist == dist_left:
                    self.state.collision_normal_x = -1.0
                    self.state.collision_normal_y = 0.0
                elif min_dist == dist_right:
                    self.state.collision_normal_x = 1.0
                    self.state.collision_normal_y = 0.0
                elif min_dist == dist_top:
                    self.state.collision_normal_x = 0.0
                    self.state.collision_normal_y = -1.0
                else:
                    self.state.collision_normal_x = 0.0
                    self.state.collision_normal_y = 1.0

                self.state.is_colliding = True
                self.state.collision_window = w.app_name
                break  # 只处理第一个碰撞

    def _check_food(self, windows: List[WindowInfo]):
        """检测果蝇是否在食物源上"""
        self.state.on_food = False
        self.state.food_window = None
        self.state.food_recovery_rate = 0.0

        fly_x, fly_y = self.state.fly_x, self.state.fly_y

        for w in windows:
            if w.app_name in FOOD_APPS and w.contains_point(fly_x, fly_y):
                self.state.on_food = True
                self.state.food_window = w.app_name
                # 终端窗口是"家"，能量恢复速率较高
                self.state.food_recovery_rate = 0.05  # 每秒恢复 5%
                break

    def _check_danger(self, windows: List[WindowInfo]):
        """检测果蝇是否在危险区域"""
        self.state.in_danger = False
        self.state.danger_window = None

        fly_x, fly_y = self.state.fly_x, self.state.fly_y

        for w in windows:
            if w.app_name in DANGER_APPS and w.contains_point(fly_x, fly_y):
                self.state.in_danger = True
                self.state.danger_window = w.app_name
                break

    def _check_predator(self):
        """检测鼠标捕食者"""
        self.state.predator_alert = False
        self.state.predator_escape = False

        dist = self.state.mouse_distance
        speed = self.state.mouse_speed

        # 鼠标很近或移动很快 → 警觉
        if dist < PREDATOR_DANGER_DISTANCE or speed > PREDATOR_SPEED_THRESHOLD:
            self.state.predator_alert = True

        # 鼠标非常近 → 逃跑
        if dist < PREDATOR_ESCAPE_DISTANCE:
            self.state.predator_escape = True

    def _compute_senses(self, windows: List[WindowInfo]):
        """计算感官输入（视觉、嗅觉、触觉）"""
        fly_x, fly_y = self.state.fly_x, self.state.fly_y
        heading = self.state.fly_heading

        # 视觉：前方障碍物距离
        # 从果蝇位置沿朝向方向发射射线，检测最近的窗口
        ray_length = 300
        nearest_dist = ray_length
        nearest_bearing = 0.0

        # 前方方向向量
        fwd_x = math.cos(heading)
        fwd_y = math.sin(heading)  # 注意：屏幕坐标 y 向下，但 heading 是 DesktopFly 的

        for w in windows:
            # 简化：计算到窗口中心的距离和方位
            cx, cy = w.center
            dx = cx - fly_x
            dy = cy - fly_y
            dist = math.hypot(dx, dy)

            if dist < nearest_dist:
                # 计算方位（相对于果蝇朝向）
                bearing = math.atan2(dy, dx) - heading
                # 归一化到 -pi 到 pi
                while bearing > math.pi:
                    bearing -= 2 * math.pi
                while bearing < -math.pi:
                    bearing += 2 * math.pi

                # 只考虑前方 120 度范围内的障碍物
                if abs(bearing) < math.pi * 2 / 3:
                    nearest_dist = dist
                    nearest_bearing = bearing

        self.state.visual_obstacle_distance = nearest_dist
        self.state.visual_obstacle_bearing = nearest_bearing

        # 嗅觉：食物源气味
        if self.state.on_food:
            self.state.olfactory_intensity = 1.0
            self.state.olfactory_type = "food"
        else:
            # 计算到最近食物源的距离，气味随距离衰减
            nearest_food_dist = 9999
            for w in windows:
                if w.app_name in FOOD_APPS:
                    dist = w.distance_to_point(fly_x, fly_y)
                    nearest_food_dist = min(nearest_food_dist, dist)

            if nearest_food_dist < 500:
                self.state.olfactory_intensity = max(0, 1.0 - nearest_food_dist / 500)
                self.state.olfactory_type = "food"
            else:
                self.state.olfactory_intensity = 0.0
                self.state.olfactory_type = "none"

        # 触觉：碰撞强度
        if self.state.is_colliding:
            self.state.tactile_force = 1.0
        else:
            self.state.tactile_force = 0.0

    def _compute_behavior_adjustments(self, dt: float):
        """计算行为调整（转向、速度、惩罚、唤醒）"""
        s = self.state

        # 重置
        s.force_turn = 0.0
        s.speed_multiplier = 1.0
        s.extra_punishment = 0.0
        s.extra_arousal = 0.0

        # 碰撞：反弹 + 减速 + 轻微惩罚
        if s.is_colliding:
            # 计算反射方向（碰撞法线的反射）
            # 简化：转向碰撞法线的反方向
            if s.collision_normal_x != 0 or s.collision_normal_y != 0:
                # 计算法线角度
                normal_angle = math.atan2(s.collision_normal_y, s.collision_normal_x)
                # 转向法线方向（逃离碰撞）
                target_heading = normal_angle
                # 计算当前朝向和目标朝向的差值
                heading_diff = target_heading - s.fly_heading
                # 归一化
                while heading_diff > math.pi:
                    heading_diff -= 2 * math.pi
                while heading_diff < -math.pi:
                    heading_diff += 2 * math.pi
                # 强制转向（归一化到 -1 到 1）
                s.force_turn = max(-1.0, min(1.0, heading_diff / math.pi))

            # 减速（碰撞时几乎停下来）
            s.speed_multiplier = 0.1
            # 惩罚（碰撞不舒服，增强惩罚让果蝇学会回避）
            s.extra_punishment = 0.5
            # 高唤醒（碰撞时警觉）
            s.extra_arousal = 0.6

        # 鼠标捕食者：逃跑 + 加速 + 高唤醒
        if s.predator_escape:
            # 转向鼠标的反方向
            # mouse_bearing 是鼠标相对于果蝇朝向的方位
            # 逃跑方向 = bearing + pi
            escape_bearing = s.mouse_bearing + math.pi
            while escape_bearing > math.pi:
                escape_bearing -= 2 * math.pi
            while escape_bearing < -math.pi:
                escape_bearing += 2 * math.pi
            s.force_turn = max(-1.0, min(1.0, escape_bearing / math.pi))
            # 加速逃跑（增强到 2.5 倍）
            s.speed_multiplier = 2.5
            # 高唤醒
            s.extra_arousal = 1.0
            # 惩罚（害怕）
            s.extra_punishment = 0.5

        elif s.predator_alert:
            # 警觉：稍微加速，高唤醒
            s.speed_multiplier = 1.3
            s.extra_arousal = 0.4

        # 危险区域：加速离开
        if s.in_danger:
            s.speed_multiplier = 1.5
            s.extra_punishment = 0.2
            s.extra_arousal = max(s.extra_arousal, 0.5)

    def get_state_dict(self) -> dict:
        """获取状态字典（用于调试或共享）"""
        s = self.state
        return {
            "fly_x": s.fly_x, "fly_y": s.fly_y,
            "fly_heading": s.fly_heading,
            "is_colliding": s.is_colliding,
            "collision_window": s.collision_window,
            "on_food": s.on_food,
            "food_window": s.food_window,
            "in_danger": s.in_danger,
            "predator_alert": s.predator_alert,
            "predator_escape": s.predator_escape,
            "mouse_distance": s.mouse_distance,
            "visual_obstacle_distance": s.visual_obstacle_distance,
            "olfactory_intensity": s.olfactory_intensity,
            "force_turn": s.force_turn,
            "speed_multiplier": s.speed_multiplier,
            "extra_punishment": s.extra_punishment,
            "extra_arousal": s.extra_arousal,
        }


if __name__ == "__main__":
    # 测试
    detector = WindowDetector()
    detector.update(force=True)

    env = DesktopEnvironment()
    env.update(detector)

    print("=== 桌面环境状态 ===")
    state = env.state
    print(f"果蝇位置: ({state.fly_x}, {state.fly_y})")
    print(f"碰撞: {state.is_colliding} ({state.collision_window})")
    print(f"食物: {state.on_food} ({state.food_window})")
    print(f"危险: {state.in_danger}")
    print(f"鼠标距离: {state.mouse_distance:.1f}")
    print(f"捕食者警觉: {state.predator_alert}, 逃跑: {state.predator_escape}")
    print(f"前方障碍物距离: {state.visual_obstacle_distance:.1f}")
    print(f"气味强度: {state.olfactory_intensity:.2f} ({state.olfactory_type})")
    print(f"强制转向: {state.force_turn:.2f}")
    print(f"速度倍率: {state.speed_multiplier:.2f}")
    print(f"额外惩罚: {state.extra_punishment:.2f}")
    print(f"额外唤醒: {state.extra_arousal:.2f}")
