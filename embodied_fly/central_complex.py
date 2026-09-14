"""
中央复合体 (Central Complex, CX) —— 果蝇导航中枢

简化模型，包含四个核心功能：
1. 方位角编码：8 方向环形吸引子网络（类似指南针）
2. 边界回避：检测果蝇是否靠近屏幕边缘，触发转向
3. 空间记忆：最近访问位置的衰减记忆，避免重复探索
4. 导航指令：输出转向偏置和速度调整，与蘑菇体输出整合

真实果蝇中央复合体包括：椭圆体(EB)、原脑桥(PB)、扇形体(FB)、小结(NO)
这里用简化的计算模型实现核心功能。
"""

import math
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class CXState:
    """中央复合体状态"""
    # 方位角编码（8 个方向的激活强度）
    heading_activation: List[float] = field(default_factory=lambda: [0.0] * 8)
    current_heading: float = 0.0  # 当前估计方位角（弧度）

    # 边界检测
    near_boundary: bool = False
    boundary_direction: float = 0.0  # 边界相对方位（弧度）
    boundary_distance: float = 9999.0  # 到最近边界的距离

    # 空间记忆
    visited_map: dict = field(default_factory=dict)  # 网格位置 → 访问次数
    novelty_at_position: float = 1.0  # 当前位置的新奇度

    # 导航输出
    turn_bias: float = 0.0  # 转向偏置 (-1 到 1)
    speed_modulation: float = 1.0  # 速度调整 (0.5 到 1.5)
    exploration_drive: float = 0.5  # 探索驱动力 (0 到 1)

    # 内部状态
    last_position: Tuple[float, float] = (0.0, 0.0)
    path_integral: Tuple[float, float] = (0.0, 0.0)  # 路径积分估计位置
    last_update_time: float = 0.0


class CentralComplex:
    """
    中央复合体导航中枢

    输入：
      - fly_position: 果蝇在屏幕上的位置 (x, y)
      - fly_heading: 果蝇朝向（弧度）
      - screen_bounds: 屏幕边界 (min_x, max_x, min_y, max_y)
      - mushroom_body_output: 蘑菇体输出（趋向/回避倾向）
      - visual_input: 视觉输入（looming 等）

    输出：
      - turn_bias: 转向偏置
      - speed_modulation: 速度调整
      - exploration_drive: 探索驱动力
    """

    def __init__(self, grid_size: float = 80.0, boundary_margin: float = 120.0):
        """
        Args:
            grid_size: 空间记忆网格大小（像素）
            boundary_margin: 边界检测边距（像素），小于此距离视为靠近边界
        """
        self.state = CXState()
        self.grid_size = grid_size
        self.boundary_margin = boundary_margin
        self._heading_bump_width = 1.5  # 方位角激活峰宽度（弧度）

    def update(
        self,
        fly_position: Tuple[float, float],
        fly_heading: float,
        screen_bounds: Tuple[float, float, float, float],
        approach: float = 0.0,
        avoidance: float = 0.0,
        looming_strength: float = 0.0,
        novelty: float = 0.0,
    ) -> CXState:
        """
        更新中央复合体状态

        Args:
            fly_position: 果蝇位置 (x, y)
            fly_heading: 果蝇朝向（弧度）
            screen_bounds: (min_x, max_x, min_y, max_y)
            approach: 蘑菇体趋向倾向 (0-1)
            avoidance: 蘑菇体回避倾向 (0-1)
            looming_strength: 视觉逼近强度 (0-1)
            novelty: 环境新奇度 (0-1)

        Returns:
            更新后的 CXState
        """
        now = time.monotonic()
        dt = now - self.state.last_update_time if self.state.last_update_time > 0 else 0.05
        self.state.last_update_time = now

        # 1. 方位角编码（环形吸引子）
        self._update_heading(fly_heading, dt)

        # 2. 边界检测
        self._detect_boundary(fly_position, screen_bounds)

        # 3. 空间记忆与路径积分
        self._update_spatial_memory(fly_position, dt)

        # 4. 导航决策
        self._compute_navigation(
            approach=approach,
            avoidance=avoidance,
            looming_strength=looming_strength,
            novelty=novelty,
            dt=dt,
        )

        return self.state

    def _update_heading(self, fly_heading: float, dt: float):
        """更新方位角编码（8 方向环形吸引子网络）"""
        # 用果蝇当前朝向作为方位角输入（简化版，真实果蝇用视觉地标校准）
        self.state.current_heading = fly_heading

        # 计算 8 个方向的激活强度（高斯峰）
        for i in range(8):
            preferred_heading = i * (2 * math.pi / 8)
            # 环形距离（最短角度差）
            angle_diff = abs(((fly_heading - preferred_heading + math.pi) % (2 * math.pi)) - math.pi)
            self.state.heading_activation[i] = math.exp(-(angle_diff ** 2) / (2 * self._heading_bump_width ** 2))

    def _detect_boundary(self, position: Tuple[float, float], bounds: Tuple[float, float, float, float]):
        """检测是否靠近屏幕边界"""
        x, y = position
        min_x, max_x, min_y, max_y = bounds

        # 计算到各边界的距离
        dist_left = x - min_x
        dist_right = max_x - x
        dist_top = y - min_y
        dist_bottom = max_y - y

        # 找到最近的边界
        distances = [
            (dist_left, math.pi),       # 左边界在果蝇的左边（方位角 pi）
            (dist_right, 0.0),          # 右边界在果蝇的右边（方位角 0）
            (dist_top, -math.pi / 2),   # 上边界
            (dist_bottom, math.pi / 2), # 下边界
        ]

        min_dist, boundary_dir = min(distances, key=lambda d: d[0])
        self.state.boundary_distance = min_dist
        self.state.boundary_direction = boundary_dir
        self.state.near_boundary = min_dist < self.boundary_margin

    def _update_spatial_memory(self, position: Tuple[float, float], dt: float):
        """更新空间记忆与路径积分"""
        x, y = position
        last_x, last_y = self.state.last_position

        # 路径积分（估计位移）
        dx = x - last_x
        dy = y - last_y
        px, py = self.state.path_integral
        self.state.path_integral = (px + dx, py + dy)

        # 更新访问地图（网格离散化）
        grid_x = int(x / self.grid_size)
        grid_y = int(y / self.grid_size)
        grid_key = (grid_x, grid_y)

        # 衰减所有记忆
        decay = math.exp(-dt * 0.05)  # 时间常数 ~20 秒
        for key in list(self.state.visited_map.keys()):
            self.state.visited_map[key] *= decay
            if self.state.visited_map[key] < 0.01:
                del self.state.visited_map[key]

        # 增加当前位置访问次数
        self.state.visited_map[grid_key] = self.state.visited_map.get(grid_key, 0.0) + 1.0

        # 计算当前位置新奇度（访问次数越少越新奇）
        visits = self.state.visited_map.get(grid_key, 0.0)
        self.state.novelty_at_position = 1.0 / (1.0 + visits * 0.3)

        self.state.last_position = position

    def _compute_navigation(
        self,
        approach: float,
        avoidance: float,
        looming_strength: float,
        novelty: float,
        dt: float,
    ):
        """计算导航指令"""
        turn = 0.0
        speed_mod = 1.0
        exploration = 0.5

        # 1. 边界回避：靠近边界时强烈转向
        if self.state.near_boundary:
            boundary_proximity = 1.0 - (self.state.boundary_distance / self.boundary_margin)
            # 转向方向：远离边界（边界方向 + pi）
            escape_direction = self.state.boundary_direction + math.pi
            # 转换为相对当前朝向的转向偏置
            angle_diff = ((escape_direction - self.state.current_heading + math.pi) % (2 * math.pi)) - math.pi
            turn += angle_diff * boundary_proximity * 2.0
            speed_mod *= 0.6  # 靠近边界时减速
            exploration = 0.2  # 停止探索，优先逃离边界

        # 2. 空间记忆驱动：在熟悉区域增加探索驱动力
        if self.state.novelty_at_position < 0.3:
            # 在熟悉区域，增加探索驱动力，倾向于转向新区域
            exploration = min(1.0, exploration + 0.4)
            # 随机转向（寻找新区域）
            turn += math.sin(time.monotonic() * 0.7) * 0.3
        else:
            # 在新区域，减速探索
            exploration = max(0.2, exploration - 0.2)
            speed_mod *= 0.85

        # 3. 蘑菇体输出整合
        if avoidance > 0.5:
            # 回避状态：快速转向，远离威胁
            turn += (avoidance - 0.5) * 1.0 * math.sin(time.monotonic() * 2.0)
            speed_mod *= 1.2  # 加速逃离
            exploration = 0.1
        elif approach > 0.7:
            # 趋向状态：稳定前进
            speed_mod *= 1.1
            exploration = 0.3

        # 4. 视觉逼近：优先躲避
        if looming_strength > 0.3:
            exploration = 0.05
            speed_mod *= 1.3

        # 5. 环境新奇度：新窗口/新刺激触发探索
        if novelty > 0.3:
            exploration = min(1.0, exploration + novelty * 0.5)
            turn += math.sin(time.monotonic() * 1.3) * novelty * 0.4

        # 限制输出范围
        self.state.turn_bias = max(-1.0, min(1.0, turn))
        self.state.speed_modulation = max(0.3, min(1.8, speed_mod))
        self.state.exploration_drive = max(0.0, min(1.0, exploration))

    def get_navigation_command(self) -> Tuple[float, float, float]:
        """获取导航指令 (turn_bias, speed_modulation, exploration_drive)"""
        return (
            self.state.turn_bias,
            self.state.speed_modulation,
            self.state.exploration_drive,
        )

    def reset(self):
        """重置中央复合体状态"""
        self.state = CXState()
