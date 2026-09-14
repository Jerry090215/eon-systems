#!/usr/bin/env python3
"""
桌面果蝇感官输入层 (Embodied Fly Sensors)

监控桌面物理变量，作为果蝇的感官输入：
  - 鼠标轨迹 → 嗅觉（移动模式编码不同气味）
  - 系统音量 → 惩罚/电击（音量突变=强刺激）
  - 屏幕亮度 → 光线（昼夜节律，可选）
  - 键盘事件 → 背景噪音/触觉

所有监控采用轮询方式，不需要辅助功能权限。
"""

import asyncio
import json
import math
import os
import subprocess
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional, Tuple

import Quartz


# ============== 数据结构 ==============

@dataclass
class MouseState:
    """鼠标状态与运动学特征"""
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0          # 速度 x (px/s)
    vy: float = 0.0          # 速度 y (px/s)
    speed: float = 0.0       # 速度大小 (px/s)
    acceleration: float = 0.0  # 加速度大小 (px/s²)
    curvature: float = 0.0   # 轨迹曲率 (1/px)
    direction_change: float = 0.0  # 方向变化率 (rad/s)
    trajectory_entropy: float = 0.0  # 轨迹熵（0=直线，1=完全随机）
    is_moving: bool = False
    stationary_time: float = 0.0  # 静止持续时间 (s)
    history: Deque[Tuple[float, float, float]] = field(default_factory=lambda: deque(maxlen=30))
    # history: (timestamp, x, y)


@dataclass
class VolumeState:
    """系统音量状态"""
    volume: float = 0.0       # 当前音量 (0-100)
    prev_volume: float = 0.0   # 上一次音量
    delta: float = 0.0         # 音量变化量
    is_muted: bool = False
    spike: bool = False         # 音量突变（>20% 增量）
    spike_magnitude: float = 0.0  # 突变幅度


@dataclass
class KeyboardState:
    """键盘状态"""
    shift: bool = False
    command: bool = False
    control: bool = False
    option: bool = False
    caps_lock: bool = False
    any_modifier: bool = False
    key_press_rate: float = 0.0  # 按键频率（次/秒，估算）
    typing_burst: bool = False   # 打字爆发（高频按键）
    burst_intensity: float = 0.0  # 爆发强度 0-1


@dataclass
class BrightnessState:
    """屏幕亮度状态"""
    brightness: float = 0.5    # 亮度 (0-1)
    available: bool = False     # 是否可用
    source: str = "unknown"     # 亮度来源：ioreg / time / unknown
    delta: float = 0.0          # 亮度变化量


@dataclass
class WindowState:
    """窗口状态（新奇刺激）"""
    frontmost_app: str = ""      # 当前前台应用名称
    prev_frontmost_app: str = ""  # 上一个前台应用
    app_switched: bool = False    # 是否刚切换了应用
    window_count: int = 0         # 当前窗口数量
    prev_window_count: int = 0    # 上一次窗口数量
    window_count_changed: bool = False  # 窗口数量是否变化
    novelty: float = 0.0          # 新奇度 0-1（应用切换+窗口变化）
    last_switch_time: float = 0.0  # 上次切换时间


@dataclass
class VisionState:
    """视觉状态（从 DesktopFly 读取）"""
    loom_left: float = 0.0       # 左眼逼近强度 0-1
    loom_right: float = 0.0      # 右眼逼近强度 0-1
    loom_total: float = 0.0      # 总逼近强度
    air_puff: float = 0.0        # 气流强度（鼠标快速移动）
    mouse_distance: float = 9999.0  # 鼠标距离果蝇的距离（px）
    mouse_bearing: float = 0.0   # 鼠标相对果蝇的方位（弧度）
    fly_x: float = 0.0           # 果蝇位置 x
    fly_y: float = 0.0           # 果蝇位置 y
    fly_heading: float = 0.0     # 果蝇朝向（弧度）
    available: bool = False       # 是否有新鲜数据
    looming: bool = False         # 是否有逼近刺激（>0.3）


@dataclass
class SensorData:
    """聚合的感官数据"""
    mouse: MouseState = field(default_factory=MouseState)
    volume: VolumeState = field(default_factory=VolumeState)
    keyboard: KeyboardState = field(default_factory=KeyboardState)
    brightness: BrightnessState = field(default_factory=BrightnessState)
    window: WindowState = field(default_factory=WindowState)
    vision: VisionState = field(default_factory=VisionState)
    timestamp: float = 0.0
    dt: float = 0.0  # 距上次更新的时间差


# ============== 鼠标监控 ==============

class MouseSensor:
    """鼠标轨迹监控，提取运动学特征作为嗅觉编码"""

    def __init__(self, sample_rate: float = 30.0):
        self.sample_rate = sample_rate
        self.state = MouseState()
        self._last_time = time.monotonic()
        self._last_speed = 0.0
        self._last_direction = 0.0

    def update(self) -> MouseState:
        """读取鼠标位置并更新运动学特征"""
        now = time.monotonic()
        dt = now - self._last_time
        if dt < 1.0 / self.sample_rate * 0.5:
            return self.state

        # 读取鼠标位置
        loc = Quartz.CGEventGetLocation(Quartz.CGEventCreate(None))
        x, y = loc.x, loc.y

        # 计算速度
        if self.state.history:
            last_t, last_x, last_y = self.state.history[-1]
            if dt > 0:
                vx = (x - last_x) / dt
                vy = (y - last_y) / dt
            else:
                vx, vy = 0, 0
            speed = math.sqrt(vx * vx + vy * vy)
        else:
            vx, vy, speed = 0, 0, 0

        # 计算加速度
        acceleration = (speed - self._last_speed) / dt if dt > 0 else 0

        # 计算方向和方向变化率
        if speed > 10:  # 只有在移动时计算方向
            direction = math.atan2(vy, vx)
            if self._last_direction != 0:
                delta_dir = direction - self._last_direction
                # 归一化到 [-pi, pi]
                delta_dir = (delta_dir + math.pi) % (2 * math.pi) - math.pi
                direction_change = abs(delta_dir) / dt if dt > 0 else 0
            else:
                direction_change = 0
            self._last_direction = direction
        else:
            direction_change = 0

        # 计算曲率（基于最近 3 个点）
        curvature = 0.0
        if len(self.state.history) >= 2:
            t1, x1, y1 = self.state.history[-1]
            t2, x2, y2 = self.state.history[-2]
            # 用三个点计算曲率
            dx1, dy1 = x1 - x2, y1 - y2
            dx2, dy2 = x - x1, y - y1
            cross = dx1 * dy2 - dy1 * dx2
            dot = dx1 * dx2 + dy1 * dy2
            len1 = math.sqrt(dx1 * dx1 + dy1 * dy1)
            len2 = math.sqrt(dx2 * dx2 + dy2 * dy2)
            if len1 > 1 and len2 > 1:
                curvature = abs(cross) / (len1 * len2) if (len1 * len2) > 0 else 0

        # 计算轨迹熵（基于方向分布）
        trajectory_entropy = self._compute_trajectory_entropy()

        # 更新状态
        self.state.x = x
        self.state.y = y
        self.state.vx = vx
        self.state.vy = vy
        self.state.speed = speed
        self.state.acceleration = abs(acceleration)
        self.state.curvature = curvature
        self.state.direction_change = direction_change
        self.state.trajectory_entropy = trajectory_entropy
        self.state.is_moving = speed > 5

        if speed <= 5:
            self.state.stationary_time += dt
        else:
            self.state.stationary_time = 0

        # 更新历史
        self.state.history.append((now, x, y))
        self._last_time = now
        self._last_speed = speed

        return self.state

    def _compute_trajectory_entropy(self) -> float:
        """基于最近轨迹的方向分布计算熵（0=直线，1=完全随机）"""
        if len(self.state.history) < 5:
            return 0.0

        # 计算每段的方向
        directions = []
        hist = list(self.state.history)
        for i in range(1, len(hist)):
            dx = hist[i][1] - hist[i - 1][1]
            dy = hist[i][2] - hist[i - 1][2]
            if math.sqrt(dx * dx + dy * dy) > 2:
                directions.append(math.atan2(dy, dx))

        if len(directions) < 3:
            return 0.0

        # 将方向分箱，计算熵
        bins = [0] * 8  # 8 个方向箱
        for d in directions:
            idx = int((d + math.pi) / (2 * math.pi) * 8) % 8
            bins[idx] += 1

        total = sum(bins)
        if total == 0:
            return 0.0

        entropy = 0.0
        for count in bins:
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)

        # 归一化到 [0, 1]（最大熵 = log2(8) = 3）
        return min(entropy / 3.0, 1.0)


# ============== 音量监控 ==============

class VolumeSensor:
    """系统音量监控，音量突变作为惩罚/电击信号"""

    def __init__(self, sample_rate: float = 5.0, spike_threshold: float = 20.0):
        self.sample_rate = sample_rate
        self.spike_threshold = spike_threshold  # 音量突变阈值（%）
        self.state = VolumeState()
        self._last_time = time.monotonic()

    def update(self) -> VolumeState:
        """读取系统音量并检测突变"""
        now = time.monotonic()
        dt = now - self._last_time
        if dt < 1.0 / self.sample_rate * 0.5:
            return self.state

        try:
            result = subprocess.run(
                ['osascript', '-e', 'output volume of (get volume settings)'],
                capture_output=True, text=True, timeout=2.0
            )
            volume = float(result.stdout.strip())
        except (subprocess.TimeoutExpired, ValueError, Exception):
            volume = self.state.volume

        # 检测突变
        delta = volume - self.state.prev_volume
        spike = delta > self.spike_threshold

        # 更新状态
        self.state.prev_volume = self.state.volume
        self.state.volume = volume
        self.state.delta = delta
        self.state.spike = spike
        self.state.spike_magnitude = delta if spike else 0.0

        self._last_time = now
        return self.state


# ============== 键盘监控 ==============

class KeyboardSensor:
    """键盘状态监控（修饰键轮询，不需要权限）"""

    def __init__(self, sample_rate: float = 10.0):
        self.sample_rate = sample_rate
        self.state = KeyboardState()
        self._last_time = time.monotonic()
        self._key_press_times: Deque[float] = deque(maxlen=20)

    def update(self) -> KeyboardState:
        """读取修饰键状态"""
        now = time.monotonic()
        dt = now - self._last_time
        if dt < 1.0 / self.sample_rate * 0.5:
            return self.state

        flags = Quartz.CGEventSourceFlagsState(Quartz.kCGEventSourceStateHIDSystemState)

        self.state.shift = bool(flags & Quartz.kCGEventFlagMaskShift)
        self.state.command = bool(flags & Quartz.kCGEventFlagMaskCommand)
        self.state.control = bool(flags & Quartz.kCGEventFlagMaskControl)
        self.state.option = bool(flags & Quartz.kCGEventFlagMaskAlternate)
        self.state.caps_lock = bool(flags & Quartz.kCGEventFlagMaskAlphaShift)
        self.state.any_modifier = any([
            self.state.shift, self.state.command,
            self.state.control, self.state.option
        ])

        # 估算按键频率（基于修饰键变化）
        if self.state.any_modifier:
            self._key_press_times.append(now)

        # 清理过期的按键时间
        while self._key_press_times and now - self._key_press_times[0] > 5.0:
            self._key_press_times.popleft()

        self.state.key_press_rate = len(self._key_press_times) / 5.0 if self._key_press_times else 0.0

        # 打字爆发检测（高频按键 = 强触觉刺激）
        self.state.typing_burst = self.state.key_press_rate > 2.0
        self.state.burst_intensity = min(1.0, self.state.key_press_rate / 5.0)

        self._last_time = now
        return self.state


# ============== 亮度监控（昼夜节律） ==============

class BrightnessSensor:
    """屏幕亮度监控，作为昼夜节律输入"""

    def __init__(self, sample_rate: float = 1.0):
        self.sample_rate = sample_rate
        self.state = BrightnessState()
        self._last_time = 0.0
        self._ioreg_available = None  # 缓存 ioreg 是否可用

    def update(self) -> BrightnessState:
        """读取屏幕亮度，失败则用时间作为昼夜节律代理"""
        now = time.monotonic()
        if now - self._last_time < 1.0 / self.sample_rate:
            return self.state

        prev_brightness = self.state.brightness
        brightness = None
        source = "unknown"

        # 方法 1: 用 ioreg 读取 AppleDisplay 亮度
        if self._ioreg_available is not False:
            try:
                result = subprocess.run(
                    ['ioreg', '-c', 'AppleDisplay', '-d', '1', '-r'],
                    capture_output=True, text=True, timeout=2.0
                )
                # 解析 "brightness" = 数字
                for line in result.stdout.split('\n'):
                    if 'brightness' in line.lower() and '=' in line:
                        try:
                            val = float(line.split('=')[-1].strip().strip('"'))
                            if 0 <= val <= 1:
                                brightness = val
                                source = "ioreg"
                                break
                            elif val > 1:
                                # 可能是 0-255 范围
                                brightness = val / 255.0
                                source = "ioreg"
                                break
                        except ValueError:
                            continue
                if brightness is not None:
                    self._ioreg_available = True
            except (subprocess.TimeoutExpired, Exception):
                self._ioreg_available = False

        # 方法 2: 用时间作为昼夜节律代理（果蝇的活动节律）
        if brightness is None:
            hour = time.localtime().tm_hour + time.localtime().tm_min / 60.0
            # 果蝇活动节律：早晨(8-10)和傍晚(17-20)高峰，中午低谷，夜晚最低
            if 7 <= hour < 11:  # 早晨高峰
                brightness = 0.8 + 0.2 * math.sin((hour - 7) / 4 * math.pi)
            elif 11 <= hour < 15:  # 中午低谷
                brightness = 0.5 + 0.1 * math.sin((hour - 11) / 4 * math.pi)
            elif 15 <= hour < 21:  # 傍晚高峰
                brightness = 0.7 + 0.3 * math.sin((hour - 15) / 6 * math.pi)
            else:  # 夜晚
                brightness = 0.15 + 0.05 * math.sin((hour - 21) / 10 * math.pi)
            source = "circadian"

        # 更新状态
        self.state.prev_brightness = prev_brightness
        self.state.brightness = max(0.0, min(1.0, brightness))
        self.state.delta = self.state.brightness - prev_brightness
        self.state.available = True
        self.state.source = source

        self._last_time = now
        return self.state


# ============== 窗口监控（新奇刺激） ==============

class WindowSensor:
    """窗口状态监控，应用切换和窗口变化作为新奇刺激"""

    def __init__(self, sample_rate: float = 2.0):
        self.sample_rate = sample_rate
        self.state = WindowState()
        self._last_time = 0.0

    def _get_frontmost_app(self) -> str:
        """获取当前前台应用名称"""
        try:
            result = subprocess.run(
                ['osascript', '-e', 'tell application "System Events" to get name of first application process whose frontmost is true'],
                capture_output=True, text=True, timeout=2.0
            )
            return result.stdout.strip()
        except (subprocess.TimeoutExpired, Exception):
            return ""

    def _get_window_count(self) -> int:
        """获取当前可见窗口数量"""
        try:
            window_list = Quartz.CGWindowListCopyWindowInfo(
                [Quartz.kCGWindowListOptionOnScreenOnly, Quartz.kCGWindowListExcludeDesktopElements],
                Quartz.kCGNullWindowID
            )
            if window_list:
                # 只统计正常窗口（排除菜单栏、Dock等）
                count = 0
                for w in window_list:
                    if w.get('kCGWindowLayer', 0) == 0 and w.get('kCGWindowBounds', {}).get('Width', 0) > 100:
                        count += 1
                return count
            return 0
        except Exception:
            return 0

    def update(self) -> WindowState:
        """检测前台应用变化和窗口数量变化"""
        now = time.monotonic()
        if now - self._last_time < 1.0 / self.sample_rate:
            return self.state

        # 获取当前状态
        frontmost = self._get_frontmost_app()
        window_count = self._get_window_count()

        # 检测应用切换
        app_switched = (
            frontmost != "" and
            self.state.frontmost_app != "" and
            frontmost != self.state.frontmost_app
        )

        # 检测窗口数量变化
        window_count_changed = (
            self.state.window_count > 0 and
            abs(window_count - self.state.window_count) >= 1
        )

        # 计算新奇度
        novelty = 0.0
        if app_switched:
            novelty = max(novelty, 0.8)  # 应用切换 = 强新奇刺激
            self.state.last_switch_time = now
        if window_count_changed:
            novelty = max(novelty, 0.4)  # 窗口变化 = 中等新奇刺激

        # 新奇度随时间衰减
        time_since_switch = now - self.state.last_switch_time
        if time_since_switch < 3.0:
            novelty = max(novelty, 0.5 * (1.0 - time_since_switch / 3.0))

        # 更新状态
        self.state.prev_frontmost_app = self.state.frontmost_app
        self.state.frontmost_app = frontmost
        self.state.app_switched = app_switched
        self.state.prev_window_count = self.state.window_count
        self.state.window_count = window_count
        self.state.window_count_changed = window_count_changed
        self.state.novelty = min(1.0, novelty)

        self._last_time = now
        return self.state


# ============== 视觉传感器（从 DesktopFly 读取） ==============

class VisionSensor:
    """视觉状态传感器，从 DesktopFly 写入的共享文件读取逼近/气流等视觉信息"""

    def __init__(self, path: str = "/tmp/fly_vision.json", stale_after: float = 1.0):
        self.path = path
        self.stale_after = stale_after
        self.state = VisionState()
        self._last_time = 0.0

    def update(self) -> VisionState:
        """读取视觉状态文件"""
        now = time.monotonic()
        if now - self._last_time < 0.03:  # 30Hz 上限
            return self.state
        self._last_time = now

        try:
            mtime = os.path.getmtime(self.path)
            if time.time() - mtime > self.stale_after:
                self.state.available = False
                return self.state

            with open(self.path, 'r') as f:
                data = json.load(f)

            self.state.loom_left = float(data.get('loom_left', 0))
            self.state.loom_right = float(data.get('loom_right', 0))
            self.state.loom_total = float(data.get('loom_total', 0))
            self.state.air_puff = float(data.get('air_puff', 0))
            self.state.mouse_distance = float(data.get('mouse_distance', 9999))
            self.state.mouse_bearing = float(data.get('mouse_bearing', 0))
            self.state.fly_x = float(data.get('fly_x', 0))
            self.state.fly_y = float(data.get('fly_y', 0))
            self.state.fly_heading = float(data.get('fly_heading', 0))
            self.state.available = True
            self.state.looming = self.state.loom_total > 0.3
        except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError, OSError):
            self.state.available = False

        return self.state


# ============== 感官聚合器 ==============

class SensorHub:
    """感官输入聚合器，统一管理所有感官通道"""

    def __init__(self):
        self.mouse = MouseSensor(sample_rate=30.0)
        self.volume = VolumeSensor(sample_rate=5.0, spike_threshold=20.0)
        self.keyboard = KeyboardSensor(sample_rate=10.0)
        self.brightness = BrightnessSensor(sample_rate=1.0)
        self.window = WindowSensor(sample_rate=2.0)
        self.vision = VisionSensor()
        self.data = SensorData()
        self._last_time = time.monotonic()

        # 慢传感器缓存（减少 subprocess 调用，提升帧率）
        self._slow_sensor_cache = {
            'volume': {'interval': 0.2, 'last_update': 0.0, 'cached': None},
            'window': {'interval': 0.5, 'last_update': 0.0, 'cached': None},
            'brightness': {'interval': 1.0, 'last_update': 0.0, 'cached': None},
            'keyboard': {'interval': 0.1, 'last_update': 0.0, 'cached': None},
        }

    def _update_slow_sensor(self, name: str, sensor, now: float):
        """更新慢传感器，带缓存机制"""
        cache = self._slow_sensor_cache[name]
        if now - cache['last_update'] >= cache['interval'] or cache['cached'] is None:
            cache['cached'] = sensor.update()
            cache['last_update'] = now
        return cache['cached']

    def update(self) -> SensorData:
        """更新所有感官通道（快传感器每帧更新，慢传感器带缓存）"""
        now = time.monotonic()
        dt = now - self._last_time

        # 快传感器：每帧更新
        self.data.mouse = self.mouse.update()
        self.data.vision = self.vision.update()

        # 慢传感器：带缓存，减少 subprocess 调用
        self.data.volume = self._update_slow_sensor('volume', self.volume, now)
        self.data.keyboard = self._update_slow_sensor('keyboard', self.keyboard, now)
        self.data.brightness = self._update_slow_sensor('brightness', self.brightness, now)
        self.data.window = self._update_slow_sensor('window', self.window, now)

        self.data.timestamp = now
        self.data.dt = dt

        self._last_time = now
        return self.data

    async def run_async(self, callback=None):
        """异步运行感官监控循环"""
        while True:
            data = self.update()
            if callback:
                await callback(data)
            await asyncio.sleep(1.0 / 60.0)  # 60Hz 主循环

    def print_status(self):
        """打印当前感官状态（用于调试）"""
        d = self.data
        print(f"\r[鼠标] x={d.mouse.x:6.0f} y={d.mouse.y:6.0f} "
              f"v={d.mouse.speed:7.1f}px/s 曲率={d.mouse.curvature:.3f} "
              f"熵={d.mouse.trajectory_entropy:.2f} | "
              f"[音量] {d.volume.volume:5.1f}% {'⚡' if d.volume.spike else ' '} | "
              f"[键盘] {d.keyboard.key_press_rate:.1f}/s {'⌨️爆发' if d.keyboard.typing_burst else ''} | "
              f"[亮度] {d.brightness.brightness:.2f} ({d.brightness.source}) | "
              f"[窗口] {d.window.frontmost_app[:10]:10s} {'✨新奇' if d.window.novelty > 0.3 else ''}",
              end='', flush=True)


# ============== 命令行测试 ==============

if __name__ == "__main__":
    print("=" * 80)
    print("桌面果蝇感官输入层测试")
    print("=" * 80)
    print("移动鼠标、调节音量、按修饰键来测试感官输入")
    print("按 Ctrl+C 退出")
    print()

    hub = SensorHub()
    try:
        while True:
            hub.update()
            hub.print_status()
            time.sleep(1.0 / 30.0)
    except KeyboardInterrupt:
        print("\n\n测试结束")
