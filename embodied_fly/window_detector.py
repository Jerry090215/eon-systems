#!/usr/bin/env python3
"""
窗口检测模块 - 实时获取 macOS 桌面所有可见窗口的位置和大小

用 AppleScript 通过 System Events 获取窗口信息，解析成 Python 可用的格式。
窗口坐标：macOS 原生坐标（左上角为原点，y 向下）
"""

import os
import sys
import time
import subprocess
import json
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class WindowInfo:
    """单个窗口的信息"""
    app_name: str       # 应用名称（如 "Google Chrome", "Terminal"）
    window_name: str    # 窗口标题
    x: int              # 左上角 x 坐标（屏幕坐标，像素）
    y: int              # 左上角 y 坐标（屏幕坐标，像素）
    width: int          # 窗口宽度
    height: int         # 窗口高度

    @property
    def center(self) -> Tuple[int, int]:
        """窗口中心点坐标"""
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def area(self) -> int:
        """窗口面积"""
        return self.width * self.height

    def contains_point(self, px: int, py: int, margin: int = 0) -> bool:
        """判断点是否在窗口内（可加边距）"""
        return (self.x - margin <= px <= self.x + self.width + margin and
                self.y - margin <= py <= self.y + self.height + margin)

    def distance_to_point(self, px: int, py: int) -> float:
        """计算点到窗口边缘的最短距离（点在窗口内返回 0）"""
        if self.contains_point(px, py):
            return 0.0
        # 计算到最近边缘的距离
        dx = max(self.x - px, 0, px - (self.x + self.width))
        dy = max(self.y - py, 0, py - (self.y + self.height))
        return (dx**2 + dy**2) ** 0.5

    def to_dict(self) -> dict:
        return {
            "app": self.app_name,
            "name": self.window_name,
            "x": self.x, "y": self.y,
            "w": self.width, "h": self.height,
        }


class WindowDetector:
    """
    窗口检测器 - 实时获取桌面所有可见窗口

    用法:
        detector = WindowDetector()
        detector.update()  # 刷新窗口列表
        windows = detector.windows
    """

    # AppleScript：获取所有可见进程的窗口信息
    # 输出格式：每行一个窗口，用 | 分隔：app|window_name|x|y|width|height
    _APPLESCRIPT = '''
    tell application "System Events"
        set output to ""
        repeat with p in (every process whose visible is true)
            set appName to name of p
            repeat with w in (every window of p)
                try
                    set wName to name of w
                    set wPos to position of w
                    set wSize to size of w
                    set wX to item 1 of wPos
                    set wY to item 2 of wPos
                    set wW to item 1 of wSize
                    set wH to item 2 of wSize
                    set output to output & appName & "|" & wName & "|" & wX & "|" & wY & "|" & wW & "|" & wH & linefeed
                end try
            end repeat
        end repeat
        return output
    end tell
    '''

    def __init__(self, refresh_interval: float = 1.0):
        """
        Args:
            refresh_interval: 窗口列表刷新间隔（秒），避免频繁调用 AppleScript
        """
        self.refresh_interval = refresh_interval
        self.windows: List[WindowInfo] = []
        self.last_refresh = 0.0
        self.screen_width = 0
        self.screen_height = 0
        self._get_screen_size()

    def _get_screen_size(self):
        """获取屏幕分辨率"""
        try:
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True, text=True, timeout=5
            )
            # 解析主显示器分辨率
            for line in result.stdout.split('\n'):
                if 'Resolution' in line:
                    parts = line.strip().split()
                    # 格式: Resolution: 2560 x 1600 Retina
                    for i, p in enumerate(parts):
                        if p == 'x' and i > 0:
                            self.screen_width = int(parts[i-1])
                            self.screen_height = int(parts[i+1])
                            break
                    break
        except Exception:
            # 默认值
            self.screen_width = 1920
            self.screen_height = 1080

    def update(self, force: bool = False) -> bool:
        """
        刷新窗口列表

        Args:
            force: 强制刷新（忽略刷新间隔）

        Returns:
            是否成功刷新
        """
        now = time.monotonic()
        if not force and (now - self.last_refresh) < self.refresh_interval:
            return False

        try:
            result = subprocess.run(
                ["osascript", "-e", self._APPLESCRIPT],
                capture_output=True, text=True, timeout=10
            )
            self._parse_output(result.stdout)
            self.last_refresh = now
            return True
        except Exception as e:
            print(f"[WindowDetector] 刷新失败: {e}")
            return False

    def _parse_output(self, output: str):
        """解析 AppleScript 输出"""
        self.windows = []
        for line in output.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            # 从右往左 split，只 split 5 次（窗口名称可能包含 | 字符）
            # 格式: app|window_name|x|y|width|height
            parts = line.rsplit('|', 5)
            if len(parts) != 6:
                continue
            try:
                app_name = parts[0]
                window_name = parts[1]
                x = int(parts[2])
                y = int(parts[3])
                width = int(parts[4])
                height = int(parts[5])
                # 过滤掉太小的窗口（可能是菜单、标签栏等）
                if width < 100 or height < 100:
                    continue
                self.windows.append(WindowInfo(
                    app_name=app_name,
                    window_name=window_name,
                    x=x, y=y,
                    width=width, height=height
                ))
            except (ValueError, IndexError):
                continue

    def get_windows_by_app(self, app_name: str) -> List[WindowInfo]:
        """按应用名称筛选窗口"""
        return [w for w in self.windows if w.app_name.lower() == app_name.lower()]

    def get_window_at_point(self, px: int, py: int) -> Optional[WindowInfo]:
        """获取指定点所在的窗口（最上层的）"""
        for w in reversed(self.windows):  # 后获取的可能在更上层
            if w.contains_point(px, py):
                return w
        return None

    def get_nearest_window(self, px: int, py: int, max_dist: float = float('inf')) -> Optional[WindowInfo]:
        """获取离指定点最近的窗口"""
        nearest = None
        min_dist = max_dist
        for w in self.windows:
            dist = w.distance_to_point(px, py)
            if dist < min_dist:
                min_dist = dist
                nearest = w
        return nearest

    def to_json(self) -> str:
        """序列化为 JSON（用于调试或共享）"""
        return json.dumps([w.to_dict() for w in self.windows], ensure_ascii=False, indent=2)


# 坐标转换工具
def fly_to_screen(fly_x: float, fly_y: float,
                  screen_width: int, screen_height: int) -> Tuple[int, int]:
    """
    将 DesktopFly 坐标转换为屏幕坐标

    DesktopFly 坐标：屏幕中心为原点 (0,0)，y 向上
    屏幕坐标：左上角为原点 (0,0)，y 向下

    Args:
        fly_x: DesktopFly x 坐标
        fly_y: DesktopFly y 坐标
        screen_width: 屏幕宽度
        screen_height: 屏幕高度

    Returns:
        (screen_x, screen_y) 屏幕坐标
    """
    screen_x = int(fly_x + screen_width / 2)
    screen_y = int(screen_height / 2 - fly_y)
    return (screen_x, screen_y)


def screen_to_fly(screen_x: int, screen_y: int,
                  screen_width: int, screen_height: int) -> Tuple[float, float]:
    """
    将屏幕坐标转换为 DesktopFly 坐标

    Args:
        screen_x: 屏幕 x 坐标
        screen_y: 屏幕 y 坐标
        screen_width: 屏幕宽度
        screen_height: 屏幕高度

    Returns:
        (fly_x, fly_y) DesktopFly 坐标
    """
    fly_x = float(screen_x - screen_width / 2)
    fly_y = float(screen_height / 2 - screen_y)
    return (fly_x, fly_y)


if __name__ == "__main__":
    # 测试
    detector = WindowDetector()
    detector.update(force=True)
    print(f"屏幕分辨率: {detector.screen_width}x{detector.screen_height}")
    print(f"检测到 {len(detector.windows)} 个窗口:")
    for w in detector.windows[:10]:
        print(f"  [{w.app_name}] {w.window_name[:30]} @ ({w.x},{w.y}) {w.width}x{w.height}")
    if len(detector.windows) > 10:
        print(f"  ... 还有 {len(detector.windows) - 10} 个窗口")
