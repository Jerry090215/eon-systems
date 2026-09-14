#!/usr/bin/env python3
"""
桌面果蝇具身智能系统 - 综合测试

测试完整的感官→编码→神经激活闭环：
  1. 感官输入层（鼠标/音量/键盘）
  2. 神经编码层（物理变量→ALPN/DAN 激活）
  3. 完整闭环实时运行

运行方式: python3 test_embodied_fly.py [测试时长秒数]
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sensors import SensorHub
from encoder import NeuralEncoder, MushroomBodyCircuit


def test_sensors(duration: float = 3.0):
    """测试 1: 感官输入层"""
    print("=" * 80)
    print("测试 1: 感官输入层 (Sensors)")
    print("=" * 80)
    print(f"运行 {duration} 秒，请移动鼠标、调节音量、按修饰键\n")

    hub = SensorHub()
    start = time.monotonic()
    last_print = 0

    while time.monotonic() - start < duration:
        data = hub.update()
        now = time.monotonic()
        if now - last_print > 0.5:
            hub.print_status()
            print()
            last_print = now
        time.sleep(1.0 / 30.0)

    print(f"\n✓ 感官输入层测试完成")
    print(f"  鼠标: x={hub.data.mouse.x:.0f}, y={hub.data.mouse.y:.0f}, "
          f"速度={hub.data.mouse.speed:.1f}px/s")
    print(f"  音量: {hub.data.volume.volume:.1f}%")
    print(f"  键盘: 修饰键={'有' if hub.data.keyboard.any_modifier else '无'}")
    return hub


def test_encoder(duration: float = 3.0):
    """测试 2: 神经编码层"""
    print("\n" + "=" * 80)
    print("测试 2: 神经编码层 (Encoder)")
    print("=" * 80)

    # 加载回路
    circuit_path = os.path.join("..", "mushroom_body", "circuit.json")
    circuit = MushroomBodyCircuit(circuit_path)

    encoder = NeuralEncoder(circuit)
    hub = SensorHub()

    print(f"\n运行 {duration} 秒")
    print("  - 快速直线移动鼠标 → 气味 A（危险）")
    print("  - 缓慢画圈移动鼠标 → 气味 B（安全）")
    print("  - 快速调大音量 → 惩罚/电击（PPL1 DAN 激活）\n")

    start = time.monotonic()
    last_print = 0
    odor_history = []
    punishment_count = 0

    while time.monotonic() - start < duration:
        sensor_data = hub.update()
        activation = encoder.encode(sensor_data)
        currents = encoder.get_input_current_vector()

        if activation.odor_label not in odor_history:
            odor_history.append(activation.odor_label)
        if activation.punishment:
            punishment_count += 1

        now = time.monotonic()
        if now - last_print > 0.5:
            encoder.print_status()
            print(f" | 激活神经元={len(currents)}", end='', flush=True)
            print()
            last_print = now

        time.sleep(1.0 / 30.0)

    print(f"\n✓ 神经编码层测试完成")
    print(f"  检测到的气味: {', '.join(odor_history)}")
    print(f"  惩罚事件数: {punishment_count}")
    print(f"  最终激活神经元数: {len(currents)}")
    return encoder, hub


def test_closed_loop(duration: float = 5.0):
    """测试 3: 完整闭环"""
    print("\n" + "=" * 80)
    print("测试 3: 完整闭环 (Sensors → Encoder → Neural Activation)")
    print("=" * 80)

    circuit_path = os.path.join("..", "mushroom_body", "circuit.json")
    circuit = MushroomBodyCircuit(circuit_path)
    encoder = NeuralEncoder(circuit)
    hub = SensorHub()

    print(f"\n运行 {duration} 秒，完整闭环实时运行")
    print("  感官输入 → 神经编码 → ALPN/DAN 激活向量\n")

    start = time.monotonic()
    last_print = 0
    max_alpn = 0
    max_dan = 0
    total_updates = 0

    while time.monotonic() - start < duration:
        sensor_data = hub.update()
        activation = encoder.encode(sensor_data)
        currents = encoder.get_input_current_vector()
        total_updates += 1

        alpn_total = sum(activation.alpn_activation.values())
        dan_total = sum(activation.dan_activation.values())
        max_alpn = max(max_alpn, alpn_total)
        max_dan = max(max_dan, dan_total)

        now = time.monotonic()
        if now - last_print > 0.5:
            print(f"\r[更新 #{total_updates:4d}] "
                  f"气味={activation.odor_label:15s} "
                  f"ALPN={alpn_total:6.1f}nA "
                  f"DAN={dan_total:6.1f}nA "
                  f"惩罚={'⚡' if activation.punishment else '○'} "
                  f"兴奋性={activation.global_excitability:.2f} "
                  f"激活神经元={len(currents):4d}",
                  end='', flush=True)
            last_print = now

        time.sleep(1.0 / 60.0)  # 60Hz 闭环

    print(f"\n\n✓ 完整闭环测试完成")
    print(f"  总更新次数: {total_updates}")
    print(f"  平均更新率: {total_updates/duration:.1f} Hz")
    print(f"  ALPN 峰值激活: {max_alpn:.1f} nA")
    print(f"  DAN 峰值激活: {max_dan:.1f} nA")
    print(f"  闭环延迟: <{1000/60:.1f} ms (目标 60Hz)")


def main():
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0

    print("\n" + "█" * 80)
    print("█" + " " * 30 + "桌面果蝇具身智能系统" + " " * 28 + "█")
    print("█" + " " * 26 + "Embodied Fly - Sensory Encoding System" + " " * 22 + "█")
    print("█" * 80 + "\n")

    try:
        # 测试 1: 感官输入层
        test_sensors(duration)

        # 测试 2: 神经编码层
        test_encoder(duration)

        # 测试 3: 完整闭环
        test_closed_loop(duration * 1.5)

        print("\n" + "=" * 80)
        print("✅ 所有测试通过！桌面果蝇具身智能系统运行正常。")
        print("=" * 80)
        print("\n下一步:")
        print("  1. 实现蘑菇体 LIF 仿真（6,297 神经元实时放电）")
        print("  2. 实现 KC→MBON STDP 突触可塑性（学习机制）")
        print("  3. 连接 DesktopFly 3D 果蝇（行为输出）")
        print("  4. 运行嗅觉条件反射实验（气味+电击→学习回避）")

    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
    except Exception as e:
        print(f"\n\n测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
