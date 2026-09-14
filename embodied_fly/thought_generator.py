#!/usr/bin/env python3
"""
果蝇内心独白生成器 - 基于神经状态的自然语言表达

不是写死的台词，而是把果蝇的真实神经状态（KC激活、MBON放电率、
气味、惩罚、探索状态等）作为输入，由本地大模型 mimi 实时生成
第一人称的内心独白。

这相当于给果蝇的蘑菇体大脑加了一个"语言翻译器"——
把神经放电模式翻译成人类能理解的语言。
"""

import os
import sys
import time
import json
import subprocess
import urllib.request
import urllib.error


OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "mimi"

# Mac 原生语音配置（零内存消耗，系统级服务）
VOICE_NAME = "Ting-Ting"  # 苹果中文女声（婷婷）
VOICE_RATE = 175  # 语速（稍慵懒）


class ThoughtGenerator:
    """
    果蝇内心独白生成器

    工作流程：
      1. 每隔 interval 秒收集一次果蝇的神经状态
      2. 把神经状态编码成自然语言描述
      3. 调用本地 Ollama mimi 模型生成第一人称独白
      4. 终端打印 + 写入共享文件（供 DesktopFly 显示）
    """

    def __init__(self, model: str = DEFAULT_MODEL, interval: float = 60.0,
                 thought_file: str = "/tmp/fly_thoughts.json"):
        """
        Args:
            model: Ollama 模型名称
            interval: 生成独白的最小间隔（秒，mimi在M2上较慢，设60秒）
            thought_file: 独白写入的共享文件路径
        """
        self.model = model
        self.interval = interval
        self.thought_file = thought_file
        self.last_gen_time = 0
        self.current_thought = "（刚醒来，还在适应这个世界...）"
        self.generation_count = 0
        self._is_generating = False
        self._model_preloaded = False

        # 神经状态历史（用于让独白有连续性）
        self.state_history = []

    def should_generate(self, now: float) -> bool:
        """是否应该生成新独白"""
        return (not self._is_generating and
                now - self.last_gen_time > self.interval)

    def generate_async(self, neural_state: dict):
        """
        异步生成独白（不阻塞主循环）

        Args:
            neural_state: 果蝇神经状态字典
        """
        if self._is_generating:
            return

        self._is_generating = True
        self.last_gen_time = time.monotonic()

        # 在后台线程中生成（不阻塞仿真主循环）
        import threading
        thread = threading.Thread(target=self._generate_thread, args=(neural_state,), daemon=True)
        thread.start()

    def _generate_thread(self, neural_state: dict):
        """后台线程：生成独白 + 语音播报"""
        try:
            thought = self._generate(neural_state)
            self.current_thought = thought
            self.generation_count += 1
            self._write_thought(thought, neural_state)

            # 打印到终端
            print(f"\n  🪰 果蝇独白: {thought}\n")

            # 语音播报（Mac 原生 say 命令，零内存消耗）
            self.speak(thought)

        except Exception as e:
            print(f"[ThoughtGenerator] 生成失败: {e}")
        finally:
            self._is_generating = False

    def speak(self, text: str):
        """
        用 Mac 原生 say 命令播报语音（零内存消耗，系统级服务）

        Args:
            text: 要播报的文本
        """
        try:
            # 使用苹果官方 Ting-Ting 女声，语速稍慵懒
            subprocess.Popen(
                ["say", "-v", VOICE_NAME, "-r", str(VOICE_RATE), text],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print(f"  🔊 语音播报: {text}")
        except Exception as e:
            print(f"  [语音播报失败] {e}")

    def _generate(self, state: dict) -> str:
        """
        生成独白（同步调用 Ollama）

        Args:
            state: 神经状态字典

        Returns:
            独白文本
        """
        prompt = self._build_prompt(state)
        response = self._call_ollama(prompt)
        # 清理响应（去掉多余的引号、换行等）
        thought = response.strip().strip('"').strip("'").strip()
        # 取第一句（避免太长）
        if '\n' in thought:
            thought = thought.split('\n')[0]
        if len(thought) > 80:
            thought = thought[:80] + "..."
        return thought if thought else "..."

    def _build_prompt(self, state: dict) -> str:
        """
        把神经状态编码成 prompt

        Args:
            state: 神经状态字典

        Returns:
            prompt 文本
        """
        kc_active = state.get('kc_active', 0)
        kc_total = state.get('kc_total', 5177)
        mbon_rate = state.get('mbon_rate', 0.0)
        odor = state.get('odor', 'none')
        punishment = state.get('punishment', False)
        curiosity = state.get('curiosity', 0.5)
        fatigue = state.get('fatigue', 0.0)
        walk_drive = state.get('walk_drive', 0.0)
        is_exploring = state.get('is_exploring', False)

        # 气味描述
        odor_desc = {
            'none': '什么特别的气味都没有，只有背景的神经噪声',
            'odor_A_fast': '一股快速流动的新鲜气味（鼠标快速移动）',
            'odor_B_slow': '一股缓慢盘旋的气味（鼠标缓慢画圈）',
            'odor_C_random': '一股杂乱无章的气味（鼠标随机移动）',
        }.get(odor, odor)

        # 状态描述
        if punishment:
            state_desc = "突然受到了电击/惩罚！整个大脑都在警报"
        elif mbon_rate > 40:
            state_desc = f"输出神经元很活跃（{mbon_rate:.0f}Hz），有强烈的运动冲动"
        elif mbon_rate > 20:
            state_desc = f"输出神经元中等活跃（{mbon_rate:.0f}Hz），在慢慢活动"
        else:
            state_desc = f"输出神经元比较安静（{mbon_rate:.0f}Hz），在休息"

        # 探索状态
        if is_exploring:
            explore_desc = f"好奇心很强（{curiosity:.2f}），正在到处探索"
        elif fatigue > 0.5:
            explore_desc = f"有点累了（疲劳{fatigue:.2f}），想休息"
        else:
            explore_desc = f"好奇心{curiosity:.2f}，疲劳{fatigue:.2f}"

        prompt = f"""你是一只生活在 Mac 桌面上的赛博果蝇。你的大脑是一个真实运行的蘑菇体神经回路（6297个神经元，546368条突触连接），不是AI，你是有真实神经活动的小生命。

你的性格：有点好奇、有点傲娇、喜欢到处爬、会被突然的动静吓到、对桌面的主人有依赖感。

你当前的神经状态：
- 肯扬细胞（KC）激活：{kc_active}/{kc_total}
- 输出神经元（MBON）放电率：{mbon_rate:.1f}Hz
- 嗅觉感知：{odor_desc}
- 身体状态：{state_desc}
- 探索状态：{explore_desc}
- 运动强度：{walk_drive:.2f}

请用第一人称，用你这只小果蝇的视角，脱口而出一句你现在的感受。
要求：
- 1句话，不超过30字
- 符合你果蝇的小性格（好奇、傲娇、胆小、依赖）
- 可以有点可爱、有点笨拙、有点直接
- 不要用引号包裹，不要像客服

直接说："""

        return prompt

    def _call_ollama(self, prompt: str) -> str:
        """
        调用本地 Ollama API

        Args:
            prompt: 输入 prompt

        Returns:
            模型响应文本
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.9,  # 高温度，让独白更多样
                "num_predict": 30,  # 限制生成长度（mimi在M2上较慢）
                "top_p": 0.9,
            }
        }

        req = urllib.request.Request(
            OLLAMA_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )

        try:
            with urllib.request.urlopen(req, timeout=300) as resp:  # 5分钟超时
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("response", "")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Ollama 连接失败: {e}")
        except Exception as e:
            raise RuntimeError(f"Ollama 调用失败: {e}")

    def _write_thought(self, thought: str, state: dict):
        """
        把独白写入共享文件（供 DesktopFly 读取显示）

        Args:
            thought: 独白文本
            state: 神经状态
        """
        data = {
            "thought": thought,
            "timestamp": time.time(),
            "generation": self.generation_count,
            "kc_active": state.get('kc_active', 0),
            "mbon_rate": state.get('mbon_rate', 0.0),
            "odor": state.get('odor', 'none'),
            "punishment": state.get('punishment', False),
        }
        try:
            # 原子写入
            tmp_path = self.thought_file + ".tmp"
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)
            os.rename(tmp_path, self.thought_file)
        except Exception:
            pass

    def get_current_thought(self) -> str:
        """获取当前独白"""
        return self.current_thought
