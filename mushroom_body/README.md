# FlyWire 蘑菇体回路 (Mushroom Body Circuit)

从完整 FlyWire FAFB v783 连接组中提取的果蝇蘑菇体回路，用于嗅觉联想学习实验。

## 数据概览

| 指标 | 数值 |
|------|------|
| 神经元总数 | **6,297** |
| 连接总数 | **546,368** |
| 突触总数 | **1,120,067** |
| 数据来源 | FlyWire FAFB v783 (proofread + annotated) |

## 神经元构成

| 角色 | 数量 | 功能 | 神经递质 |
|------|------|------|----------|
| **KC** (Kenyon Cell) | 5,177 | 蘑菇体主要神经元，联想学习的核心 | 乙酰胆碱（兴奋性，已修正 FlyWire 预测错误） |
| **ALPN** (Antennal Lobe Projection Neuron) | 685 | 触角叶投射神经元，嗅觉输入 | 乙酰胆碱（主要）/GABA |
| **DAN** (Dopamine Neuron) | 339 | 多巴胺神经元，惩罚/奖励信号，调制突触可塑性 | 多巴胺（调制性） |
| **MBON** (Mushroom Body Output Neuron) | 96 | 蘑菇体输出神经元，驱动回避/趋向行为 | 乙酰胆碱/谷氨酸/GABA |

### KC 亚型分布
- KCg-m: 2,189（γ叶主群）
- KCab: 1,643（α/β叶）
- KCapbp-m: 338
- KCapbp-ap2: 298
- KCg-d: 295
- KCapbp-ap1: 280
- KCab-p: 128

### DAN 群分布
- PAM01-PAM15: 307 个（投射到蘑菇体萼部/柄/叶，奖励信号为主）
- PPL101-PPL108, PPL201-PPL204: 24 个（投射到蘑菇体叶，惩罚信号为主）
- PPM1201-PPM1205: 16 个

### MBON 类型
MBON01-MBON35，共约 37 种，每种 2-9 个神经元，分别投射到蘑菇体叶的不同区域。

## 关键通路

| 通路 | 连接数 | 突触数 | 功能 |
|------|--------|--------|------|
| **ALPN → KC** | 27,848 | 327,646 | 嗅觉输入：气味信息从触角叶传到蘑菇体 |
| **KC → MBON** | 62,261 | 185,962 | 学习输出：KC 的活动通过 MBON 驱动行为 |
| **DAN → KC** | 46,286 | 56,362 | 多巴胺调制：惩罚/奖励信号调节 KC→MBON 突触强度 |
| **DAN → MBON** | 2,008 | 9,123 | 多巴胺直接调制输出神经元 |
| **KC → KC** | 293,762 | 337,959 | 循环连接：KC 之间的侧向抑制/兴奋 |
| KC → DAN | 80,719 | 115,002 | KC 反馈到多巴胺神经元 |
| MBON → DAN | 2,333 | 5,468 | 输出反馈到调制系统 |
| MBON → KC | 10,474 | 12,260 | 输出反馈到 KC |

## 神经递质分布

| 神经递质 | 连接数 | 符号 | 作用 |
|----------|--------|------|------|
| acetylcholine（乙酰胆碱） | 481,481 | +1.0 | 兴奋性 |
| dopamine（多巴胺） | 56,097 | +0.5 | 调制性（STDP 信号） |
| gaba（γ-氨基丁酸） | 4,789 | -1.0 | 抑制性 |
| glutamate（谷氨酸） | 2,050 | -1.0 | 抑制性（果蝇中） |
| serotonin（血清素） | 1,822 | +0.5 | 调制性 |
| octopamine（章鱼胺） | 129 | +0.5 | 调制性 |

## 文件说明

| 文件 | 大小 | 说明 |
|------|------|------|
| `extract_circuit.py` | 9 KB | 提取脚本，从完整连接组提取蘑菇体回路 |
| `circuit.json` | ~50 MB | 提取后的回路数据（神经元+连接+坐标+报告） |
| `circuit_report.json` | ~5 KB | 回路完整性验证报告 |

## circuit.json 结构

```json
{
  "name": "FlyWire Mushroom Body Circuit",
  "source": "...",
  "neurons": [
    {
      "id": 7205759406...,
      "role": "KC" | "MBON" | "DAN" | "ALPN",
      "type": "KCg-m",
      "class": "Kenyon_Cell",
      "super_class": "central",
      "nt": "acetylcholine",
      "side": "left" | "right" | "center",
      "soma": [x, y, z],  // 胞体坐标（4x4x40nm 体素）
      "pos": [x, y, z]     // 位置坐标
    }
  ],
  "edges": [
    {
      "pre": 7205759406...,   // 突触前神经元 ID
      "post": 7205759406...,  // 突触后神经元 ID
      "syn": 7,                 // 突触数量
      "nt": "acetylcholine",    // 神经递质
      "weight": 7.0             // 带符号权重（syn * nt_sign）
    }
  ],
  "report": { ... }
}
```

## 学习实验设计（下一步）

### 经典嗅觉条件反射
1. **基线测试**：单独给气味 A，果蝇随机走动（未学习，不回避）
2. **训练**：气味 A + 电击（DAN 激活）同时出现，重复 10 次
3. **立即测试**：单独给气味 A，观察果蝇是否回避
4. **对照测试**：单独给气味 B（未配对电击），应不回避（证明特异性学习）
5. **消退测试**：反复给气味 A 但不给电击，观察回避行为是否消退

### 实现要点
- **气味输入**：给特定 ALPN 集群注入电流（不同气味 = 不同 ALPN 激活模式）
- **电击输入**：给 PPL1 群 DAN 注入强电流（惩罚信号）
- **学习机制**：在 KC→MBON 突触上实现 STDP（脉冲时序依赖可塑性）
  - 当 KC 放电和 DAN 放电（多巴胺）时间重合时，KC→MBON 突触权重变化
- **行为输出**：MBON 的活动驱动运动通路（回避 = 转向+行走）

## 数据来源与许可

- **连接数据**: FlyWire Whole-brain Connectome Connectivity Data v783.0 (Zenodo, CC BY 4.0)
- **注释数据**: Schlegel et al. 2024, "Whole-brain annotation and multi-connectome cell typing of Drosophila", Nature (flyconnectome/flywire_annotations, MIT)
- **提取脚本**: 本项目原创

## 已知限制

1. **KC 神经递质修正**: FlyWire 将 KC 预测为多巴胺能，但真实果蝇 KC 是胆碱能的，已手动修正
2. **突触权重近似**: 权重 = 突触数 × 神经递质符号，真实突触强度需要电生理数据校准
3. **神经元参数统一**: 所有神经元使用相同的 LIF 参数，真实神经元的膜特性差异很大
4. **缺少感官前端**: ALPN 的激活模式是人为设定的，没有真实的触角叶→肾小球处理
5. **缺少身体反馈**: MBON 输出没有连接到真实的运动通路，行为输出需要建模
