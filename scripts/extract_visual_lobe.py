#!/usr/bin/env python3
"""
从 FlyWire v783 连接组中提取视叶核心通路神经元。

提取范围：
1. T4a-d / T5a-d — 运动检测神经元（4个方向，On/Off通道）
2. LC11 / LC12 / LC17 — 小叶输出神经元（looming、小物体运动等）
3. Tm1 / Tm2 / Tm9 / Tm3 — 髓质中间神经元（On/Off通道核心）
4. L1 / L2 / L3 / L4 / L5 — lamina 神经元（如果有标注）

输出：visual_lobe/circuit.json（神经元列表 + 连接矩阵）
"""

import json
import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict

# 路径配置
RAW_DATA_DIR = "/Users/fujierui/Desktop/Eon Systems/raw_data"
ANNOTATIONS_FILE = os.path.join(RAW_DATA_DIR, "neuron_annotations.tsv")
CONNECTIONS_FILE = os.path.join(RAW_DATA_DIR, "proofread_connections_783.feather")
OUTPUT_DIR = "/Users/fujierui/Desktop/Eon Systems/visual_lobe"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 要提取的细胞类型（按优先级）
# T4/T5: 运动检测，每个亚型取 N 个
# Tm: 髓质中间神经元，每个类型取 N 个
# LC: 小叶输出，全部或取 N 个
TARGET_CELL_TYPES = {
    # 运动检测神经元（T4=On通道，T5=Off通道，a-d=4个方向）
    "T4a": 80, "T4b": 80, "T4c": 80, "T4d": 80,
    "T5a": 80, "T5b": 80, "T5c": 80, "T5d": 80,
    # 髓质中间神经元（On/Off通道核心）
    "Tm1": 60, "Tm2": 60, "Tm3": 60, "Tm9": 60,
    "Tm4": 40, "Tm20": 40, "Tm21": 40,
    # 小叶输出神经元（视觉特征输出到中央脑）
    "LC11": 50,  # looming 检测
    "LC12": 50,  # 小物体运动
    "LC17": 40,  # 大范围运动
    "LC4": 30,   # 逃逸反射
    "LC6": 30,   # 运动
}

# 细胞类型到功能类别的映射
FUNCTIONAL_CLASS = {
    "T4a": "motion_on", "T4b": "motion_on", "T4c": "motion_on", "T4d": "motion_on",
    "T5a": "motion_off", "T5b": "motion_off", "T5c": "motion_off", "T5d": "motion_off",
    "Tm1": "medulla_off", "Tm2": "medulla_off", "Tm9": "medulla_off",
    "Tm3": "medulla_on", "Tm4": "medulla_on", "Tm20": "medulla_on", "Tm21": "medulla_on",
    "LC11": "output_looming", "LC12": "output_small_object",
    "LC17": "output_large_field", "LC4": "output_escape", "LC6": "output_motion",
}

# 运动方向映射（T4/T5 的 a-d 对应 4 个方向）
MOTION_DIRECTION = {
    "a": "front_to_back",    # 前→后
    "b": "back_to_front",    # 后→前
    "c": "up_to_down",       # 上→下
    "d": "down_to_up",       # 下→上
}


def load_annotations():
    """加载神经元标注文件"""
    print("[1/4] 加载神经元标注...")
    df = pd.read_csv(ANNOTATIONS_FILE, sep="\t", low_memory=False)
    print(f"  总神经元数: {len(df)}")
    print(f"  列名: {list(df.columns[:10])}...")
    return df


def select_neurons(df):
    """根据目标细胞类型筛选神经元"""
    print("\n[2/4] 筛选视叶核心神经元...")

    selected = []
    type_counts = defaultdict(int)

    for cell_type, max_count in TARGET_CELL_TYPES.items():
        # 精确匹配 cell_type
        mask = df["cell_type"] == cell_type
        subset = df[mask].copy()

        if len(subset) == 0:
            # 尝试模糊匹配（有些标注可能有后缀）
            mask = df["cell_type"].str.startswith(cell_type, na=False)
            subset = df[mask].copy()

        if len(subset) > 0:
            # 按 soma 位置排序，均匀采样（覆盖视觉空间）
            if "soma_x" in subset.columns and "soma_y" in subset.columns:
                subset = subset.sort_values(["soma_x", "soma_y"])

            # 均匀采样
            if len(subset) > max_count:
                indices = np.linspace(0, len(subset) - 1, max_count, dtype=int)
                subset = subset.iloc[indices]

            for _, row in subset.iterrows():
                root_id = int(row["root_id"])
                functional = FUNCTIONAL_CLASS.get(cell_type, "unknown")

                # 运动方向
                direction = ""
                if cell_type.startswith("T4") or cell_type.startswith("T5"):
                    subtype = cell_type[-1]
                    direction = MOTION_DIRECTION.get(subtype, "")

                selected.append({
                    "root_id": root_id,
                    "cell_type": cell_type,
                    "functional_class": functional,
                    "motion_direction": direction,
                    "soma_x": float(row.get("soma_x", 0)),
                    "soma_y": float(row.get("soma_y", 0)),
                    "soma_z": float(row.get("soma_z", 0)),
                    "super_class": str(row.get("super_class", "")),
                    "cell_class": str(row.get("cell_class", "")),
                })
                type_counts[cell_type] += 1

    print(f"  筛选出 {len(selected)} 个神经元:")
    for cell_type, count in sorted(type_counts.items()):
        print(f"    {cell_type:10s}: {count:4d} ({FUNCTIONAL_CLASS.get(cell_type, '?')})")

    return selected


def load_connections(selected_ids):
    """加载连接数据，筛选目标神经元之间的连接"""
    print("\n[3/4] 加载连接数据并筛选...")

    # 读取连接文件
    df = pd.read_feather(CONNECTIONS_FILE)
    print(f"  总连接数: {len(df)}")

    # 列名可能不同，检查一下
    print(f"  列名: {list(df.columns)}")

    # 筛选：pre 和 post 都在目标神经元中
    id_set = set(selected_ids)

    # 假设列名是 pre_pt_root_id, post_pt_root_id（FlyWire v783 格式）
    pre_col = "pre_pt_root_id" if "pre_pt_root_id" in df.columns else df.columns[0]
    post_col = "post_pt_root_id" if "post_pt_root_id" in df.columns else df.columns[1]
    weight_col = "syn_count" if "syn_count" in df.columns else df.columns[2]

    print(f"  使用列: pre={pre_col}, post={post_col}, weight={weight_col}")

    mask = df[pre_col].isin(id_set) & df[post_col].isin(id_set)
    connections = df[mask].copy()

    print(f"  目标神经元之间的连接: {len(connections)}")
    print(f"  突触总数: {int(connections[weight_col].sum())}")

    # 神经递质列（如果有）
    nt_cols = [c for c in ["gaba_avg", "ach_avg", "glut_avg", "oct_avg", "ser_avg", "da_avg"] if c in df.columns]

    # 转换为列表
    conn_list = []
    for _, row in connections.iterrows():
        conn_data = {
            "pre": int(row[pre_col]),
            "post": int(row[post_col]),
            "weight": int(row[weight_col]),
        }
        # 加入神经递质平均浓度
        for nt in nt_cols:
            conn_data[nt] = float(row[nt])
        conn_list.append(conn_data)

    return conn_list, nt_cols


def build_circuit(selected, connections, nt_cols=None):
    """构建电路数据结构"""
    print("\n[4/4] 构建电路数据结构...")

    if nt_cols is None:
        nt_cols = []

    # 建立 root_id → 索引映射
    id_to_idx = {n["root_id"]: i for i, n in enumerate(selected)}

    # 构建连接矩阵（稀疏格式）
    pre_indices = []
    post_indices = []
    weights = []
    nt_data = {nt: [] for nt in nt_cols}

    for conn in connections:
        if conn["pre"] in id_to_idx and conn["post"] in id_to_idx:
            pre_indices.append(id_to_idx[conn["pre"]])
            post_indices.append(id_to_idx[conn["post"]])
            weights.append(conn["weight"])
            for nt in nt_cols:
                nt_data[nt].append(conn.get(nt, 0.0))

    print(f"  有效连接: {len(pre_indices)}")
    print(f"  突触总数: {sum(weights)}")

    # 推断每个神经元的主要神经递质（基于输出连接的平均浓度）
    neuron_nt = {}
    for i, n in enumerate(selected):
        nt_sums = {nt: 0.0 for nt in nt_cols}
        nt_count = 0
        for j in range(len(pre_indices)):
            if pre_indices[j] == i:
                for nt in nt_cols:
                    nt_sums[nt] += nt_data[nt][j]
                nt_count += 1
        if nt_count > 0:
            for nt in nt_cols:
                nt_sums[nt] /= nt_count
            # 找出最高的
            main_nt = max(nt_sums, key=nt_sums.get) if nt_sums else "unknown"
            n["main_neurotransmitter"] = main_nt
            n["nt_concentrations"] = nt_sums
        else:
            n["main_neurotransmitter"] = "unknown"
            n["nt_concentrations"] = {nt: 0.0 for nt in nt_cols}

    # 统计神经递质分布
    nt_counts = defaultdict(int)
    for n in selected:
        nt_counts[n["main_neurotransmitter"]] += 1
    print(f"\n  神经递质分布:")
    for nt, count in sorted(nt_counts.items(), key=lambda x: -x[1]):
        print(f"    {nt:10s}: {count:4d}")

    # 统计每个神经元的输入/输出连接数
    in_degree = np.zeros(len(selected), dtype=int)
    out_degree = np.zeros(len(selected), dtype=int)
    for i in range(len(pre_indices)):
        out_degree[pre_indices[i]] += 1
        in_degree[post_indices[i]] += 1

    # 功能类别统计
    func_counts = defaultdict(int)
    for n in selected:
        func_counts[n["functional_class"]] += 1

    print(f"\n  功能类别分布:")
    for func, count in sorted(func_counts.items()):
        print(f"    {func:20s}: {count:4d}")

    # 构建电路数据
    circuit = {
        "name": "Drosophila Visual Lobe (Core Pathway)",
        "source": "FlyWire v783 proofread connectome",
        "description": "提取自 FlyWire v783 的视叶核心通路：lamina→medulla→T4/T5运动检测→LC输出",
        "neuron_count": len(selected),
        "connection_count": len(pre_indices),
        "synapse_count": int(sum(weights)),
        "neurons": selected,
        "connections": {
            "pre_indices": pre_indices,
            "post_indices": post_indices,
            "weights": weights,
            "neurotransmitters": nt_data if nt_cols else {},
        },
        "statistics": {
            "in_degree_mean": float(np.mean(in_degree)),
            "in_degree_max": int(np.max(in_degree)),
            "out_degree_mean": float(np.mean(out_degree)),
            "out_degree_max": int(np.max(out_degree)),
            "functional_classes": dict(func_counts),
        },
    }

    return circuit


def main():
    print("=" * 60)
    print("FlyWire 视叶核心通路提取")
    print("=" * 60)

    # 1. 加载标注
    df = load_annotations()

    # 2. 筛选神经元
    selected = select_neurons(df)
    selected_ids = [n["root_id"] for n in selected]

    if len(selected) == 0:
        print("ERROR: 未筛选到任何神经元！")
        sys.exit(1)

    # 3. 加载连接
    connections, nt_cols = load_connections(selected_ids)

    # 4. 构建电路
    circuit = build_circuit(selected, connections, nt_cols)

    # 5. 保存
    output_file = os.path.join(OUTPUT_DIR, "circuit.json")
    print(f"\n保存到: {output_file}")
    with open(output_file, "w") as f:
        json.dump(circuit, f, ensure_ascii=False, indent=2)

    file_size = os.path.getsize(output_file)
    print(f"文件大小: {file_size / 1024 / 1024:.1f} MB")

    print("\n" + "=" * 60)
    print("提取完成！")
    print(f"  神经元: {circuit['neuron_count']}")
    print(f"  连接: {circuit['connection_count']}")
    print(f"  突触: {circuit['synapse_count']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
