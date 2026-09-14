#!/usr/bin/env python3
"""
从完整 FlyWire 连接组中提取蘑菇体（Mushroom Body）回路。

提取的神经元类型：
  - KC   (Kenyon Cell)           : 蘑菇体主要神经元，联想学习的核心
  - MBON (Mushroom Body Output Neuron) : 蘑菇体输出神经元，驱动行为
  - DAN  (Dopamine Neuron)       : 多巴胺神经元，惩罚/奖励信号，调制突触可塑性
  - ALPN (Antennal Lobe Projection Neuron) : 触角叶投射神经元，嗅觉输入

神经递质符号映射（修正 KC 为胆碱能）：
  ACH   = +1.0  (兴奋性)
  GABA  = -1.0  (抑制性)
  GLUT  = -1.0  (抑制性，果蝇中谷氨酸主要为抑制性)
  DA    = +0.5  (多巴胺，调制性，在 STDP 中起作用)
  SER   = +0.5  (血清素，调制性)
  OCT   = +0.5  (章鱼胺，调制性)
  未知  = +1.0  (默认兴奋性)

用法: python3 extract_circuit.py
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow.feather as feather

# ============== 路径配置 ==============
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "raw_data")
OUT_DIR = os.path.join(BASE_DIR, "mushroom_body")

ANNOT_FILE = os.path.join(RAW_DIR, "neuron_annotations.parquet")
CONN_FILE = os.path.join(RAW_DIR, "proofread_connections_783.feather")
OUT_CIRCUIT = os.path.join(OUT_DIR, "circuit.json")
OUT_REPORT = os.path.join(OUT_DIR, "circuit_report.json")

# ============== 神经递质符号映射 ==============
NT_SIGN = {
    "acetylcholine": 1.0,
    "gaba": -1.0,
    "glutamate": -1.0,
    "dopamine": 0.5,
    "serotonin": 0.5,
    "octopamine": 0.5,
}

# KC 的真实神经递质是乙酰胆碱（兴奋性），FlyWire 预测为多巴胺是错误的
KC_TRUE_NT = "acetylcholine"

# ============== 神经元筛选 ==============

def select_neurons(ann: pd.DataFrame) -> pd.DataFrame:
    """从注释数据中筛选蘑菇体回路相关神经元。"""
    masks = {
        "KC": ann["cell_class"] == "Kenyon_Cell",
        "MBON": ann["cell_type"].str.contains("MBON", case=False, na=False),
        "DAN": ann["cell_type"].str.match(r"^(PAM|PPL1|PPM)", na=False),
        "ALPN": ann["cell_class"] == "ALPN",
    }

    selected = pd.DataFrame()
    for role, mask in masks.items():
        subset = ann[mask].copy()
        subset["role"] = role
        selected = pd.concat([selected, subset])

    # 去重（某些神经元可能同时匹配多个条件，优先更具体的角色）
    selected = selected[~selected.index.duplicated(keep="first")]

    # 修正 KC 的神经递质
    selected.loc[selected["role"] == "KC", "top_nt"] = KC_TRUE_NT

    print(f"神经元筛选完成:")
    for role in ["KC", "MBON", "DAN", "ALPN"]:
        count = (selected["role"] == role).sum()
        print(f"  {role:6}: {count:5} 个")
    print(f"  总计: {len(selected):5} 个")

    return selected


# ============== 连接提取 ==============

def extract_connections(conn_table, neuron_ids: set) -> list:
    """从完整连接组中提取目标神经元之间的连接。"""
    print("\n开始提取连接...")

    # 分批处理连接数据（1685 万行）
    batch_size = 2_000_000
    total_rows = conn_table.num_rows
    edges = []
    edge_set = set()  # 用于去重

    for start in range(0, total_rows, batch_size):
        end = min(start + batch_size, total_rows)
        batch = conn_table.slice(start, end - start).to_pandas()

        # 过滤：突触前和突触后都在目标神经元中
        mask = batch["pre_pt_root_id"].isin(neuron_ids) & batch["post_pt_root_id"].isin(neuron_ids)
        filtered = batch[mask]

        for _, row in filtered.iterrows():
            pre = int(row["pre_pt_root_id"])
            post = int(row["post_pt_root_id"])
            syn = int(row["syn_count"])

            # 确定神经递质（取概率最高的）
            nt_probs = {
                "acetylcholine": row.get("ach_avg", 0),
                "gaba": row.get("gaba_avg", 0),
                "glutamate": row.get("glut_avg", 0),
                "dopamine": row.get("da_avg", 0),
                "serotonin": row.get("ser_avg", 0),
                "octopamine": row.get("oct_avg", 0),
            }
            top_nt = max(nt_probs, key=nt_probs.get)
            sign = NT_SIGN.get(top_nt, 1.0)
            weight = round(syn * sign, 1)

            edge_key = (pre, post)
            if edge_key not in edge_set:
                edge_set.add(edge_key)
                edges.append({
                    "pre": pre,
                    "post": post,
                    "syn": syn,
                    "nt": top_nt,
                    "weight": weight,
                })

        if (start // batch_size) % 2 == 0:
            print(f"  处理进度: {end:,}/{total_rows:,} ({end/total_rows*100:.1f}%), 已提取连接: {len(edges):,}")

    print(f"连接提取完成: {len(edges):,} 条")
    return edges


# ============== 回路验证 ==============

def validate_circuit(neurons: pd.DataFrame, edges: list) -> dict:
    """验证提取的回路完整性。"""
    print("\n=== 回路验证 ===")

    neuron_ids = set(neurons["root_id"].values)
    role_of = dict(zip(neurons["root_id"], neurons["role"]))

    # 按角色统计连接
    role_edges = defaultdict(int)
    role_syn = defaultdict(int)
    for e in edges:
        pre_role = role_of.get(e["pre"], "?")
        post_role = role_of.get(e["post"], "?")
        key = f"{pre_role}->{post_role}"
        role_edges[key] += 1
        role_syn[key] += e["syn"]

    print("\n连接类型统计:")
    for key in sorted(role_edges.keys()):
        print(f"  {key:15}: {role_edges[key]:6,} 条连接, {role_syn[key]:8,} 个突触")

    # 关键通路检查
    checks = {
        "ALPN->KC (嗅觉输入)": role_edges.get("ALPN->KC", 0) > 0,
        "KC->MBON (学习输出)": role_edges.get("KC->MBON", 0) > 0,
        "DAN->KC (多巴胺调制)": role_edges.get("DAN->KC", 0) > 0,
        "DAN->MBON (多巴胺调制)": role_edges.get("DAN->MBON", 0) > 0,
        "KC->KC (循环连接)": role_edges.get("KC->KC", 0) > 0,
    }

    print("\n关键通路检查:")
    all_passed = True
    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        if not passed:
            all_passed = False
        print(f"  {status} {check}")

    # 神经递质统计
    nt_counts = defaultdict(int)
    for e in edges:
        nt_counts[e["nt"]] += 1

    print("\n神经递质统计:")
    for nt, count in sorted(nt_counts.items(), key=lambda x: -x[1]):
        print(f"  {nt:15}: {count:6,} 条连接")

    # 神经元坐标统计
    has_soma = neurons["soma_x"].notna().sum()
    has_pos = neurons["pos_x"].notna().sum()

    report = {
        "total_neurons": len(neurons),
        "total_edges": len(edges),
        "total_synapses": sum(e["syn"] for e in edges),
        "neurons_by_role": neurons["role"].value_counts().to_dict(),
        "edges_by_type": dict(role_edges),
        "synapses_by_type": dict(role_syn),
        "neurotransmitter_counts": dict(nt_counts),
        "has_soma_coords": int(has_soma),
        "has_pos_coords": int(has_pos),
        "key_pathways": checks,
        "all_checks_passed": all_passed,
    }

    print(f"\n验证总结:")
    print(f"  神经元: {len(neurons):,}")
    print(f"  连接: {len(edges):,}")
    print(f"  突触: {sum(e['syn'] for e in edges):,}")
    print(f"  所有关键通路: {'✓ 通过' if all_passed else '✗ 有缺失'}")

    return report


# ============== 保存回路 ==============

def save_circuit(neurons: pd.DataFrame, edges: list, report: dict):
    """保存回路数据为 JSON。"""
    # 神经元信息
    neuron_list = []
    for _, row in neurons.iterrows():
        neuron_list.append({
            "id": int(row["root_id"]),
            "role": row["role"],
            "type": row.get("cell_type", ""),
            "class": row.get("cell_class", ""),
            "super_class": row.get("super_class", ""),
            "nt": row.get("top_nt", ""),
            "side": row.get("side", ""),
            "soma": [
                float(row["soma_x"]) if pd.notna(row["soma_x"]) else None,
                float(row["soma_y"]) if pd.notna(row["soma_y"]) else None,
                float(row["soma_z"]) if pd.notna(row["soma_z"]) else None,
            ],
            "pos": [
                float(row["pos_x"]) if pd.notna(row["pos_x"]) else None,
                float(row["pos_y"]) if pd.notna(row["pos_y"]) else None,
                float(row["pos_z"]) if pd.notna(row["pos_z"]) else None,
            ],
        })

    circuit = {
        "name": "FlyWire Mushroom Body Circuit",
        "source": "FlyWire FAFB v783 (proofread_connections_783.feather + neuron_annotations.tsv)",
        "neurons": neuron_list,
        "edges": edges,
        "report": report,
    }

    with open(OUT_CIRCUIT, "w") as f:
        json.dump(circuit, f, ensure_ascii=False)
    print(f"\n回路已保存: {OUT_CIRCUIT}")

    with open(OUT_REPORT, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"报告已保存: {OUT_REPORT}")


# ============== 主函数 ==============

def main():
    print("=" * 60)
    print("FlyWire 蘑菇体回路提取")
    print("=" * 60)

    # 1. 读取注释数据
    print("\n[1/4] 读取注释数据...")
    ann = pd.read_parquet(ANNOT_FILE)
    print(f"  总注释神经元: {len(ann):,}")

    # 2. 筛选神经元
    print("\n[2/4] 筛选蘑菇体回路神经元...")
    neurons = select_neurons(ann)
    neuron_ids = set(neurons["root_id"].values)

    # 3. 提取连接
    print("\n[3/4] 读取连接数据并提取回路连接...")
    conn_table = feather.read_table(CONN_FILE)
    print(f"  总连接记录: {conn_table.num_rows:,}")
    edges = extract_connections(conn_table, neuron_ids)

    # 4. 验证并保存
    print("\n[4/4] 验证回路完整性并保存...")
    report = validate_circuit(neurons, edges)
    save_circuit(neurons, edges, report)

    print("\n" + "=" * 60)
    print("提取完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
