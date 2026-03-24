#!/usr/bin/env python
"""
结果汇总统计脚本

功能:
- 汇总所有口袋的评估结果
- 生成统计报告
- 输出CSV汇总表格

使用方法:
    python results_summary.py --eval_dir ./generated_mols
    python results_summary.py --eval_dir ./generated_mols --output_dir ./reports
"""

import os
import sys
import argparse
import json
import glob
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm


def load_evaluation_results(eval_dir):
    """加载所有评估结果"""
    results = []

    # 查找所有评估结果文件
    result_files = glob.glob(os.path.join(eval_dir, "**/evaluation_results.json"), recursive=True)

    print(f"发现 {len(result_files)} 个评估结果文件")

    for result_file in tqdm(result_files, desc="加载结果"):
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            results.append(data)
        except Exception as e:
            print(f"加载失败: {result_file}, 错误: {e}")

    return results


def extract_metrics(results):
    """从结果中提取指标"""
    all_molecules = []

    for pocket_result in results:
        pocket_id = pocket_result.get("pocket_id", "unknown")

        for mol_result in pocket_result.get("results", []):
            if mol_result.get("status") != "success":
                continue

            record = {
                "pocket_id": pocket_id,
                "mol_name": mol_result.get("mol_name", ""),
                "smiles": mol_result.get("smiles", ""),
            }

            # 化学性质
            chem = mol_result.get("chem_results", {})
            record["qed"] = chem.get("qed", np.nan)
            record["sa"] = chem.get("sa", np.nan)
            record["logp"] = chem.get("logp", np.nan)
            record["lipinski"] = chem.get("lipinski", 0)
            record["atom_num"] = chem.get("atom_num", 0)

            # 对接分数
            record["vina_score"] = mol_result.get("vina_score", np.nan)
            record["vina_minimize"] = mol_result.get("vina_minimize", np.nan)

            # 构象指标
            record["strain"] = mol_result.get("strain", np.nan)

            all_molecules.append(record)

    return pd.DataFrame(all_molecules)


def compute_statistics(df):
    """计算统计指标"""
    stats = {}

    metrics = ["qed", "sa", "logp", "vina_score", "vina_minimize", "strain"]

    for metric in metrics:
        if metric in df.columns:
            values = df[metric].dropna()
            if len(values) > 0:
                stats[metric] = {
                    "count": len(values),
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "median": float(values.median()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "q25": float(values.quantile(0.25)),
                    "q75": float(values.quantile(0.75)),
                }

    # 特殊统计
    if "vina_score" in df.columns:
        vina_values = df["vina_score"].dropna()
        if len(vina_values) > 0:
            stats["vina_score"]["neg_ratio"] = float((vina_values < 0).sum() / len(vina_values))
            stats["vina_score"]["good_ratio"] = float((vina_values < -7).sum() / len(vina_values))

    if "qed" in df.columns:
        qed_values = df["qed"].dropna()
        if len(qed_values) > 0:
            stats["qed"]["high_ratio"] = float((qed_values > 0.5).sum() / len(qed_values))

    if "sa" in df.columns:
        sa_values = df["sa"].dropna()
        if len(sa_values) > 0:
            stats["sa"]["low_ratio"] = float((sa_values < 5).sum() / len(sa_values))

    if "strain" in df.columns:
        strain_values = df["strain"].dropna()
        if len(strain_values) > 0:
            stats["strain"]["low_ratio"] = float((strain_values < 20).sum() / len(strain_values))

    return stats


def compute_pocket_statistics(df):
    """按口袋计算统计指标"""
    pocket_stats = []

    for pocket_id, group in df.groupby("pocket_id"):
        stat = {
            "pocket_id": pocket_id,
            "num_molecules": len(group),
        }

        # 计算各指标的平均值
        for metric in ["qed", "sa", "vina_score", "vina_minimize", "strain"]:
            if metric in group.columns:
                values = group[metric].dropna()
                if len(values) > 0:
                    stat[f"{metric}_mean"] = float(values.mean())
                    stat[f"{metric}_median"] = float(values.median())

        # 特殊比例
        if "vina_score" in group.columns:
            vina_values = group["vina_score"].dropna()
            if len(vina_values) > 0:
                stat["vina_neg_ratio"] = float((vina_values < 0).sum() / len(vina_values))

        pocket_stats.append(stat)

    return pd.DataFrame(pocket_stats)


def generate_report(df, stats, pocket_df, output_dir):
    """生成报告"""
    report = []
    report.append("=" * 60)
    report.append("MolCRAFT 分子生成评估报告")
    report.append("=" * 60)
    report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    report.append("1. 总体统计")
    report.append("-" * 40)
    report.append(f"总分子数: {len(df)}")
    report.append(f"口袋数: {df['pocket_id'].nunique()}")
    report.append("")

    report.append("2. 化学性质指标")
    report.append("-" * 40)
    if "qed" in stats:
        report.append(f"QED (药物相似性):")
        report.append(f"  均值: {stats['qed']['mean']:.3f}")
        report.append(f"  中位数: {stats['qed']['median']:.3f}")
        report.append(f"  高质量比例(>0.5): {stats['qed'].get('high_ratio', 0)*100:.1f}%")
    if "sa" in stats:
        report.append(f"SA (合成可及性):")
        report.append(f"  均值: {stats['sa']['mean']:.3f}")
        report.append(f"  中位数: {stats['sa']['median']:.3f}")
        report.append(f"  易合成比例(<5): {stats['sa'].get('low_ratio', 0)*100:.1f}%")
    report.append("")

    report.append("3. 对接指标")
    report.append("-" * 40)
    if "vina_score" in stats:
        report.append(f"Vina Score:")
        report.append(f"  均值: {stats['vina_score']['mean']:.3f}")
        report.append(f"  中位数: {stats['vina_score']['median']:.3f}")
        report.append(f"  负值比例(<0): {stats['vina_score'].get('neg_ratio', 0)*100:.1f}%")
        report.append(f"  高亲和力比例(<-7): {stats['vina_score'].get('good_ratio', 0)*100:.1f}%")
    if "vina_minimize" in stats:
        report.append(f"Vina Minimize:")
        report.append(f"  均值: {stats['vina_minimize']['mean']:.3f}")
        report.append(f"  中位数: {stats['vina_minimize']['median']:.3f}")
    report.append("")

    report.append("4. 构象指标")
    report.append("-" * 40)
    if "strain" in stats:
        report.append(f"Strain (应变能):")
        report.append(f"  均值: {stats['strain']['mean']:.3f}")
        report.append(f"  中位数: {stats['strain']['median']:.3f}")
        report.append(f"  低应变比例(<20): {stats['strain'].get('low_ratio', 0)*100:.1f}%")
    report.append("")

    report.append("5. 分子大小分布")
    report.append("-" * 40)
    if "atom_num" in df.columns:
        atom_counts = df["atom_num"].dropna()
        report.append(f"原子数均值: {atom_counts.mean():.1f}")
        report.append(f"原子数中位数: {atom_counts.median():.1f}")
        report.append(f"原子数范围: {atom_counts.min():.0f} - {atom_counts.max():.0f}")
    report.append("")

    # Top 10 口袋
    report.append("6. Top 10 口袋 (按Vina Score)")
    report.append("-" * 40)
    if "vina_score_mean" in pocket_df.columns:
        top_pockets = pocket_df.nsmallest(10, "vina_score_mean")
        for i, row in top_pockets.iterrows():
            report.append(f"  {row['pocket_id']}: {row.get('vina_score_mean', 'N/A'):.3f} ({row['num_molecules']} 分子)")
    report.append("")

    report.append("=" * 60)

    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description='结果汇总统计')

    parser.add_argument('--eval_dir', type=str, default='./generated_mols',
                        help='评估结果目录')
    parser.add_argument('--output_dir', type=str, default='./reports',
                        help='报告输出目录')

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载结果
    print("加载评估结果...")
    results = load_evaluation_results(args.eval_dir)

    if not results:
        print("未找到评估结果")
        return

    # 提取指标
    print("提取指标...")
    df = extract_metrics(results)

    print(f"共 {len(df)} 个成功生成的分子")

    # 计算统计
    print("计算统计指标...")
    stats = compute_statistics(df)
    pocket_df = compute_pocket_statistics(df)

    # 保存CSV
    print("保存CSV文件...")
    df.to_csv(os.path.join(args.output_dir, "all_molecules.csv"), index=False)
    pocket_df.to_csv(os.path.join(args.output_dir, "pocket_summary.csv"), index=False)

    # 保存统计JSON
    with open(os.path.join(args.output_dir, "statistics.json"), 'w') as f:
        json.dump(stats, f, indent=2)

    # 生成报告
    print("生成报告...")
    report = generate_report(df, stats, pocket_df, args.output_dir)

    # 保存报告
    report_file = os.path.join(args.output_dir, "report.txt")
    with open(report_file, 'w') as f:
        f.write(report)

    # 打印报告
    print("\n" + report)
    print(f"\n报告已保存到: {report_file}")


if __name__ == '__main__':
    main()
