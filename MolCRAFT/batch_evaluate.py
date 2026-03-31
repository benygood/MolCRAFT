#!/usr/bin/env python
"""
批量评估生成分子脚本

功能:
- 读取生成的SDF文件
- 计算标准评估指标:
  - QED, SA (化学性质)
  - vina_score, vina_minimize (对接)
  - strain (应变能)
- 输出JSON格式结果

使用方法:
    python batch_evaluate.py --mol_dir ./generated_mols --protein_root ./data/test_set
    python batch_evaluate.py --mol_dir ./generated_mols --exhaustiveness 16
"""

import os
import sys
import argparse
import json
import glob
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool
from functools import partial

import numpy as np
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Descriptors, Crippen, DataStructs, rdFingerprintGenerator
from tqdm import tqdm

# 禁用RDKit警告
RDLogger.DisableLog('rdApp.*')

# ECFP4 指纹生成器 (用于活性分子相似性计算)
MGEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def evaluate_single_mol(sdf_file, protein_root, exhaustiveness, active_fps=None, active_smis=None):
    """评估单个分子

    Args:
        sdf_file: SDF文件路径
        protein_root: 蛋白质文件根目录
        exhaustiveness: Vina搜索强度
        active_fps: 活性分子ECFP4指纹列表 (可选)
        active_smis: 活性分子SMILES列表 (可选)
    """
    result = {
        "sdf_file": sdf_file,
        "status": "failed",
        "error": None
    }

    try:
        # 读取分子
        suppl = Chem.SDMolSupplier(sdf_file, sanitize=False, removeHs=False)
        mol = suppl[0]

        if mol is None:
            result["error"] = "无法读取分子"
            return result

        # 获取分子名称
        mol_name = mol.GetProp('_Name') if mol.HasProp('_Name') else Path(sdf_file).stem
        result["mol_name"] = mol_name

        # SMILES
        try:
            Chem.SanitizeMol(mol)
            smiles = Chem.MolToSmiles(mol)
            if '.' in smiles:
                result["error"] = "分子不完整(包含片段)"
                return result
            result["smiles"] = smiles
        except Exception as e:
            result["error"] = f"SMILES转换失败: {e}"
            return result

        # 1. 化学性质评估
        from core.evaluation.utils import scoring_func
        try:
            chem_results = scoring_func.get_chem(mol)
            result["chem_results"] = {
                "qed": chem_results.get('qed', np.nan),
                "sa": chem_results.get('sa', np.nan),
                "logp": chem_results.get('logp', np.nan),
                "lipinski": chem_results.get('lipinski', 0),
                "atom_num": mol.GetNumAtoms(),
                # 新增指标 (参考 prop_eval.ipynb)
                "mw": Descriptors.MolWt(mol),                    # 分子量
                "heavy_atoms": mol.GetNumHeavyAtoms(),           # 重原子数
                "rotatable_bonds": Descriptors.NumRotatableBonds(mol),  # 可旋转键
                "tpsa": Descriptors.TPSA(mol),                   # 拓扑极性表面积
            }
        except Exception as e:
            result["chem_results"] = {"error": str(e)}

        # 1.5 活性分子相似性评估 (可选，需要提供活性分子文件)
        if active_fps and len(active_fps) > 0:
            try:
                gen_fp = MGEN.GetFingerprint(mol)
                sims = DataStructs.BulkTanimotoSimilarity(gen_fp, active_fps)
                max_sim = max(sims) if sims else 0.0
                result["max_ecfp4_tanimoto"] = max_sim
                if active_smis:
                    result["most_similar_active"] = active_smis[sims.index(max_sim)]
            except Exception as e:
                result["similarity_error"] = str(e)

        # 2. 对接评估
        try:
            from core.evaluation.docking_vina import VinaDockingTask

            # 查找对应的蛋白质文件
            ligand_fn = mol_name
            if '/' in ligand_fn or '\\' in ligand_fn:
                ligand_fn = ligand_fn.split('/')[-1].split('\\')[-1]

            # 尝试不同的蛋白质文件命名格式
            protein_fn = None
            possible_names = [
                os.path.join(protein_root, f"{ligand_fn[:10]}71.pdb"),
                os.path.join(protein_root, f"{ligand_fn[:10]}.pdb"),
                os.path.join(protein_root, f"{ligand_fn[:10]}_with_h.pdb"),
            ]

            for pn in possible_names:
                if os.path.exists(pn):
                    protein_fn = pn
                    break

            if protein_fn is None:
                # 搜索匹配的PDB文件
                pdb_pattern = os.path.join(protein_root, f"{ligand_fn[:10]}*.pdb")
                pdb_files = glob.glob(pdb_pattern)
                if pdb_files:
                    protein_fn = pdb_files[0]

            if protein_fn and os.path.exists(protein_fn):
                # 获取分子坐标
                pos = mol.GetConformer(0).GetPositions()

                vina_task = VinaDockingTask.from_generated_mol(
                    mol, ligand_fn, pos=pos, protein_root=protein_root
                )

                # score_only模式
                score_only_results = vina_task.run(mode='score_only', exhaustiveness=exhaustiveness)
                result["vina_score"] = score_only_results[0]['affinity']

                # minimize模式
                minimize_results = vina_task.run(mode='minimize', exhaustiveness=exhaustiveness)
                result["vina_minimize"] = minimize_results[0]['affinity']

                # 计算效率指标 (参考 prop_eval.ipynb)
                # 这些指标在 chem_results 可用时计算
                if "chem_results" in result and "error" not in result["chem_results"]:
                    tpsa = result["chem_results"].get("tpsa", 0)
                    logp = result["chem_results"].get("logp", 0)
                    heavy_atoms = result["chem_results"].get("heavy_atoms", 0)
                    affinity = result["vina_minimize"]

                    # SEI: 表面效率指数 = -affinity / tpsa (越大越好)
                    if tpsa > 0:
                        result["sei"] = -affinity / tpsa
                    else:
                        result["sei"] = np.nan

                    # LLE: 配体亲脂效率 = -affinity - logp (越大越好)
                    result["lle"] = -affinity - logp

                    # LBE: 配体结合效率 = affinity / heavy_atoms (越小越好，注意affinity为负)
                    if heavy_atoms > 0:
                        result["lbe"] = affinity / heavy_atoms
                    else:
                        result["lbe"] = np.nan
            else:
                result["vina_error"] = f"找不到蛋白质文件: {ligand_fn[:10]}"

        except Exception as e:
            result["vina_error"] = str(e)

        # 3. PoseCheck评估 (strain)
        try:
            from posecheck import PoseCheck

            pc = PoseCheck()
            pc.load_ligands_from_mols([mol])
            strain = pc.calculate_strain_energy()[0]
            result["strain"] = float(strain) if strain is not None else np.nan

        except Exception as e:
            result["strain_error"] = str(e)

        result["status"] = "success"

    except Exception as e:
        result["error"] = str(e)
        import traceback
        result["traceback"] = traceback.format_exc()

    return result


def evaluate_pocket(pocket_dir, protein_root, exhaustiveness, activesmi_dir=None):
    """评估单个口袋的所有分子

    Args:
        pocket_dir: 口袋目录路径
        protein_root: 蛋白质文件根目录
        exhaustiveness: Vina搜索强度
        activesmi_dir: 活性分子SMILES文件目录 (可选)
    """
    pocket_id = Path(pocket_dir).name
    result_file = os.path.join(pocket_dir, "evaluation_results.json")

    # 检查是否已评估
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            existing = json.load(f)
        if existing.get("status") == "completed":
            print(f"[SKIP] {pocket_id} already evaluated")
            return existing

    # 查找所有SDF文件
    sdf_files = glob.glob(os.path.join(pocket_dir, "*.sdf"))

    if not sdf_files:
        return {
            "pocket_id": pocket_id,
            "status": "no_molecules",
            "num_molecules": 0
        }

    # 加载活性分子 (用于相似性计算)
    active_fps = []
    active_smis = []
    if activesmi_dir:
        # pocket_id 前4个字符为 PDB ID
        pdb_id = pocket_id[:4].lower()
        active_smi_file = os.path.join(activesmi_dir, f"{pdb_id}.smi")
        if os.path.exists(active_smi_file):
            try:
                with open(active_smi_file, 'r') as f:
                    active_smis = [line.rstrip('\n').strip() for line in f if line.strip()]
                for smi in active_smis:
                    mol = Chem.MolFromSmiles(smi)
                    if mol is not None:
                        fp = MGEN.GetFingerprint(mol)
                        active_fps.append(fp)
                print(f"  [INFO] 加载 {len(active_fps)} 个活性分子指纹 ({pocket_id})")
            except Exception as e:
                print(f"  [WARN] 加载活性分子失败: {e}")

    results = []
    for sdf_file in tqdm(sdf_files, desc=f"评估 {pocket_id}", leave=False):
        result = evaluate_single_mol(sdf_file, protein_root, exhaustiveness, active_fps, active_smis)
        result["pocket_id"] = pocket_id
        results.append(result)

    # 统计结果
    success_count = sum(1 for r in results if r["status"] == "success")

    summary = {
        "pocket_id": pocket_id,
        "status": "completed",
        "num_molecules": len(results),
        "num_success": success_count,
        "success_rate": success_count / len(results) if results else 0,
        "results": results,
        "timestamp": datetime.now().isoformat()
    }

    # 计算汇总统计
    qed_values = [r["chem_results"]["qed"] for r in results
                  if r["status"] == "success" and "chem_results" in r and "qed" in r.get("chem_results", {})]
    sa_values = [r["chem_results"]["sa"] for r in results
                 if r["status"] == "success" and "chem_results" in r and "sa" in r.get("chem_results", {})]
    vina_scores = [r["vina_score"] for r in results
                   if "vina_score" in r and r["vina_score"] is not None]
    vina_mins = [r["vina_minimize"] for r in results
                 if "vina_minimize" in r and r["vina_minimize"] is not None]
    strains = [r["strain"] for r in results
               if "strain" in r and r["strain"] is not None and not np.isnan(r.get("strain", np.nan))]

    # 新增指标统计
    mw_values = [r["chem_results"]["mw"] for r in results
                 if r["status"] == "success" and "chem_results" in r and "mw" in r.get("chem_results", {})]
    heavy_atoms_values = [r["chem_results"]["heavy_atoms"] for r in results
                          if r["status"] == "success" and "chem_results" in r and "heavy_atoms" in r.get("chem_results", {})]
    tpsa_values = [r["chem_results"]["tpsa"] for r in results
                   if r["status"] == "success" and "chem_results" in r and "tpsa" in r.get("chem_results", {})]
    rotatable_values = [r["chem_results"]["rotatable_bonds"] for r in results
                        if r["status"] == "success" and "chem_results" in r and "rotatable_bonds" in r.get("chem_results", {})]
    sei_values = [r["sei"] for r in results
                  if "sei" in r and r["sei"] is not None and not np.isnan(r.get("sei", np.nan))]
    lle_values = [r["lle"] for r in results
                  if "lle" in r and r["lle"] is not None and not np.isnan(r.get("lle", np.nan))]
    lbe_values = [r["lbe"] for r in results
                  if "lbe" in r and r["lbe"] is not None and not np.isnan(r.get("lbe", np.nan))]
    # 活性分子相似性统计
    similarity_values = [r["max_ecfp4_tanimoto"] for r in results
                         if "max_ecfp4_tanimoto" in r and r["max_ecfp4_tanimoto"] is not None]

    if qed_values:
        summary["qed_mean"] = float(np.mean(qed_values))
        summary["qed_median"] = float(np.median(qed_values))
    if sa_values:
        summary["sa_mean"] = float(np.mean(sa_values))
        summary["sa_median"] = float(np.median(sa_values))
    if vina_scores:
        summary["vina_score_mean"] = float(np.mean(vina_scores))
        summary["vina_score_median"] = float(np.median(vina_scores))
        summary["vina_score_neg_ratio"] = float(sum(1 for v in vina_scores if v < 0) / len(vina_scores))
    if vina_mins:
        summary["vina_min_mean"] = float(np.mean(vina_mins))
        summary["vina_min_median"] = float(np.median(vina_mins))
    if strains:
        summary["strain_mean"] = float(np.mean(strains))
        summary["strain_median"] = float(np.median(strains))

    # 新增指标汇总
    if mw_values:
        summary["mw_mean"] = float(np.mean(mw_values))
        summary["mw_median"] = float(np.median(mw_values))
    if heavy_atoms_values:
        summary["heavy_atoms_mean"] = float(np.mean(heavy_atoms_values))
        summary["heavy_atoms_median"] = float(np.median(heavy_atoms_values))
    if tpsa_values:
        summary["tpsa_mean"] = float(np.mean(tpsa_values))
        summary["tpsa_median"] = float(np.median(tpsa_values))
    if rotatable_values:
        summary["rotatable_bonds_mean"] = float(np.mean(rotatable_values))
        summary["rotatable_bonds_median"] = float(np.median(rotatable_values))
    if sei_values:
        summary["sei_mean"] = float(np.mean(sei_values))
        summary["sei_median"] = float(np.median(sei_values))
    if lle_values:
        summary["lle_mean"] = float(np.mean(lle_values))
        summary["lle_median"] = float(np.median(lle_values))
    if lbe_values:
        summary["lbe_mean"] = float(np.mean(lbe_values))
        summary["lbe_median"] = float(np.median(lbe_values))

    # 活性分子相似性汇总
    if similarity_values:
        summary["ecfp4_similarity_mean"] = float(np.mean(similarity_values))
        summary["ecfp4_similarity_median"] = float(np.median(similarity_values))
        summary["ecfp4_similarity_max"] = float(np.max(similarity_values))
        # 高相似性比例 (>0.5, >0.7)
        summary["ecfp4_similarity_gt05_ratio"] = sum(1 for v in similarity_values if v > 0.5) / len(similarity_values)
        summary["ecfp4_similarity_gt07_ratio"] = sum(1 for v in similarity_values if v > 0.7) / len(similarity_values)

    # 类药性组合指标: QED > 0.5 且 SA < 5 的比例
    druglike_count = sum(1 for r in results
                         if r["status"] == "success" and "chem_results" in r
                         and r["chem_results"].get("qed", 0) > 0.5
                         and r["chem_results"].get("sa", 10) < 5)
    summary["druglike_ratio"] = druglike_count / success_count if success_count > 0 else 0

    # 保存结果
    with open(result_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


def main():
    parser = argparse.ArgumentParser(description='批量评估生成分子')

    # 输入输出参数
    parser.add_argument('--mol_dir', type=str, default='./generated_mols',
                        help='生成分子目录')
    parser.add_argument('--protein_root', type=str, default='./data/test_set',
                        help='蛋白质文件根目录')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='评估结果输出目录(默认与mol_dir相同)')

    # 评估参数
    parser.add_argument('--exhaustiveness', type=int, default=16,
                        help='Vina对接搜索强度')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='并行工作进程数')
    parser.add_argument('--sequential', action='store_true',
                        help='顺序执行而非并行')
    parser.add_argument('--activesmi_dir', type=str, default=None,
                        help='活性分子SMILES文件目录 (可选，用于计算ECFP4相似性)')
    parser.add_argument('--activesmi_dir', type=str, default=None,
                        help='活性分子SMILES文件目录 (可选，用于计算ECFP4相似性)')

    args = parser.parse_args()

    # 设置输出目录
    output_dir = args.output_dir or args.mol_dir
    os.makedirs(output_dir, exist_ok=True)

    # 查找所有口袋目录
    pocket_dirs = []
    for item in Path(args.mol_dir).iterdir():
        if item.is_dir():
            # 检查是否包含SDF文件
            sdf_files = list(item.glob("*.sdf"))
            if sdf_files:
                pocket_dirs.append(str(item))

    print(f"发现 {len(pocket_dirs)} 个口袋目录")

    if not pocket_dirs:
        print("未找到有效的口袋目录")
        return

    # 评估
    all_results = []

    if args.sequential:
        for pocket_dir in tqdm(pocket_dirs, desc="评估进度"):
            result = evaluate_pocket(pocket_dir, args.protein_root, args.exhaustiveness, args.activesmi_dir)
            all_results.append(result)
    else:
        # 并行评估
        with Pool(args.num_workers) as pool:
            func = partial(evaluate_pocket,
                          protein_root=args.protein_root,
                          exhaustiveness=args.exhaustiveness,
                          activesmi_dir=args.activesmi_dir)
            all_results = list(tqdm(pool.imap(func, pocket_dirs),
                                   total=len(pocket_dirs),
                                   desc="评估进度"))

    # 汇总所有结果
    total_summary = {
        "num_pockets": len(pocket_dirs),
        "timestamp": datetime.now().isoformat(),
        "pockets": all_results
    }

    # 计算全局统计
    all_qed = []
    all_sa = []
    all_vina_score = []
    all_vina_min = []
    all_strain = []

    for r in all_results:
        all_qed.extend([r["results"][i]["chem_results"]["qed"]
                       for i in range(len(r.get("results", [])))
                       if r["results"][i].get("status") == "success"
                       and "chem_results" in r["results"][i]
                       and "qed" in r["results"][i]["chem_results"]])
        # 类似地收集其他指标...

    # 保存全局结果
    summary_file = os.path.join(output_dir, "all_evaluation_results.json")
    with open(summary_file, 'w') as f:
        json.dump(total_summary, f, indent=2, default=str)

    print(f"\n评估完成!")
    print(f"结果保存在: {output_dir}")
    print(f"汇总文件: {summary_file}")

    # 打印简要统计
    success_pockets = sum(1 for r in all_results if r.get("status") == "completed")
    print(f"\n成功评估的口袋数: {success_pockets}/{len(pocket_dirs)}")


if __name__ == '__main__':
    main()
