#!/usr/bin/env python
"""
批量生成靶点口袋分子脚本

功能:
- 遍历test_set目录中的所有口袋
- 每个口袋生成指定数量的分子
- 支持多GPU并行处理
- 输出SDF文件到指定目录

使用方法:
    python batch_generate.py --num_samples 1000 --gpu_ids 0,1,2,3
    python batch_generate.py --pocket_list pocket_list.txt --num_samples 100
"""

import os
import sys
import argparse
import json
import subprocess
from pathlib import Path
from datetime import datetime
from multiprocessing import Process, Queue
import time

import torch
from tqdm import tqdm


def get_pockets_from_testset(test_set_dir):
    """从test_set目录提取所有口袋ID"""
    pockets = set()
    test_set_path = Path(test_set_dir)

    # 查找所有PDB文件（递归搜索子目录）
    for pdb_file in test_set_path.glob("**/*.pdb"):
        # 文件名格式: 1NAV_1035771.pdb -> 提取 1NAV_10357
        stem = pdb_file.stem
        pocket_id = stem[:10]
        pockets.add(pocket_id)

    return sorted(list(pockets))


def get_pdb_sdf_files(test_set_dir, pocket_id):
    """获取指定口袋的PDB和SDF文件路径"""
    # test_set中的文件命名格式: {pocket_id}71.pdb, {pocket_id}71.sdf
    pdb_file = os.path.join(test_set_dir, f"{pocket_id}71.pdb")
    sdf_file = os.path.join(test_set_dir, f"{pocket_id}71.sdf")

    # 检查文件是否存在（递归搜索子目录）
    if not os.path.exists(pdb_file):
        # 尝试其他可能的命名
        pdb_files = list(Path(test_set_dir).glob(f"**/{pocket_id}*.pdb"))
        if pdb_files:
            pdb_file = str(pdb_files[0])

    if not os.path.exists(sdf_file):
        sdf_files = list(Path(test_set_dir).glob(f"**/{pocket_id}*.sdf"))
        if sdf_files:
            sdf_file = str(sdf_files[0])

    return pdb_file, sdf_file


def generate_for_pocket(pocket_id, pdb_file, sdf_file, output_dir,
                        ckpt_path, config_path, num_samples, sample_steps,
                        gpu_id, beta1, sigma1_coord, sampling_strategy):
    """为单个口袋生成分子"""

    # 设置GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # 创建输出目录
    pocket_output_dir = os.path.join(output_dir, pocket_id)
    os.makedirs(pocket_output_dir, exist_ok=True)

    # 检查是否已完成
    result_file = os.path.join(pocket_output_dir, "generation_done.json")
    if os.path.exists(result_file):
        print(f"[SKIP] {pocket_id} already completed")
        return True

    # 导入必要的模块
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from pytorch_lightning import seed_everything
    from torch_geometric.loader import DataLoader
    from torch_geometric.transforms import Compose

    from core.config.config import Config
    from core.models.sbdd_train_loop import SBDDTrainLoop
    from core.callbacks.basic import NormalizerCallback
    from core.callbacks.validation_callback_for_sample import DockingTestCallback, OUT_DIR
    import core.utils.transforms as trans
    from core.datasets.utils import PDBProtein, parse_sdf_file
    from core.datasets.pl_data import ProteinLigandData, torchify_dict, FOLLOW_BATCH
    import pytorch_lightning as pl

    try:
        # 加载配置
        cfg = Config(config_path)
        seed_everything(cfg.seed)

        # 设置参数
        cfg.evaluation.protein_path = pdb_file
        cfg.evaluation.ligand_path = sdf_file
        cfg.evaluation.ckpt_path = ckpt_path
        cfg.test_only = True
        cfg.no_wandb = True
        cfg.evaluation.num_samples = num_samples
        cfg.evaluation.sample_steps = sample_steps
        cfg.dynamics.beta1 = beta1
        cfg.dynamics.sigma1_coord = sigma1_coord
        cfg.dynamics.sampling_strategy = sampling_strategy

        device = torch.device(f'cuda:0')
        # 检查蛋白或分子是否有H，没有的话添加H
        
        
        # 加载蛋白质和配体
        protein = PDBProtein(pdb_file, device=device)
        ligand_dict = parse_sdf_file(sdf_file)

        # 提取口袋
        pdb_block_pocket = protein.residues_to_pdb_block(
            protein.query_residues_ligand(ligand_dict, cfg.dynamics.net_config.r_max)
        )
        pocket = PDBProtein(pdb_block_pocket, device=device)
        pocket_dict = pocket.to_dict_atom()

        # 创建数据对象
        data = ProteinLigandData.from_protein_ligand_dicts(
            protein_dict=torchify_dict(pocket_dict),
            ligand_dict=torchify_dict(ligand_dict),
        )
        data.protein_filename = os.path.basename(pdb_file)
        data.ligand_filename = os.path.basename(sdf_file)

        # 变换
        protein_featurizer = trans.FeaturizeProteinAtom()
        ligand_featurizer = trans.FeaturizeLigandAtom(cfg.data.transform.ligand_atom_mode)
        transform = Compose([protein_featurizer, ligand_featurizer])

        cfg.dynamics.protein_atom_feature_dim = protein_featurizer.feature_dim
        cfg.dynamics.ligand_atom_feature_dim = ligand_featurizer.feature_dim

        # 创建数据加载器
        test_set = [transform(data)] * cfg.evaluation.num_samples
        cfg.evaluation.num_samples = 1
        test_loader = DataLoader(
            test_set,
            batch_size=cfg.evaluation.batch_size,
            shuffle=False,
            follow_batch=FOLLOW_BATCH,
            exclude_keys=["ligand_nbh_list"]
        )

        cfg.evaluation.docking_config.protein_root = os.path.dirname(os.path.abspath(pdb_file))

        # 修改输出目录 - 动态修改callback模块的全局变量
        import core.callbacks.validation_callback_for_sample as callback_module
        callback_module.OUT_DIR = pocket_output_dir

        # 创建模型
        model = SBDDTrainLoop(config=cfg)

        # 创建训练器
        trainer = pl.Trainer(
            default_root_dir=cfg.accounting.logdir,
            max_epochs=cfg.train.epochs,
            devices=[0],
            num_sanity_val_steps=0,
            callbacks=[
                NormalizerCallback(normalizer_dict=cfg.data.normalizer_dict),
                DockingTestCallback(
                    dataset=None,
                    atom_decoder=cfg.data.atom_decoder,
                    atom_enc_mode=cfg.data.transform.ligand_atom_mode,
                    atom_type_one_hot=False,
                    single_bond=True,
                    docking_config=cfg.evaluation.docking_config,
                ),
            ],
        )

        # 运行生成
        trainer.test(model, dataloaders=test_loader, ckpt_path=cfg.evaluation.ckpt_path)

        # 记录完成状态
        result = {
            "pocket_id": pocket_id,
            "pdb_file": pdb_file,
            "sdf_file": sdf_file,
            "num_samples": num_samples,
            "sample_steps": sample_steps,
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }

        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)

        return True

    except Exception as e:
        print(f"[ERROR] {pocket_id}: {e}")
        import traceback
        traceback.print_exc()

        # 记录错误
        result = {
            "pocket_id": pocket_id,
            "pdb_file": pdb_file,
            "sdf_file": sdf_file,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "status": "failed"
        }

        error_file = os.path.join(pocket_output_dir, "generation_error.json")
        with open(error_file, 'w') as f:
            json.dump(result, f, indent=2)

        return False


def worker(gpu_id, task_queue, result_queue, args):
    """工作进程"""
    while True:
        task = task_queue.get()
        if task is None:
            break

        pocket_id, pdb_file, sdf_file = task

        success = generate_for_pocket(
            pocket_id, pdb_file, sdf_file,
            args.output_dir, args.ckpt_path, args.config_path,
            args.num_samples, args.sample_steps, gpu_id,
            args.beta1, args.sigma1_coord, args.sampling_strategy
        )

        result_queue.put((pocket_id, success))


def main():
    parser = argparse.ArgumentParser(description='批量生成靶点口袋分子')

    # 输入输出参数
    parser.add_argument('--test_set_dir', type=str, default='./data/test_set',
                        help='test_set目录路径')
    parser.add_argument('--output_dir', type=str, default='./generated_mols',
                        help='输出目录')
    parser.add_argument('--pocket_list', type=str, default=None,
                        help='口袋列表文件(每行一个口袋ID), 不指定则使用全部')

    # 生成参数
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='每个口袋生成的分子数')
    parser.add_argument('--sample_steps', type=int, default=100,
                        help='采样步数')
    parser.add_argument('--beta1', type=float, default=1.5,
                        help='BFN beta1参数')
    parser.add_argument('--sigma1_coord', type=float, default=0.03,
                        help='坐标噪声参数')
    parser.add_argument('--sampling_strategy', type=str, default='end_back_pmf',
                        choices=['vanilla', 'end_back_pmf'],
                        help='采样策略')

    # 模型参数
    parser.add_argument('--ckpt_path', type=str, default='./checkpoints/last.ckpt',
                        help='模型检查点路径')
    parser.add_argument('--config_path', type=str, default='./configs/default.yaml',
                        help='配置文件路径')

    # GPU参数
    parser.add_argument('--gpu_ids', type=str, default='0',
                        help='GPU ID列表, 用逗号分隔 (如: 0,1,2,3)')
    parser.add_argument('--sequential', action='store_true',
                        help='顺序执行而非并行')

    args = parser.parse_args()

    # 解析GPU列表
    gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
    num_gpus = len(gpu_ids)

    # 获取口袋列表
    if args.pocket_list and os.path.exists(args.pocket_list):
        with open(args.pocket_list, 'r') as f:
            pockets = [line.strip() for line in f if line.strip()]
    else:
        pockets = get_pockets_from_testset(args.test_set_dir)

    print(f"共发现 {len(pockets)} 个口袋")
    print(f"每个口袋生成 {args.num_samples} 个分子")
    print(f"使用GPU: {gpu_ids}")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 保存配置
    config = {
        "test_set_dir": args.test_set_dir,
        "output_dir": args.output_dir,
        "num_samples": args.num_samples,
        "sample_steps": args.sample_steps,
        "gpu_ids": gpu_ids,
        "pockets": pockets,
        "timestamp": datetime.now().isoformat()
    }
    with open(os.path.join(args.output_dir, "batch_config.json"), 'w') as f:
        json.dump(config, f, indent=2)

    # 准备任务列表
    tasks = []
    for pocket_id in pockets:
        pdb_file, sdf_file = get_pdb_sdf_files(args.test_set_dir, pocket_id)
        if os.path.exists(pdb_file) and os.path.exists(sdf_file):
            tasks.append((pocket_id, pdb_file, sdf_file))
        else:
            print(f"[WARN] 跳过 {pocket_id}: 文件不存在 (PDB: {pdb_file}, SDF: {sdf_file})")

    print(f"有效任务数: {len(tasks)}")

    if args.sequential or num_gpus == 1:
        # 顺序执行
        for idx, (pocket_id, pdb_file, sdf_file) in enumerate(tqdm(tasks, desc="生成进度")):
            gpu_id = gpu_ids[idx % num_gpus]
            print(f"\n[{idx+1}/{len(tasks)}] 处理 {pocket_id} (GPU {gpu_id})")
            generate_for_pocket(
                pocket_id, pdb_file, sdf_file,
                args.output_dir, args.ckpt_path, args.config_path,
                args.num_samples, args.sample_steps, gpu_id,
                args.beta1, args.sigma1_coord, args.sampling_strategy
            )
    else:
        # 并行执行
        task_queue = Queue()
        result_queue = Queue()

        for task in tasks:
            task_queue.put(task)

        # 添加结束标记
        for _ in range(num_gpus):
            task_queue.put(None)

        # 启动工作进程
        processes = []
        for gpu_id in gpu_ids:
            p = Process(target=worker, args=(gpu_id, task_queue, result_queue, args))
            p.start()
            processes.append(p)

        # 等待完成
        completed = 0
        with tqdm(total=len(tasks), desc="生成进度") as pbar:
            while completed < len(tasks):
                pocket_id, success = result_queue.get()
                completed += 1
                pbar.update(1)
                status = "成功" if success else "失败"
                pbar.set_postfix({"最近": f"{pocket_id}({status})"})

        # 等待所有进程结束
        for p in processes:
            p.join()

    print("\n批量生成完成!")
    print(f"结果保存在: {args.output_dir}")


if __name__ == '__main__':
    #需要确保输入的蛋白和分子都带H，否则生成的分子可能不正确
    main()
