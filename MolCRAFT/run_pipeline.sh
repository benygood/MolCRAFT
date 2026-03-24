#!/bin/bash
#
# MolCRAFT 批量生成和评估流程
#
# 使用方法:
#   ./run_pipeline.sh                    # 完整运行
#   ./run_pipeline.sh --test             # 测试模式(仅3个口袋)
#   ./run_pipeline.sh --generate-only    # 仅生成
#   ./run_pipeline.sh --evaluate-only    # 仅评估
#

set -e

# 配置参数
NUM_SAMPLES=${NUM_SAMPLES:-1000}
SAMPLE_STEPS=${SAMPLE_STEPS:-100}
GPU_IDS=${GPU_IDS:-0}
EXHAUSTIVENESS=${EXHAUSTIVENESS:-16}
TEST_SET_DIR=${TEST_SET_DIR:-"./data/test_set"}
OUTPUT_DIR=${OUTPUT_DIR:-"./generated_mols"}
CKPT_PATH=${CKPT_PATH:-"./checkpoints/last.ckpt"}
CONFIG_PATH=${CONFIG_PATH:-"./configs/default.yaml"}

# 测试模式
TEST_MODE=false
GENERATE_ONLY=false
EVALUATE_ONLY=false

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            TEST_MODE=true
            NUM_SAMPLES=10
            shift
            ;;
        --generate-only)
            GENERATE_ONLY=true
            shift
            ;;
        --evaluate-only)
            EVALUATE_ONLY=true
            shift
            ;;
        --num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --gpu-ids)
            GPU_IDS="$2"
            shift 2
            ;;
        --help)
            echo "使用方法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --test              测试模式(仅3个口袋, 10个样本)"
            echo "  --generate-only     仅运行生成步骤"
            echo "  --evaluate-only     仅运行评估步骤"
            echo "  --num-samples N     每个口袋生成N个分子(默认1000)"
            echo "  --gpu-ids IDS       GPU ID列表(默认0)"
            echo "  --help              显示帮助信息"
            echo ""
            echo "环境变量:"
            echo "  NUM_SAMPLES         每个口袋生成分子数"
            echo "  SAMPLE_STEPS        采样步数"
            echo "  GPU_IDS             GPU ID列表"
            echo "  EXHAUSTIVENESS      Vina对接搜索强度"
            echo "  TEST_SET_DIR        测试集目录"
            echo "  OUTPUT_DIR          输出目录"
            echo "  CKPT_PATH           模型检查点路径"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "MolCRAFT 批量生成和评估流程"
echo "=============================================="
echo ""
echo "配置:"
echo "  每口袋分子数: ${NUM_SAMPLES}"
echo "  采样步数: ${SAMPLE_STEPS}"
echo "  GPU IDs: ${GPU_IDS}"
echo "  测试模式: ${TEST_MODE}"
echo "  输出目录: ${OUTPUT_DIR}"
echo ""

# 创建测试口袋列表
if [ "$TEST_MODE" = true ]; then
    echo "[测试模式] 创建3个测试口袋列表..."
    POCKET_LIST=$(mktemp)
    find ${TEST_SET_DIR} -name "*.pdb" | head -3 | xargs -I {} basename {} | cut -c1-10 | sort -u > ${POCKET_LIST}
    POCKET_LIST_ARG="--pocket_list ${POCKET_LIST}"
    echo "测试口袋:"
    cat ${POCKET_LIST}
    echo ""
else
    POCKET_LIST_ARG=""
fi

# 步骤1: 生成分子
if [ "$EVALUATE_ONLY" = false ]; then
    echo "=============================================="
    echo "步骤1: 批量生成分子"
    echo "=============================================="
    echo ""

    GENERATE_CMD="python batch_generate.py \
        --test_set_dir ${TEST_SET_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --num_samples ${NUM_SAMPLES} \
        --sample_steps ${SAMPLE_STEPS} \
        --gpu_ids ${GPU_IDS} \
        --ckpt_path ${CKPT_PATH} \
        --config_path ${CONFIG_PATH} \
        --sequential \
        ${POCKET_LIST_ARG}"

    echo "执行: ${GENERATE_CMD}"
    echo ""

    ${GENERATE_CMD}

    echo ""
    echo "生成完成!"
    echo ""
fi

# 步骤2: 评估分子
if [ "$GENERATE_ONLY" = false ]; then
    echo "=============================================="
    echo "步骤2: 批量评估分子"
    echo "=============================================="
    echo ""

    EVALUATE_CMD="python batch_evaluate.py \
        --mol_dir ${OUTPUT_DIR} \
        --protein_root ${TEST_SET_DIR} \
        --exhaustiveness ${EXHAUSTIVENESS} \
        --sequential"

    echo "执行: ${EVALUATE_CMD}"
    echo ""

    ${EVALUATE_CMD}

    echo ""
    echo "评估完成!"
    echo ""
fi

# 步骤3: 汇总结果
if [ "$GENERATE_ONLY" = false ]; then
    echo "=============================================="
    echo "步骤3: 汇总统计结果"
    echo "=============================================="
    echo ""

    SUMMARY_CMD="python results_summary.py \
        --eval_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR}/reports"

    echo "执行: ${SUMMARY_CMD}"
    echo ""

    ${SUMMARY_CMD}

    echo ""
    echo "汇总完成!"
    echo ""
fi

# 清理临时文件
if [ "$TEST_MODE" = true ] && [ -n "${POCKET_LIST}" ] && [ -f "${POCKET_LIST}" ]; then
    rm -f ${POCKET_LIST}
fi

echo "=============================================="
echo "全部完成!"
echo "=============================================="
echo ""
echo "输出文件:"
echo "  生成分子: ${OUTPUT_DIR}/{pocket_id}/*.sdf"
echo "  评估结果: ${OUTPUT_DIR}/{pocket_id}/evaluation_results.json"
echo "  汇总报告: ${OUTPUT_DIR}/reports/"
echo ""
