#!/bin/bash

# 如果任何一个命令失败，立即停止执行
set -e

echo "开始执行流水线任务..."

# 1. 运行训练脚本
echo "--------------------------------------"
echo "[Step 1/3] 正在开始训练 (train.py)..."
python train.py

# 2. 运行推理脚本
echo "--------------------------------------"
echo "[Step 2/3] 正在生成结果 (inference_fixed.py)..."
python inference_fixed.py

# 3. 运行评估脚本
echo "--------------------------------------"
echo "[Step 3/3] 正在计算 CLIPScore (eval_clipscore.py)..."
python eval_clipscore.py

echo "--------------------------------------"
echo "所有任务已完成！请查看 results.json 和终端输出的分数。"