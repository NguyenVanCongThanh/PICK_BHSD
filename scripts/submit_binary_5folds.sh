#!/bin/bash
#SBATCH --job-name=pick_binary_5f
#SBATCH --array=0-4%2
#SBATCH --output=/home/u001015/experiments/logs_pick_binary_fold_%a_%j.log
#SBATCH --error=/home/u001015/experiments/logs_pick_binary_fold_%a_%j.err
#SBATCH --partition=gpu-queue
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00

# 1. Kích hoạt môi trường ảo
source /home/u001015/.venv/nnunet/bin/activate

# 2. Lấy Fold ID từ Slurm Array (0, 1, 2, 3, 4)
FOLD_ID=$SLURM_ARRAY_TASK_ID

# 3. Chuyển vào thư mục PICK
cd /home/u001015/PICK

# 4. Thiết lập thư mục lưu logs/experiments
mkdir -p /home/u001015/experiments

echo "=========================================================================="
echo "=== BẮT ĐẦU CHẠY PICK BINARY FOLD $FOLD_ID (Slurm Job ID: $SLURM_JOB_ID, Task: $FOLD_ID) ==="
echo "=========================================================================="

# 5. Khởi chạy huấn luyện
python BHSD_binary_PICK_train.py \
    --root_path /home/u001015/notebook/preprocessed_bhsd_pick/ \
    --exp BHSD_binary_PICK_5folds \
    --fold $FOLD_ID \
    --model VNet \
    --labelnum 15 \
    --max_samples 2200 \
    --batch_size 4 \
    --labeled_bs 2 \
    --gpu 0

echo "=== HOÀN TẤT HUẤN LUYỆN PICK BINARY CHO FOLD $FOLD_ID ==="
