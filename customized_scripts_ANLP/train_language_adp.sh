#!/bin/bash
#SBATCH --job-name=mlm-training          # Job name
#SBATCH --output=mlm-training-%j.out     # Standard output file, %j is the job ID
#SBATCH --error=mlm-training-%j.err      # Standard error file, %j is the job ID
#SBATCH --ntasks=1                       # Number of tasks (typically 1 for this kind of job)
#SBATCH --cpus-per-task=4                # Number of CPU cores per task
#SBATCH --mem=128GB                       # Amount of memory requested
#SBATCH --time=24:00:00                  # Max time (hh:mm:ss)
#SBATCH --gres=gpu:1                     # Request 1 GPU
#SBATCH --partition=react                # Optional: Specify the GPU partition if needed

# Load Python 3.9 module 
module load python/3.9

# activate virtual environment
source /user/zhuojing.huang/u14729/myenv/bin/activate

export TRAIN_FILE="/user/zhuojing.huang/u14729/test/th_lo_my.json"
export VALIDATION_FILE="/user/zhuojing.huang/u14729/test/th_lo_my_val.json"

python run_mlm.py \
    --model_name_or_path xlm-roberta-base \
    --train_file $TRAIN_FILE \
    --validation_file $VALIDATION_FILE \
    --do_train \
    --do_eval \
    --learning_rate 1e-4 \
    --num_train_epochs 3.0 \
    --per_device_eval_batch_size 32 \
    --per_device_train_batch_size 32 \
    --output_dir "/user/zhuojing.huang/u14729/test/output_th_km_my" \
    --train_adapter \
    --adapter_config "seq_bn_inv" \
    --save_steps 5000 \
    --logging_steps 5000  \
    --save_total_limit 3  # Keeps only the last 3 checkpoints

