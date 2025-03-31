#!/bin/bash
#SBATCH --job-name=mlm-training          # Job name
#SBATCH --output=mlm-training-%j.out     # Standard output file
#SBATCH --error=mlm-training-%j.err      # Standard error file
#SBATCH --ntasks=1                       # Number of tasks (single process)
#SBATCH --cpus-per-task=4                # Number of CPU cores per task
#SBATCH --mem=128GB                      # Memory allocation
#SBATCH --time=24:00:00                  # Max execution time
#SBATCH --gres=gpu:1                     # Request 1 GPU
#SBATCH --partition=grete:interactive    # Specify GPU partition if necessary

# Load Python 3.9 module
module load python/3.9

# Activate the virtual environment
source /user/zhuojing.huang/u14729/myenv/bin/activate

# Ensure GPU assignment
export CUDA_VISIBLE_DEVICES=0  # Ensure PyTorch uses the correct GPU
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

# Clear GPU cache
echo "Clearing GPU cache before training..."
python -c "import torch; torch.cuda.empty_cache()"


# Debugging: Check PyTorch device
python -c "import torch; print(f'Available GPUs: {torch.cuda.device_count()}'); print(f'Default GPU: {torch.cuda.current_device()} ({torch.cuda.get_device_name(0)})')"

export TASK_NAME=mnli  # adjust the task name for specific task; defaul is English

python run_glue.py \
  --model_name_or_path xlm-roberta-base \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 1e-4 \
  --num_train_epochs 3.0 \
  --output_dir "/user/zhuojing.huang/u14729/test/glue_mnli"  \
  --overwrite_output_dir \
  --train_adapter \
  --adapter_config seq_bn \
  --save_steps 5000 \
  --logging_steps 5000 \
  --save_total_limit 3  # Keeps only the last 3 checkpoints