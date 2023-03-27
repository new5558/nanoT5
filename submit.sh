#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1 -c 16
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH -t 80:00:00
#SBATCH -J electra-training-test

module purge
module load Miniconda3/22.11.1-1
conda deactivate
conda activate /home/superai008/.conda/envs/nanoT5

# nvidia-smi

export HF_DATASETS_CACHE="/project/lt900001-ai23ta/new/nanoT5/.cache"
# export HF_DATASETS_OFFLINE="1"

# python scripts/cli.py train-tokenizer --vocab-size=50000 --num-docs=500000 --slurm
# python scripts/cli.py convert-tokenizer
# python scripts/cli.py pretrain --input-path="./data/preprocesses_data_128"
# python scripts/cli.py pretrain --num-gpus=1 --max-steps=2000 --warmup-steps=500

source /project/lt900038-ai23tn/useproxy

# python -m nanoT5.main
sleep 24000
# python -m nanoT5.main
# python check_t5_thai.py
