#!/bin/bash

#SBATCH -J hvg_ae_ssl_50p
#SBATCH -p gpu_p
#SBATCH --qos gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=6
#SBATCH -t 2-00:00:00
#SBATCH --mem=90GB
#SBATCH --nice=10000


# if [ -n "$1" ]; then
#     source "$1"
# else
#     source "$HOME/.bashrc"
# fi

DATA_PATH=$1
MODEL_PATH=$2
SUBSAMPLE=$3
CHECKPOINT_INT=$4


python -u self_supervision/trainer/masking/train.py --mask_rate 0.5 --model 'MLP' --dropout 0.1 --weight_decay 0.01 --lr 0.001 --data_path $DATA_PATH --model_path $MODEL_PATH --subsample_frac $SUBSAMPLE --checkpoint_interval $CHECKPOINT_INT --decoder