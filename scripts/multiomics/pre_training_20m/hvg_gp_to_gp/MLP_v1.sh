#!/bin/bash

#SBATCH -J Multi_Pretrain_20M_GP_to_GP
#SBATCH -p gpu_p
#SBATCH --qos gpu_long
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=28
#SBATCH -t 2-00:00:00
#SBATCH --mem=240GB
#SBATCH --nice=10000


if [ -n "$1" ]; then
    source "$1"
else
    source "$HOME/.profile_prak7"
fi
conda activate prak7




cd $SSL_PROJECT_HOME/self_supervision/trainer/multiomics
cd /home/icb/till.richter/git/self_supervision/self_supervision/trainer/multiomics
conda activate celldreamer
python train.py --masking_strategy 'gp_to_gp' --dataset "20M" --version 'gp2gp_Pretrain_20M_v1' --model 'MAE' --mode 'pre_training' --dropout 0.05 --weight_decay 0.0 --learning_rate 0.001  --batch_size 8192
