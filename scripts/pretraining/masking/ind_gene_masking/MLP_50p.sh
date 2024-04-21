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
SSL_PROJECT_HOME=/home/v-mahughes/sc-SFM/ssl_in_scg
cd $SSL_PROJECT_HOME/self_supervision/trainer/masking

torchrun --nproc-per-node 4 train.py --mask_rate 0.5 --model 'MLP' --dropout 0.1 --weight_decay 0.01 --lr 0.001 --data_path /home/v-mahughes/data/data_test --model_path /home/v-mahughes/sc-SFM-saved-models/ssl/tests --wandb_job_name 'test' --version 0 --frac_seed_label 'idx0_seed0' --decoder 