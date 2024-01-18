#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -c12
#SBATCH -p gpu
#SBATCH -C gmem24
#SBATCH --job-name=disco-actions
#SBATCH --output=./logs/%x.out

# Load modules
module load anaconda3
module list                            # Have Nvidia tell us the GPU/CPU mapping so we know
nvidia-smi topo -m
#source activate /home/siddiqui/.conda/envs/mdm

python -m train.train_mdm --save_dir save/disco-actions --dataset ntu60 --cond_mask_prob 0 --lambda_rcxyz 1 --lambda_vel 1 --lambda_fc 1 --overwrite

#python -m sample.generate --model_path "/home/siddiqui/motion-diffusion-model/save/ntu60_train_xyz_translation/model000440000.pt"  --action_name "sit down"

