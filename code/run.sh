#!/bin/bash 
#  main.py
#SBATCH --job-name=autodim_test 
#SBATCH --time=3:00:00
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1               # 1 computer nodes
#SBATCH --cpus-per-task=1       # 1 cpu per task
#SBATCH --mem=32GB              # 32GB mem on EACH NODE
#SBATCH --gres=gpu:1            # 1 gpu per nodes
##  #SBATCH --gpus-per-task=1
python3 ./main.py