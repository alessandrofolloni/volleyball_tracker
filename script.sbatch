#!/bin/bash
#SBATCH --job-name=prova
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alessandro.folloni2@studio.unibo.it
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --output=/public.hpc/alessandro.folloni2/logs/volley/%x.o
#SBATCH --error=/public.hpc/alessandro.folloni2/logs/volley/%x.e
#SBATCH --gres=gpu:1

. /public.hpc/alessandro.folloni2/volleyball_tracker/env/bin/activate
cd /public.hpc/alessandro.folloni2/volleyball_tracker || exit
python3 train.py
