#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=64000
#SBATCH -p batch
#SBATCH --job-name=RL_atari_benchmark
#SBATCH --time=48:00:00

module load conda/py3-latest
source activate atari_work

cd /home/$USER/MyRLJourney
pip install -e .

staskfarm farm_atari.txt