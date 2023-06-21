#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH -p batch
#SBATCH --job-name=RL_atari_benchmark
#SBATCH --time=48:00:00

module load conda/py3-latest
source activate atari_work

cd /home/$USER/tjc2g19
pip install -e .

staskfarm MyRLJourney/farm_atari.txt