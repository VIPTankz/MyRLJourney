#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --mem=60000
#SBATCH --partition=ecsstaff
#SBATCH --account=ecsstaff
#SBATCH --job-name=RL_atari_benchmark3
#SBATCH --time=10:00:00

module load conda/py3-latest
source activate atari_work

cd /home/$USER/MyRLJourney
pip install -e .

staskfarm farm_atari_cpu3.txt