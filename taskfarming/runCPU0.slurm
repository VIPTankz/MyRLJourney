#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --mem=60000
#SBATCH --partition=ecsstaff
#SBATCH --account=ecsstaff
#SBATCH --job-name=RL_atari_benchmark0
#SBATCH --time=10:00:00
#SBATCH --mail-user=tjc2g19@soton.ac.uk
#SBATCH --mail-type=ALL

module load conda/py3-latest
source activate atari_work

cd /home/$USER/MyRLJourney/taskfarming
pip install -e .

staskfarm farm_atari_cpu0.txt