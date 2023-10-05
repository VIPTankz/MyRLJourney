#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --mem=300000
#SBATCH --partition=ecsstaff
#SBATCH --account=ecsstaff
#SBATCH --job-name=RL_atari_benchmark0
#SBATCH --time=20:00:00
#SBATCH --mail-user=tjc2g19@soton.ac.uk
#SBATCH --mail-type=ALL

module load conda/py3-latest
source activate atari_work

cd /home/$USER/MyRLJourney
pip install -e .

staskfarm farm_atari.txt