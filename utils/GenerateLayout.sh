#!/bin/bash
#SBATCH -A research
#SBATCH -c 40
#SBATCH -w gnode40
#SBATCH --gres=gpu:4
#SBATCH --time=4-00:00:00
#SBATCH --job-name=layout_Racklay
#SBATCH -o layout_Racklay.logs

source ~/.bashrc
cd /scratch/
conda activate monolayout

cp ~/finalData.zip ./data/
git clone https://github.com/AnuragSahu/WareSynth_PostIROS.git

python WareSynth_PostIROS/scripts/ExendedShelfCentricLayouts/GenerateShelfCentricLayouts.py

zip -r Data.zip data/
cp Data.zip  ~/