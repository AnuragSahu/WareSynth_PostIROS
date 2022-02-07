#!/bin/bash
#SBATCH -A research
#SBATCH -c 40
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00
#SBATCH --job-name=layout_Racklay
#SBATCH -o layout_Racklay.logs

source ~/.bashrc
cd /scratch/
conda activate monolayout

rm -rf afinalData
rm -rf WareSynth_PostIROS

cp ~/Racklay_10k_far_box_lay.zip ./
unzip ./Racklay_10k_far_box_lay.zip

git clone https://github.com/AnuragSahu/WareSynth_PostIROS.git
cd WareSynth_PostIROS/
git checkout PostICRA
cd ../

python WareSynth_PostIROS/scripts/ExendedShelfCentricLayouts/GenerateShelfCentricLayouts.py

zip -r Racklay_10k_far_box_lay.zip afinalData/
cp Racklay_10k_far_box_lay.zip  ~/
