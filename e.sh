#!/bin/bash
#SBATCH --account marasovic-gpu-np
#SBATCH --partition marasovic-gpu-np
#SBATCH --qos=marasovic-gpulong-np
#SBATCH --ntasks=32
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=36:00:00
#SBATCH --mem=80GB
#SBATCH --mail-user=jacob.k.johnson@utah.edu
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH -o filename-%j

cd /uufs/chpc.utah.edu/common/home/u0403624/scratch/ScoNe/ScoNe

source /scratch/general/vast/u0403624/miniconda3/etc/profile.d/conda.sh
conda activate 38b

python e.py > results
