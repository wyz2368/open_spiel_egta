#!/bin/bash

#SBATCH --job-name=egta_kuhn_poker
#SBATCH --mail-user=qmaai@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=40g
#SBATCH --time=02-00:00:00
#SBATCH --account=wellman1
#SBATCH --partition=largemem
#SBATCH --output=/home/qmaai/se/open_spiel/open_spiel/python/algorithms/psro_variations/root_result/slurmout.log

module load python3.6-anaconda/5.2.0
cd ${SLURM_SUBMIT_DIR}
python rl_psro_example.py
