#!/bin/bash -l
#
# Build every central arctic ensemble member's case directories
#
#Before:
#module load slurm/blanca
#mkdir -p slurm-logs
#sbatch setup_ensembles.sbatch
#
#SBATCH --job-name=icepack_setup
#SBATCH --qos=blanca-appm
#SBATCH --partition=blanca-appm
#SBATCH --account=blanca-appm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --export=NONE
#SBATCH --output=slurm-logs/%x_%j.log
#SBATCH --mail-user=dagr5998@colorado.edu
#SBATCH --mail-type=ALL

module purge
module load anaconda
conda activate icepack

cd Icepack
echo "job       : $SLURM_JOB_ID"
echo "directory : $PWD"
echo "started   : $(date)"

./ensemble_setup.sh

echo "finished  : $(date)"
