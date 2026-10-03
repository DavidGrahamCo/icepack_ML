#!/bin/bash -l
#
# Run a parameter sweep, one array task per location in parallel
# Requires ensemble_setup.sh to have been run
#
#SBATCH --job-name=icepack_sweep
#SBATCH --qos=blanca-appm
#SBATCH --partition=blanca-appm
#SBATCH --account=blanca-appm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --export=NONE
#SBATCH --output=/projects/dagr5998/icepack_ML/slurm-logs/%x_%A_%a.log
#SBATCH --mail-user=dagr5998@colorado.edu
#SBATCH --mail-type=ALL

module purge
module load anaconda
conda activate icepack

LOCATIONS=(centralarctic barents sibchuk coastalcanada)   
LOC=${LOCATIONS[$SLURM_ARRAY_TASK_ID]}

echo "job       : ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "location  : ${LOC}"
echo "started   : $(date)"

/projects/dagr5998/icepack_ML/scripts/parameter_sweep.sh "$1" "$LOC"

echo "finished  : $(date)"
