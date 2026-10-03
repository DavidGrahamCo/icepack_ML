#!/bin/bash -l
#
# Apply every .mods file in a passed in directory to the central Arctic
# ensemble, run it, and move results to scratch.
#
#
#SBATCH --job-name=ca_sweep_icepack
#SBATCH --qos=blanca-appm
#SBATCH --partition=blanca-appm
#SBATCH --account=blanca-appm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --output=slurm-logs/%x_%j.log
#SBATCH --mail-user=dagr5998@colorado.edu
#SBATCH --mail-type=ALL

module purge
module load anaconda
conda activate icepack
which conda
#conda install -n icepack -c conda-forge xarray netcdf4 --solver=libmamba
#This requires the case directories to exist already, use setupensemble.sh

FILENAME="${1:?usage: $0 <parameter-directory>}"
PARAMDIR=$(readlink -f "$FILENAME")
SCRATCH="/scratch/alpine/dagr5998/$(basename "$FILENAME")"

mods_files=( "$PARAMDIR"/*.mods ) #input is relative to icepack-dirs

cd Icepack


for MODS in "${mods_files[@]}"; do
  tag=$(basename "$MODS" .mods)
  

  for i in $(seq 1 5); do
    n=$(printf '%04d' "$i")
    for yr in $(seq 2000 2013); do
      case="centralarctic_forced_${n}_${yr}-$((yr+2))"
      (
        cd "$case" || exit 1
        cp icepack_in_clean icepack_in
        ./casescripts/parse_namelist.sh icepack_in "$MODS"
        ./icepack.submit

        rundir="$(pwd)/../../runs/${case}"
        out="${SCRATCH}/${tag}/${case}"

        if ! compgen -G "$rundir/history/*.nc" > /dev/null; then
          echo "FAILED: ${tag} ${case}" >&2
          exit 1
        fi

        mkdir -p "$out"
        mv "$rundir/history" "$out/history" && mkdir -p "$rundir/history"
      )
    done
  done
  mkdir -p "${SCRATCH}/${tag}"
  cp -p "$MODS" "${SCRATCH}/${tag}/"
done