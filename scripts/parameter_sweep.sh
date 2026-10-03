#!/usr/bin/env bash
# applies all of the mods in given directory to given location, saves results (first is parameter set, second is location)

PARAMSET="$1"
LOC="$2"

PARAMDIR="/projects/dagr5998/icepack_ML/parameters/${PARAMSET}"
SCRATCH="/scratch/alpine/dagr5998/${PARAMSET}"
rundir="/projects/dagr5998/icepack-dirs/runs/${LOC}"
mods_files=( "$PARAMDIR"/*.mods )

cd "/projects/dagr5998/icepack-dirs/Icepack/${LOC}" || exit 1

for MODS in "${mods_files[@]}"; do
  tag=$(basename "$MODS" .mods)
  out="${SCRATCH}/${LOC}_${tag}"

  cp icepack_in_clean icepack_in
  ./casescripts/parse_namelist.sh icepack_in "$MODS"
  ./icepack.submit

  if ! compgen -G "$rundir/history/*.nc" > /dev/null; then
    echo "FAILED: ${LOC} ${tag}" >&2
    continue
  fi

  mkdir -p "$out"
  mv "$rundir/history" "$out/history" && mkdir -p "$rundir/history"
  cp icepack_in "$out/"
done