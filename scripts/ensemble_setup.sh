#!/usr/bin/env bash
set -euo pipefail
 
for loc in centralarctic barents sibchuk coastalcanada; do
    ./icepack.setup --case "${loc}" --mach conda --env linux -s "ionetcdf,djdefaultsettings,MDF${loc},djbuildsettings"
    ( cd "${loc}" && ./icepack.build && cp icepack_in icepack_in_clean )
    echo "built ${loc}"
done
 
echo "Done with setup"
