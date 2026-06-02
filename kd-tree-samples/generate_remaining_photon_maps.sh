#!/bin/bash

SCENES="sponza water-caustic"
COUNTS="1000000 5000000 10000000"

for SCENE in $SCENES; do
  for NORMAL in $COUNTS; do
    for CAUSTIC in $COUNTS; do
      echo "=== $SCENE : normal=$NORMAL caustic=$CAUSTIC ==="
      ./photonEmitter.exe "$SCENE" "$NORMAL" "$CAUSTIC" || echo "FAILED: $SCENE $NORMAL $CAUSTIC"
    done
  done
done

echo "Remaining photon map generation done!"
