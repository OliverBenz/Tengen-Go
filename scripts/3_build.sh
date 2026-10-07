#!/bin/bash

cd "$(dirname "$0")"
. ./0_settings.sh

preset="Linux64-$configuration"

echo "Entering Development Environment"
echo "Using preset: $preset"
nix develop .. --command bash -c "
  cd .. &&
  cmake --preset $preset -DTENGEN_BUILD_TESTS=$buildTests &&
  cmake --build --preset $preset --parallel
"
