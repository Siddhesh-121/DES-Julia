#!/bin/bash

# Simple script to run Julia with compiled system image
# Usage: ./run_compiled.sh script.jl [args...]

SYSIMAGE="DESLibrary_compiled.so"

if [ -f "$SYSIMAGE" ]; then
    echo "Running with compiled system image: $SYSIMAGE"
    julia --sysimage "$SYSIMAGE" "$@"
else
    echo "Compiled system image not found, running normally"
    echo "To compile: julia build_sysimage.jl"
    julia "$@"
fi