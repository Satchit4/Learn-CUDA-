#!/bin/bash
# Helper script to run CUDA programs with ZLUDA
# Usage: ./run_with_zluda.sh ./your_cuda_program [args]

# Get the absolute path of the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ZLUDA_DIR="$SCRIPT_DIR/zluda"

# Ensure the local libamdhip64.so.6 symlink is found
# We add SCRIPT_DIR to LD_LIBRARY_PATH because that's where we created the symlink
export LD_LIBRARY_PATH="$SCRIPT_DIR:$ZLUDA_DIR:$LD_LIBRARY_PATH"

# Run the command
exec "$@"
