#!/bin/bash

# Set the path to the script
SCRIPT_PATH="./sijapi/helpers/start.py"

# Check if the script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Script not found at $SCRIPT_PATH"
    exit 1
fi

# Set up the environment
source "./sijapi/config/.env"

# Run the Python script
python "$SCRIPT_PATH"