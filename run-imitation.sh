#!/usr/bin/env bash
set -euo pipefail

# Resolve the directory this script lives in, so it works from anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR=".venv"

# Create the virtual environment if it does not exist yet.
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
fi

# Activate the virtual environment.
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Install/update dependencies.
echo "Installing requirements ..."
pip install -r requirements.txt

# Start the application, forwarding any arguments passed to this script.
echo "Starting imitation.py ..."
exec python imitation.py "$@"
