#!/bin/bash
# Launches the Rat Gait Analysis web interface.
# Double-click this file in Finder, or run: ./run_app.sh
# Then open http://localhost:5050 in a browser.
cd "$(dirname "$0")"
./.venv/bin/python server.py
