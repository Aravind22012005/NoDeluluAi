#!/bin/bash
# NoDelulu AI — Startup Script
echo "==============================="
echo "  NoDelulu AI — Stay Grounded."
echo "==============================="

# Install dependencies
echo ""
echo "[1/3] Installing dependencies..."
pip install -r requirements.txt --break-system-packages -q

# Start Flask
echo ""
echo "[2/3] Starting Flask API on http://localhost:5000 ..."
echo "[3/3] Open index.html in your browser (or visit http://localhost:5000)"
echo ""
echo "Press Ctrl+C to stop."
echo ""
python app.py
