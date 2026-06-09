#!/bin/bash
set -e

echo "=== Running init_db.py — creating table schema + seed data ==="
python /app/init_db.py

echo "=== Starting uvicorn ==="
exec "$@"
