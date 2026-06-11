#!/bin/bash
set -e

# pin bcrypt for passlib 1.7.4 compatibility (bcrypt>=4.1.0 breaks passlib)
pip install -q bcrypt==4.0.1

echo "=== Starting uvicorn ==="
exec "$@"
