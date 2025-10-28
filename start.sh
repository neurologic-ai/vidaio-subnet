#!/bin/bash
set -e

# Start uvicorn in background
uvicorn app:app --host 127.0.0.1 --port 8000 &
UVICORN_PID=$!

# Start nginx in foreground
nginx -g "daemon off;" &
NGINX_PID=$!

# Function to handle shutdown
cleanup() {
    echo "Shutting down..."
    kill $UVICORN_PID $NGINX_PID 2>/dev/null || true
    wait $UVICORN_PID $NGINX_PID 2>/dev/null || true
    exit 0
}

# Trap signals
trap cleanup SIGTERM SIGINT

# Wait for both processes
wait $UVICORN_PID $NGINX_PID

