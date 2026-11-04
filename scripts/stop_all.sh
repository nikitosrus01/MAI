#!/usr/bin/env bash
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "Остановка всех процессов симуляционного комплекса..."
if [ -f "${PROJECT_DIR}/logs/running.pids" ]; then
    kill $(cat "${PROJECT_DIR}/logs/running.pids") 2>/dev/null || true
    rm -f "${PROJECT_DIR}/logs/running.pids"
fi
pkill -f "gz sim" 2>/dev/null || true
pkill -f "sim_vehicle.py" 2>/dev/null || true
pkill -f "arducopter" 2>/dev/null || true
pkill -f "parameter_bridge" 2>/dev/null || true
echo "Все процессы остановлены."
