#!/usr/bin/env bash
set -e
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/jazzy/setup.bash

if [ ! -f "/home/nikita/drone/ardupilot_gazebo/build/libArduPilotPlugin.so" ]; then
    echo "Сборка ArduPilotPlugin для Gazebo Harmonic..."
    mkdir -p /home/nikita/drone/ardupilot_gazebo/build
    cd /home/nikita/drone/ardupilot_gazebo/build
    cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
    make -j"$(nproc)"
fi

export GZ_SIM_SYSTEM_PLUGIN_PATH="/home/nikita/drone/ardupilot_gazebo/build:${GZ_SIM_SYSTEM_PLUGIN_PATH}"
export GZ_SIM_RESOURCE_PATH="${PROJECT_DIR}/models:/home/nikita/drone/ardupilot_gazebo/models:/home/nikita/drone/ardupilot_gazebo/worlds:${GZ_SIM_RESOURCE_PATH}"

# --- Зачистка хвостов предыдущих запусков (защита от "связи нет") ---
echo "[0/3] Очистка процессов предыдущего запуска..."
pkill -f sim_vehicle.py 2>/dev/null || true
pkill -f arducopter 2>/dev/null || true
pkill -f mavproxy 2>/dev/null || true
pkill -f "gz sim" 2>/dev/null || true
pkill -f "sleep infinity" 2>/dev/null || true
sleep 3
if ss -tulpn 2>/dev/null | grep -qE ':5760|:9002 '; then
    echo "ВНИМАНИЕ: порты 5760/9002 ещё заняты, добейте вручную: ss -tulpn | grep -E '5760|9002'" >&2
fi

mkdir -p "${PROJECT_DIR}/logs"

# Чистим логи, чтобы не путаться в старых записях
: > "${PROJECT_DIR}/logs/gazebo.log"
: > "${PROJECT_DIR}/logs/sitl.log"
: > "${PROJECT_DIR}/logs/ros_gz_bridge.log"

echo "[1/3] Запуск Gazebo Harmonic с миром виртуального карьера..."
if [ "${HEADLESS:-0}" = "1" ]; then
    gz sim -s -r "${PROJECT_DIR}/worlds/quarry.sdf" > "${PROJECT_DIR}/logs/gazebo.log" 2>&1 &
else
    gz sim -r --render-engine ogre "${PROJECT_DIR}/worlds/quarry.sdf" > "${PROJECT_DIR}/logs/gazebo.log" 2>&1 &
fi
GZ_PID=$!
sleep 4

echo "[2/3] Запуск ArduPilot SITL (ArduCopter, модель gazebo-iris)..."
# Заглушка stdin: у фоновых команд неинтерактивного шелла stdin == /dev/null,
# MAVProxy читает EOF и мгновенно выходит, убивая весь SITL.
# Держим для SITL FIFO, который никогда не закрывается.
FIFO="${PROJECT_DIR}/logs/sitl_stdin.fifo"
[ -p "$FIFO" ] || mkfifo "$FIFO"
sleep infinity > "$FIFO" &
STDIN_KEEPER_PID=$!
/home/nikita/ardupilot/Tools/autotest/sim_vehicle.py -v ArduCopter -f gazebo-iris --model JSON --no-rebuild --map --out 127.0.0.1:14551 < "$FIFO" > "${PROJECT_DIR}/logs/sitl.log" 2>&1 &
SITL_PID=$!
sleep 6

if ! kill -0 $SITL_PID 2>/dev/null; then
    echo "ОШИБКА: SITL не запустился, последние строки logs/sitl.log:" >&2
    tail -n 20 "${PROJECT_DIR}/logs/sitl.log" >&2
    kill $GZ_PID $STDIN_KEEPER_PID 2>/dev/null || true
    exit 1
fi

echo "[3/3] Запуск моста ROS 2 <-> Gazebo (ros_gz_bridge)..."
ros2 run ros_gz_bridge parameter_bridge --ros-args -p config_file:="${PROJECT_DIR}/config/bridge_config.yaml" > "${PROJECT_DIR}/logs/ros_gz_bridge.log" 2>&1 &
BRIDGE_PID=$!

echo "${GZ_PID} ${SITL_PID} ${BRIDGE_PID} ${STDIN_KEEPER_PID}" > "${PROJECT_DIR}/logs/running.pids"
echo "Комплекс запущен! Для управления откройте терминал оператора:"
echo "  python3 ${PROJECT_DIR}/ros2_ws/src/quarry_drone_vlm/quarry_drone_vlm/operator_cli.py"