#!/usr/bin/env bash
set -e
echo "=== ДИАГНОСТИКА ОКРУЖЕНИЯ DRONE_VLM_QUARRY ==="
lsb_release -d || true
source /opt/ros/jazzy/setup.bash
echo "ROS_DISTRO: $ROS_DISTRO"
gz sim --version | head -n 2
echo "Проверка плагина ArduPilotPlugin:"
ls -l /home/nikita/drone/ardupilot_gazebo/build/libArduPilotPlugin.so || echo "ВНИМАНИЕ: Соберите плагин в /home/nikita/drone/ardupilot_gazebo/build"
echo "Проверка пакетов ros_gz:"
ros2 pkg list | grep ros_gz
echo "Проверка Ollama и модели qwen2.5vl:7b:"
ollama list | grep qwen2.5vl:7b
echo "=== ДИАГНОСТИКА ЗАВЕРШЕНА ==="
