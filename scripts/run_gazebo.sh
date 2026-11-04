#!/usr/bin/env bash
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export GZ_SIM_SYSTEM_PLUGIN_PATH="/home/nikita/drone/ardupilot_gazebo/build:${GZ_SIM_SYSTEM_PLUGIN_PATH}"
export GZ_SIM_RESOURCE_PATH="${PROJECT_DIR}/models:/home/nikita/drone/ardupilot_gazebo/models:/home/nikita/drone/ardupilot_gazebo/worlds:${GZ_SIM_RESOURCE_PATH}"
exec gz sim -v4 -r "${PROJECT_DIR}/worlds/quarry.sdf"
