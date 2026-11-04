# -*- coding: utf-8 -*-
"""
Уровень 3: Исполнительный модуль управления квадрокоптером в ArduPilot SITL через MAVLink (pymavlink).
"""

import math
import time
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from pymavlink import mavutil
    HAS_PYMAVLINK = True
except ImportError:
    mavutil = None
    HAS_PYMAVLINK = False


ALLOWED_SITL_PREFIXES = (
    "udp:127.0.0.1:",
    "udp:localhost:",
    "tcp:127.0.0.1:",
    "tcp:localhost:",
    "udpin:127.0.0.1:",
    "udpin:0.0.0.0:",
    "mock://",
)


@dataclass
class TelemetryState:
    connected: bool = False
    mode: str = "UNKNOWN"
    armed: bool = False
    ekf_ok: bool = False
    home_available: bool = False
    x: float = 0.0          # Север (м, локальная система карьера)
    y: float = 0.0          # Запад (м, локальная система карьера)
    z: float = 0.0          # Высота вверх (м)
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    yaw_deg: float = 0.0
    battery_v: float = 16.8
    last_heartbeat_ts: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ArduPilotSITLController:
    """Контроллер управления ArduPilot SITL с поддержкой реального MAVLink и тестового Mock-режима."""

    def __init__(
        self,
        connection_string: str = "udp:127.0.0.1:14550",
        heartbeat_timeout_sec: float = 3.0,
        acceptance_radius_m: float = 1.2,
    ):
        self._verify_sitl_only_connection(connection_string)
        self.connection_string = connection_string
        self.heartbeat_timeout_sec = heartbeat_timeout_sec
        self.acceptance_radius_m = acceptance_radius_m
        self.is_mock = connection_string.startswith("mock://")
        self.master = None
        self.state = TelemetryState()
        self._abort_requested = False

    @staticmethod
    def _verify_sitl_only_connection(conn_str: str) -> None:
        clean = conn_str.strip().lower()
        if "/dev/tty" in clean or "com" in clean[:4]:
            raise PermissionError(
                f"БЕЗОПАСНОСТЬ: Подключение к физическому последовательному порту ({conn_str}) строго запрещено! "
                "Разрешён только локальный ArduPilot SITL."
            )
        if not any(clean.startswith(prefix) for prefix in ALLOWED_SITL_PREFIXES):
            raise PermissionError(
                f"БЕЗОПАСНОСТЬ: Недопустимый адрес подключения ({conn_str}). "
                f"Разрешены только локальные SITL-адреса: {ALLOWED_SITL_PREFIXES}"
            )

    def connect(self, timeout_sec: float = 15.0) -> bool:
        if self.is_mock:
            self.state.connected = True
            self.state.mode = "STABILIZE"
            self.state.armed = False
            self.state.ekf_ok = True
            self.state.home_available = True
            self.state.last_heartbeat_ts = time.monotonic()
            return True

        if not HAS_PYMAVLINK:
            raise RuntimeError("Библиотека pymavlink не установлена. Установите: pip install pymavlink")

        self.master = mavutil.mavlink_connection(self.connection_string, source_system=255)
        hb = self.master.wait_heartbeat(timeout=timeout_sec)
        if hb is None:
            self.state.connected = False
            return False

        self.state.connected = True
        self.state.last_heartbeat_ts = time.monotonic()
        self.state.ekf_ok = True
        self.state.home_available = True

        self.master.mav.request_data_stream_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL,
            10,
            1,
        )
        self.poll_telemetry()
        return True

    def poll_telemetry(self) -> TelemetryState:
        if self.is_mock:
            self.state.last_heartbeat_ts = time.monotonic()
            return self.state

        if self.master is None:
            self.state.connected = False
            return self.state

        while True:
            msg = self.master.recv_match(blocking=False)
            if msg is None:
                break
            mtype = msg.get_type()
            if mtype == "HEARTBEAT":
                self.state.last_heartbeat_ts = time.monotonic()
                self.state.connected = True
                self.state.mode = mavutil.mode_string_v10(msg)
                self.state.armed = bool(
                    msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                )
            elif mtype == "LOCAL_POSITION_NED":
                self.state.x = float(msg.x)
                self.state.y = -float(msg.y)
                self.state.z = -float(msg.z)
                self.state.vx = float(msg.vx)
                self.state.vy = -float(msg.vy)
                self.state.vz = -float(msg.vz)
            elif mtype == "ATTITUDE":
                self.state.yaw_deg = math.degrees(float(msg.yaw)) % 360.0
            elif mtype == "SYS_STATUS":
                if msg.voltage_battery > 0:
                    self.state.battery_v = msg.voltage_battery / 1000.0
            elif mtype == "EKF_STATUS_REPORT":
                self.state.ekf_ok = bool(msg.flags & 1)

        if (time.monotonic() - self.state.last_heartbeat_ts) > self.heartbeat_timeout_sec:
            self.state.connected = False

        return self.state

    def request_abort(self) -> None:
        self._abort_requested = True
        if self.is_mock:
            self.state.mode = "BRAKE"
            self.state.vx = 0.0
            self.state.vy = 0.0
            self.state.vz = 0.0
            return
        if self.master is not None:
            mode_mapping = self.master.mode_mapping()
            brake_id = mode_mapping.get("BRAKE", mode_mapping.get("LOITER", 5))
            self.master.set_mode(brake_id)

    def clear_abort(self) -> None:
        self._abort_requested = False

    def set_mode(self, mode_name: str = "GUIDED", timeout_sec: float = 8.0) -> bool:
        self.poll_telemetry()
        if not self.state.connected:
            return False

        if self.is_mock:
            self.state.mode = mode_name.upper()
            return True

        mode_mapping = self.master.mode_mapping()
        if mode_name not in mode_mapping:
            return False
        mode_id = mode_mapping[mode_name]
        t0 = time.monotonic()
        while (time.monotonic() - t0) < timeout_sec:
            self.master.set_mode(mode_id)
            time.sleep(0.3)
            self.poll_telemetry()
            if self.state.mode == mode_name:
                return True
        return False

    def arm_and_takeoff(self, target_alt_m: float = 5.0, timeout_sec: float = 90.0) -> Tuple[bool, str]:
        self.clear_abort()
        self.poll_telemetry()
        if not self.state.connected:
            return False, "Потеряна связь с ArduPilot SITL перед взлётом."

        if not self.set_mode("GUIDED"):
            return False, "Не удалось переключить автопилот в режим GUIDED."

        if self.is_mock:
            self.state.armed = True
            steps = 10
            start_z = self.state.z
            for i in range(1, steps + 1):
                if self._abort_requested:
                    return False, "Взлёт прерван командой оператора (ABORT)."
                self.state.z = start_z + (target_alt_m - start_z) * (i / steps)
                time.sleep(0.05)
            return True, f"Взлёт завершён. Текущая высота: {self.state.z:.2f} м."

        t0 = time.monotonic()
        while not self.state.armed and (time.monotonic() - t0) < 15.0:
            if self._abort_requested:
                return False, "Взлёт прерван во время постановки на охрану (ARM)."
            self.master.arducopter_arm()
            time.sleep(0.5)
            self.poll_telemetry()

        if not self.state.armed:
            return False, "Тайм-аут постановки на охрану (ARM). Проверьте статус PreArm в SITL."

        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0,
            0, 0, 0, 0, 0, 0,
            float(target_alt_m),
        )

        t_takeoff = time.monotonic()
        while (time.monotonic() - t_takeoff) < timeout_sec:
            if self._abort_requested:
                self.request_abort()
                return False, "Взлёт аварийно остановлен оператором."
            self.poll_telemetry()
            if not self.state.connected:
                return False, "Потеря связи с автопилотом во время набора высоты."
            if abs(self.state.z - target_alt_m) <= 0.6:
                return True, f"Взлёт выполнен успешно. Высота: {self.state.z:.2f} м."
            time.sleep(0.2)

        return False, f"Тайм-аут набора высоты {target_alt_m}м (достигнута высота {self.state.z:.2f}м)."

    def rotate_yaw(self, yaw_deg: float, relative: bool = False, speed_deg_s: float = 30.0) -> Tuple[bool, str]:
        """Поворачивает квадрокоптер по курсу (рысканью / yaw) через MAV_CMD_CONDITION_YAW."""
        self.poll_telemetry()
        if not self.state.connected:
            return False, "Нет связи с автопилотом для выполнения поворота."

        if self.is_mock:
            if relative:
                self.state.yaw_deg = (self.state.yaw_deg + yaw_deg) % 360.0
            else:
                self.state.yaw_deg = yaw_deg % 360.0
            time.sleep(0.1)
            return True, f"Поворот выполнен. Текущий курс: {self.state.yaw_deg:.1f}°."

        is_rel = 1 if relative else 0
        direction = 1 if yaw_deg >= 0 else -1
        target_angle = abs(float(yaw_deg)) if relative else (float(yaw_deg) % 360.0)

        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_CONDITION_YAW,
            0,
            target_angle,
            float(speed_deg_s),
            direction,
            is_rel,
            0, 0, 0,
        )

        wait_sec = min(10.0, max(2.0, (target_angle if relative else 90.0) / max(10.0, speed_deg_s) + 1.0))
        t0 = time.monotonic()
        while (time.monotonic() - t0) < wait_sec:
            if self._abort_requested:
                self.request_abort()
                return False, "Поворот прерван оператором."
            self.poll_telemetry()
            time.sleep(0.2)

        return True, f"Поворот по yaw выполнен (целевой угол {yaw_deg}°, текущий курс {self.state.yaw_deg:.1f}°)."

    def goto_local_xyz(
        self,
        x_north: float,
        y_west: float,
        z_up: float,
        timeout_sec: float = 45.0,
        progress_cb: Optional[Callable[[TelemetryState], None]] = None,
    ) -> Tuple[bool, str]:
        self.poll_telemetry()
        if not self.state.connected:
            return False, "Нет связи с автопилотом для выполнения перемещения."

        if self.is_mock:
            start_x, start_y, start_z = self.state.x, self.state.y, self.state.z
            steps = 12
            for i in range(1, steps + 1):
                if self._abort_requested:
                    return False, "Перемещение прервано командой ABORT."
                alpha = i / steps
                self.state.x = start_x + (x_north - start_x) * alpha
                self.state.y = start_y + (y_west - start_y) * alpha
                self.state.z = start_z + (z_up - start_z) * alpha
                if progress_cb:
                    progress_cb(self.state)
                time.sleep(0.04)
            return True, f"Достигнута точка ({self.state.x:.1f}, {self.state.y:.1f}, {self.state.z:.1f}) м."

        north = float(x_north)
        east = -float(y_west)
        down = -float(z_up)

        type_mask = 0b0000111111111000
        t0 = time.monotonic()
        while (time.monotonic() - t0) < timeout_sec:
            if self._abort_requested:
                self.request_abort()
                return False, "Полёт к путевой точке прерван оператором."

            self.master.mav.set_position_target_local_ned_send(
                0,
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                type_mask,
                north,
                east,
                down,
                0, 0, 0,
                0, 0, 0,
                0, 0,
            )
            time.sleep(0.25)
            self.poll_telemetry()
            if progress_cb:
                progress_cb(self.state)

            if not self.state.connected:
                return False, "Потеря связи с SITL при движении к путевой точке."

            dist = math.sqrt(
                (self.state.x - x_north) ** 2
                + (self.state.y - y_west) ** 2
                + (self.state.z - z_up) ** 2
            )
            if dist <= self.acceptance_radius_m:
                return True, f"Точка ({x_north:.1f}, {y_west:.1f}, {z_up:.1f}) достигнута (ошибка {dist:.2f}м)."

        return False, f"Тайм-аут достижения точки ({x_north:.1f}, {y_west:.1f}, {z_up:.1f})."

    def safe_return_home(self, waypoints: List[Tuple[float, float, float]]) -> Tuple[bool, str]:
        self.poll_telemetry()
        if not self.state.connected:
            return False, "ОШИБКА БЕЗОПАСНОСТИ: Потеряна связь с автопилотом, автоматический возврат невозможен."

        if not self.state.ekf_ok or not self.state.home_available:
            self.set_mode("LAND")
            return (
                False,
                "ОШИБКА БЕЗОПАСНОСТИ: EKF нестабилен или домашняя позиция недоступна. "
                "Автоматический возврат отменён, активирована безопасная посадка на месте (LAND).",
            )

        for wp in waypoints:
            ok, msg = self.goto_local_xyz(wp[0], wp[1], wp[2])
            if not ok:
                return False, f"Ошибка при возврате домой: {msg}"

        return True, "Квадрокоптер успешно вернулся в домашнюю точку."

    def land(self, timeout_sec: float = 30.0) -> Tuple[bool, str]:
        self.poll_telemetry()
        if not self.state.connected:
            return False, "Нет связи с автопилотом для посадки."

        if self.is_mock:
            self.state.mode = "LAND"
            self.state.z = 0.0
            self.state.armed = False
            return True, "Посадка завершена, двигатели дизармированы."

        if not self.set_mode("LAND"):
            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_CMD_NAV_LAND,
                0,
                0, 0, 0, 0, 0, 0, 0,
            )

        t0 = time.monotonic()
        while (time.monotonic() - t0) < timeout_sec:
            self.poll_telemetry()
            if not self.state.armed or self.state.z <= 0.25:
                return True, "Посадка успешно выполнена."
            time.sleep(0.3)

        return False, "Тайм-аут ожидания завершения посадки."