#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import time
import zlib
import struct
import threading
import argparse

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from vlm.vlm_client import OllamaVLMClient
from planner.plan_validator import PlanValidator
from controller.ardupilot_controller import ArduPilotSITLController
from controller.mission_orchestrator import MissionOrchestrator, FramePacket

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False


def raw_rgb_to_png(raw_bytes: bytes, width: int, height: int, encoding: str = "rgb8") -> bytes:
    """Конвертирует сырой буфер sensor_msgs/msg/Image (RGB8/BGR8) в валидный PNG для Ollama VLM."""
    if raw_bytes.startswith(b"\x89PNG") or raw_bytes.startswith(b"\xff\xd8"):
        return raw_bytes

    expected_len = width * height * 3
    if len(raw_bytes) < expected_len or width <= 0 or height <= 0:
        return raw_bytes

    rows = []
    stride = width * 3
    is_bgr = "bgr" in encoding.lower()
    for y in range(height):
        row = raw_bytes[y * stride : (y + 1) * stride]
        if is_bgr:
            # Перестановка BGR -> RGB
            ba = bytearray(len(row))
            ba[0::3] = row[2::3]
            ba[1::3] = row[1::3]
            ba[2::3] = row[0::3]
            row = bytes(ba)
        rows.append(b"\x00" + row)

    raw_scanlines = b"".join(rows)

    def make_chunk(chunk_type: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    idat = zlib.compress(raw_scanlines, level=6)
    return (
        b"\x89PNG\r\n\x1a\n"
        + make_chunk(b"IHDR", ihdr)
        + make_chunk(b"IDAT", idat)
        + make_chunk(b"IEND", b"")
    )


class CameraListenerNode(Node):
    """Фоновый ROS 2 узел для приёма видеопотока с виртуальной камеры Gazebo."""

    def __init__(self):
        super().__init__("operator_cli_camera_listener")
        self.latest_frame = None
        self.sub1 = self.create_subscription(Image, "/quarry/camera/image_raw", self._cb, 10)
        self.sub2 = self.create_subscription(Image, "/camera/image_raw", self._cb, 10)

    def _cb(self, msg: Image):
        png_bytes = raw_rgb_to_png(bytes(msg.data), msg.width, msg.height, getattr(msg, "encoding", "rgb8"))
        self.latest_frame = FramePacket(
            image_bytes=png_bytes,
            timestamp=time.monotonic(),
            width=msg.width,
            height=msg.height,
        )


def main():
    parser = argparse.ArgumentParser(description="Терминал оператора БПЛА в виртуальном карьере")
    parser.add_argument("--sitl", default="udp:127.0.0.1:14550", help="Адрес SITL (или mock://sitl для теста)")
    parser.add_argument("--ollama", default="http://127.0.0.1:11434", help="Адрес сервера Ollama")
    parser.add_argument("--model", default="qwen2.5vl:7b", help="Имя VLM модели")
    parser.add_argument("--once", default=None, help="Выполнить одну команду и выйти")
    parser.add_argument("--allow-fallback", action="store_true", help="Резервный парсер при недоступности Ollama")
    args = parser.parse_args()

    cam_node = None
    if HAS_ROS2:
        try:
            rclpy.init()
            cam_node = CameraListenerNode()
            spin_thread = threading.Thread(target=rclpy.spin, args=(cam_node,), daemon=True)
            spin_thread.start()
        except Exception as exc:
            print(f"[ПРЕДУПРЕЖДЕНИЕ] Не удалось запустить ROS 2 слушатель камеры: {exc}")

    vlm = OllamaVLMClient(ollama_host=args.ollama, model_name=args.model, timeout_sec=180.0)
    validator = PlanValidator(os.path.join(ROOT_DIR, "config"))
    controller = ArduPilotSITLController(connection_string=args.sitl)

    print("=== ТЕРМИНАЛ ОПЕРАТОРА БПЛА (ВИРТУАЛЬНЫЙ КАРЬЕР) ===")
    print(f"Подключение к автопилоту: {args.sitl} ...")
    if not controller.connect(timeout_sec=10.0):
        print("[ОШИБКА] Не удалось подключиться к ArduPilot SITL.")
        sys.exit(1)

    def get_frame():
        if cam_node is not None and cam_node.latest_frame is not None:
            return cam_node.latest_frame
        return None

    orchestrator = MissionOrchestrator(
        vlm_client=vlm,
        validator=validator,
        controller=controller,
        frame_provider=get_frame,
        status_callback=lambda s: print(s),
        allow_vlm_fallback=args.allow_fallback,
    )

    if args.once:
        rep = orchestrator.execute_russian_command(args.once)
        print("\n=== ИТОГОВЫЙ ОТЧЁТ МИССИИ ===")
        print(f"Статус: {'УСПЕШНО' if rep.success else 'ОШИБКА / ПРЕРВАНО'}")
        print(f"Выполненные шаги: {rep.executed_steps}")
        if rep.abort_reason:
            print(f"Причина остановки: {rep.abort_reason}")
        sys.exit(0 if rep.success else 2)

    print("Специальные команды: 'стоп' (аварийная остановка), 'домой' (возврат), 'статус', 'выход'.")
    while True:
        try:
            cmd = input("\nОператор > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nЗавершение работы оператора.")
            break
        if not cmd:
            continue
        if cmd.lower() in ("выход", "exit", "quit"):
            break
        if cmd.lower() in ("стоп", "abort"):
            controller.request_abort()
            print("[АВАРИЙНЫЙ СТОП] Команда торможения отправлена в ArduPilot SITL.")
            continue
        if cmd.lower() in ("домой", "rtl"):
            cmd = "Вернись в исходную точку и сядь"
        if cmd.lower() == "статус":
            st = controller.poll_telemetry()
            cam_ok = "ОК" if (cam_node and cam_node.latest_frame) else "НЕТ КАДРА"
            print(
                f"Телеметрия: режим={st.mode}, armed={st.armed}, "
                f"X={st.x:.1f}м, Y={st.y:.1f}м, Z={st.z:.1f}м, Yaw={st.yaw_deg:.1f}°, Камера={cam_ok}"
            )
            continue

        rep = orchestrator.execute_russian_command(cmd)
        print(f"--- Итог: {'УСПЕХ' if rep.success else 'ОШИБКА'} (время {rep.total_duration_sec}с) ---")


if __name__ == "__main__":
    main()