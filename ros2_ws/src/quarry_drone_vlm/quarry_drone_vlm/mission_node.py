#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import time
import json
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from vlm.vlm_client import OllamaVLMClient
from planner.plan_validator import PlanValidator
from controller.ardupilot_controller import ArduPilotSITLController
from controller.mission_orchestrator import MissionOrchestrator, FramePacket


class QuarryMissionNode(Node):
    def __init__(self):
        super().__init__("quarry_mission_node")
        self.declare_parameter("sitl_connection", "udp:127.0.0.1:14550")
        self.declare_parameter("ollama_host", "http://127.0.0.1:11434")
        self.declare_parameter("vlm_model", "qwen2.5vl:7b")

        conn_str = self.get_parameter("sitl_connection").value
        ollama_host = self.get_parameter("ollama_host").value
        vlm_model = self.get_parameter("vlm_model").value

        self.latest_frame = None
        self.sub_img = self.create_subscription(Image, "/quarry/camera/image_raw", self.on_image, 10)
        self.sub_cmd = self.create_subscription(String, "/quarry/operator/command", self.on_command, 10)
        self.pub_status = self.create_publisher(String, "/quarry/mission/status", 10)

        self.vlm = OllamaVLMClient(ollama_host=ollama_host, model_name=vlm_model)
        self.validator = PlanValidator(os.path.join(ROOT_DIR, "config"))
        self.controller = ArduPilotSITLController(connection_string=conn_str)
        self.orchestrator = MissionOrchestrator(
            vlm_client=self.vlm,
            validator=self.validator,
            controller=self.controller,
            frame_provider=lambda: self.latest_frame,
            status_callback=self.publish_status,
        )
        self.get_logger().info("QuarryMissionNode запущен. Подключение к ArduPilot SITL...")
        self.controller.connect(timeout_sec=10.0)

    def publish_status(self, text: str):
        self.get_logger().info(text)
        msg = String()
        msg.data = text
        self.pub_status.publish(msg)

    def on_image(self, msg: Image):
        self.latest_frame = FramePacket(
            image_bytes=bytes(msg.data),
            timestamp=time.monotonic(),
            width=msg.width,
            height=msg.height,
        )

    def on_command(self, msg: String):
        cmd = msg.data.strip()
        if cmd.lower() in ("abort", "стоп", "авария"):
            self.controller.request_abort()
            self.publish_status("[АВАРИЙНАЯ ОСТАНОВКА] Миссия прервана оператором.")
            return
        report = self.orchestrator.execute_russian_command(cmd)
        self.publish_status(json.dumps({
            "command": report.command_ru,
            "success": report.success,
            "executed_steps": report.executed_steps,
            "abort_reason": report.abort_reason,
            "duration_sec": report.total_duration_sec,
        }, ensure_ascii=False))


def main(args=None):
    rclpy.init(args=args)
    node = QuarryMissionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
