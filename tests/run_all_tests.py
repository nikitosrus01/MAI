#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import time
import unittest
import xml.etree.ElementTree as ET

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from vlm.vlm_client import OllamaVLMClient
from planner.plan_validator import PlanValidator
from controller.ardupilot_controller import ArduPilotSITLController
from controller.mission_orchestrator import MissionOrchestrator, FramePacket


class TestQuarryVLMSystem(unittest.TestCase):
    def test_A_world_and_collisions(self):
        sdf_path = os.path.join(ROOT_DIR, "worlds/quarry.sdf")
        self.assertTrue(os.path.exists(sdf_path), "Файл worlds/quarry.sdf не найден")
        tree = ET.parse(sdf_path)
        root = tree.getroot()
        world = root.find("world")
        self.assertIsNotNone(world)
        models = {m.attrib.get("name"): m for m in world.findall("model")}
        for expected in ("quarry_floor", "home_helipad", "quarry_terraces_and_slopes", "north_large_boulder", "east_boulder_cluster"):
            self.assertIn(expected, models, f"Модель {expected} отсутствует в мире карьера")
            collisions = models[expected].findall(".//collision")
            self.assertGreater(len(collisions), 0, f"У модели {expected} отсутствуют физические коллизии")

    def test_B_autopilot_controller_and_safety(self):
        with self.assertRaises(PermissionError):
            ArduPilotSITLController(connection_string="/dev/ttyACM0")

        ctrl = ArduPilotSITLController(connection_string="mock://sitl")
        self.assertTrue(ctrl.connect())
        ok, _ = ctrl.arm_and_takeoff(5.0)
        self.assertTrue(ok)
        self.assertAlmostEqual(ctrl.state.z, 5.0, places=1)

        ok, _ = ctrl.goto_local_xyz(16.0, 0.0, 5.5)
        self.assertTrue(ok)
        self.assertAlmostEqual(ctrl.state.x, 16.0, places=1)

        ok, _ = ctrl.safe_return_home([(0.0, 0.0, 5.5)])
        self.assertTrue(ok)
        ok, _ = ctrl.land()
        self.assertTrue(ok)
        self.assertFalse(ctrl.state.armed)

    def test_C_camera_freshness_and_bridge_config(self):
        bridge_cfg = os.path.join(ROOT_DIR, "config/bridge_config.yaml")
        with open(bridge_cfg, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("sensor_msgs/msg/Image", content)
        self.assertIn("/quarry/camera/image_raw", content)

        vlm = OllamaVLMClient(ollama_host="http://127.0.0.1:9", timeout_sec=0.5)
        val = PlanValidator(os.path.join(ROOT_DIR, "config"))
        ctrl = ArduPilotSITLController("mock://sitl")
        ctrl.connect()
        stale_frame = FramePacket(image_bytes=b"test", timestamp=time.monotonic() - 12.0)
        orch = MissionOrchestrator(vlm, val, ctrl, frame_provider=lambda: stale_frame, allow_vlm_fallback=True)
        rep = orch.execute_russian_command("Взлети на высоту 5 метров, найди крупный камень и пролетай над ним")
        self.assertFalse(rep.success)
        self.assertIn("устарело", rep.abort_reason)

    def test_D_vlm_validation_and_malformed_rejection(self):
        val = PlanValidator(os.path.join(ROOT_DIR, "config"))
        drone_state = {"connected": True, "x": 0.0, "y": 0.0, "z": 0.0, "armed": False}

        bad_steps = [
            {"action": "takeoff", "params": {"alt": 95.0}},
            {"action": "execute_shell", "params": {"cmd": "rm -rf /"}},
        ]
        rep = val.validate_plan(bad_steps, drone_state)
        self.assertFalse(rep.is_valid)
        self.assertGreaterEqual(len(rep.errors), 2)

    def test_E_end_to_end_russian_mission(self):
        vlm = OllamaVLMClient(ollama_host="http://127.0.0.1:9", timeout_sec=0.5)
        val = PlanValidator(os.path.join(ROOT_DIR, "config"))
        ctrl = ArduPilotSITLController("mock://sitl")
        ctrl.connect()

        fresh_frame = lambda: FramePacket(image_bytes=b"\x89PNG\r\n\x1a\nFRAME", timestamp=time.monotonic())
        orch = MissionOrchestrator(
            vlm_client=vlm,
            validator=val,
            controller=ctrl,
            frame_provider=fresh_frame,
            allow_vlm_fallback=True,
        )
        cmd = "Взлети на высоту 5 метров, найди крупный камень в северной части карьера, пролетай над ним и затем вернись в исходную точку"
        report = orch.execute_russian_command(cmd)
        self.assertTrue(report.success, f"Сквозной сценарий завершился с ошибкой: {report.abort_reason}")
        self.assertGreaterEqual(len(report.executed_steps), 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
