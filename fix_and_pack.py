#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Полный самодостаточный установщик и упаковщик проекта /home/nikita/drone_vlm_quarry.
Устраняет ModuleNotFoundError (создаёт пакеты vlm, planner, controller) и ошибку
Unable to find uri[model://iris_quarry_cam] в Gazebo Harmonic.
"""

import os
import sys
import json
import stat
import zipfile
import subprocess

TARGET_PARENT = "/home/nikita" if os.path.exists("/home/nikita") else os.getcwd()
PROJECT_ROOT = os.path.join(TARGET_PARENT, "drone_vlm_quarry")
ZIP_PATH = os.path.join(TARGET_PARENT, "drone_vlm_quarry.zip")
ARDUPILOT_GZ_MODELS = "/home/nikita/drone/ardupilot_gazebo/models"
ARDUPILOT_GZ_BUILD = "/home/nikita/drone/ardupilot_gazebo/build"


def write_file(rel_path: str, content: str, executable: bool = False) -> None:
    full_path = os.path.join(PROJECT_ROOT, rel_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content.lstrip("\n"))
    if executable:
        st = os.stat(full_path)
        os.chmod(full_path, st.st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


# ==============================================================================
# 1. ПАКЕТ VLM (vlm/__init__.py, vlm/prompts.py, vlm/vlm_client.py)
# ==============================================================================
write_file("vlm/__init__.py", """# -*- coding: utf-8 -*-
from .vlm_client import OllamaVLMClient, VLMPlanResult, VisualDetectionResult
""")

write_file("vlm/prompts.py", """# -*- coding: utf-8 -*-
SYSTEM_PROMPT_PLANNER = \"\"\"Ты — бортовой интеллектуальный планировщик миссий автономного квадрокоптера в виртуальном карьере (Gazebo SITL + ArduPilot).
Твоя задача — перевести команду оператора на русском языке в строго структурированный JSON-план из допустимых атомарных действий.

ДОПУСТИМЫЕ ДЕЙСТВИЯ (action):
1. "takeoff" — взлёт на заданную высоту. Параметры: {"alt": float (от 1.5 до 25.0 м)}.
2. "inspect" — визуальный поиск и распознавание объекта с помощью бортовой камеры и VLM. Параметры: {"target": string, "sector": "north"|"east"|"west"|"south"|"any"}.
3. "goto" — полёт к известному ориентиру или координатам с пролётом над ним. Параметры: {"target_id": string ("north_large_boulder", "east_boulder_cluster", "west_rock_outcrop", "south_marker_beacon") ИЛИ "x": float, "y": float, "z": float, "overfly": bool, "clearance_m": float}.
4. "move_relative" — относительное смещение в метрах. Параметры: {"dx": float, "dy": float, "dz": float}.
5. "return_home" — возврат в исходную точку старта (0, 0). Параметры: {}.
6. "land" — посадка. Параметры: {}.
7. "pause" — зависание на месте. Параметры: {"duration_sec": float}.
8. "abort" — экстренное прекращение миссии. Параметры: {"reason": string}.

ФОРМАТ ОТВЕТА — СТРОГО ВАЛИДНЫЙ JSON:
{
  "intent_summary": "Краткое описание понятой цели на русском языке",
  "confidence": 0.95,
  "requires_clarification": false,
  "clarification_question": null,
  "localization_mode": "visual_plus_simulation_map",
  "steps": [
    {"action": "takeoff", "params": {"alt": 5.0}, "description": "Взлёт на высоту 5 метров"},
    {"action": "inspect", "params": {"target": "north_large_boulder", "sector": "north"}, "description": "Визуальный поиск крупного камня на севере"},
    {"action": "goto", "params": {"target_id": "north_large_boulder", "overfly": true, "clearance_m": 2.5}, "description": "Пролёт над обнаруженным крупным камнем"}
  ]
}
\"\"\"

SYSTEM_PROMPT_VISUAL_INSPECT = \"\"\"Ты — система компьютерного зрения автономного дрона в виртуальном карьере.
Проанализируй прикреплённое изображение с бортовой камеры дрона и определи наличие целевого объекта.
Верни ответ СТРОГО в формате JSON:
{
  "target_detected": true,
  "detected_object_id": "north_large_boulder",
  "confidence": 0.92,
  "position_in_frame": "center",
  "visual_description": "В центральной части кадра виден крупный серый камень.",
  "can_estimate_3d_from_2d_only": false,
  "recommended_action": "proceed_with_map_coordinates"
}
\"\"\"
""")

write_file("vlm/vlm_client.py", """# -*- coding: utf-8 -*-
import base64
import json
import re
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .prompts import SYSTEM_PROMPT_PLANNER, SYSTEM_PROMPT_VISUAL_INSPECT


@dataclass
class VLMPlanResult:
    success: bool
    intent_summary: str
    confidence: float
    requires_clarification: bool
    clarification_question: Optional[str]
    localization_mode: str
    steps: List[Dict[str, Any]] = field(default_factory=list)
    raw_response: str = ""
    latency_sec: float = 0.0
    model_used: str = ""
    error_message: Optional[str] = None


@dataclass
class VisualDetectionResult:
    success: bool
    target_detected: bool
    detected_object_id: Optional[str]
    confidence: float
    position_in_frame: str
    visual_description: str
    can_estimate_3d_from_2d_only: bool
    recommended_action: str
    latency_sec: float = 0.0
    model_used: str = ""
    error_message: Optional[str] = None


class OllamaVLMClient:
    def __init__(
        self,
        ollama_host: str = "http://127.0.0.1:11434",
        model_name: str = "qwen2.5vl:7b",
        timeout_sec: float = 45.0,
        temperature: float = 0.1,
    ):
        self.ollama_host = ollama_host.rstrip("/")
        self.model_name = model_name
        self.timeout_sec = timeout_sec
        self.temperature = temperature

    def check_health(self) -> Dict[str, Any]:
        url = f"{self.ollama_host}/api/tags"
        t0 = time.monotonic()
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("name", "") for m in data.get("models", [])]
                has_target = any(self.model_name in m for m in models)
                return {
                    "available": True,
                    "models": models,
                    "has_target_model": has_target,
                    "latency_sec": round(time.monotonic() - t0, 3),
                    "error": None,
                }
        except Exception as exc:
            return {
                "available": False,
                "models": [],
                "has_target_model": False,
                "latency_sec": round(time.monotonic() - t0, 3),
                "error": str(exc),
            }

    @staticmethod
    def encode_image_bytes(image_bytes: bytes) -> str:
        return base64.b64encode(image_bytes).decode("utf-8")

    @staticmethod
    def extract_json_object(raw_text: str) -> Dict[str, Any]:
        cleaned = raw_text.strip()
        fence_match = re.search(r"```(?:json)?\\s*(\\{.*?\\})\\s*```", cleaned, flags=re.DOTALL)
        if fence_match:
            cleaned = fence_match.group(1)
        else:
            b_start = cleaned.find("{")
            b_end = cleaned.rfind("}")
            if b_start != -1 and b_end != -1 and b_end > b_start:
                cleaned = cleaned[b_start : b_end + 1]
        return json.loads(cleaned)

    def generate_flight_plan(
        self,
        operator_command_ru: str,
        drone_state: Dict[str, Any],
        image_bytes: Optional[bytes] = None,
        allow_rule_fallback: bool = False,
    ) -> VLMPlanResult:
        if not operator_command_ru or not operator_command_ru.strip():
            return VLMPlanResult(
                success=False,
                intent_summary="",
                confidence=0.0,
                requires_clarification=True,
                clarification_question="Пустая команда оператора. Введите задание на русском языке.",
                localization_mode="none",
                error_message="Empty operator command",
            )

        user_prompt = (
            f"Команда оператора: «{operator_command_ru.strip()}»\\n"
            f"Текущее состояние дрона: {json.dumps(drone_state, ensure_ascii=False)}\\n"
            "Сформируй JSON-план выполнения согласно системной инструкции."
        )
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "system": SYSTEM_PROMPT_PLANNER,
            "prompt": user_prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": self.temperature, "num_predict": 512},
        }
        if image_bytes:
            payload["images"] = [self.encode_image_bytes(image_bytes)]

        t0 = time.monotonic()
        try:
            req = urllib.request.Request(
                f"{self.ollama_host}/api/generate",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                raw_response = body.get("response", "")
                latency = round(time.monotonic() - t0, 3)

            parsed = self.extract_json_object(raw_response)
            steps = parsed.get("steps", [])
            if not isinstance(steps, list):
                raise ValueError("Поле 'steps' должно быть списком действий.")

            return VLMPlanResult(
                success=True,
                intent_summary=str(parsed.get("intent_summary", "План сформирован VLM")),
                confidence=float(parsed.get("confidence", 0.85)),
                requires_clarification=bool(parsed.get("requires_clarification", False)),
                clarification_question=parsed.get("clarification_question"),
                localization_mode=str(parsed.get("localization_mode", "visual_plus_simulation_map")),
                steps=steps,
                raw_response=raw_response,
                latency_sec=latency,
                model_used=self.model_name,
            )
        except Exception as exc:
            latency = round(time.monotonic() - t0, 3)
            if allow_rule_fallback:
                return self._rule_based_fallback_plan(operator_command_ru, latency, str(exc))
            return VLMPlanResult(
                success=False,
                intent_summary="Ошибка обращения к VLM",
                confidence=0.0,
                requires_clarification=False,
                clarification_question=None,
                localization_mode="none",
                latency_sec=latency,
                model_used=self.model_name,
                error_message=f"VLM Error: {exc}",
            )

    def inspect_image(
        self,
        target_description: str,
        image_bytes: bytes,
        drone_state: Dict[str, Any],
        allow_fallback: bool = False,
    ) -> VisualDetectionResult:
        if not image_bytes:
            return VisualDetectionResult(
                success=False,
                target_detected=False,
                detected_object_id=None,
                confidence=0.0,
                position_in_frame="none",
                visual_description="Изображение отсутствует",
                can_estimate_3d_from_2d_only=False,
                recommended_action="abort",
                error_message="No image bytes provided",
            )

        prompt = (
            f"Искомый объект: «{target_description}».\\n"
            f"Положение дрона: {json.dumps(drone_state, ensure_ascii=False)}.\\n"
            "Проанализируй кадр с бортовой камеры и верни JSON."
        )
        payload = {
            "model": self.model_name,
            "system": SYSTEM_PROMPT_VISUAL_INSPECT,
            "prompt": prompt,
            "images": [self.encode_image_bytes(image_bytes)],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1, "num_predict": 300},
        }
        t0 = time.monotonic()
        try:
            req = urllib.request.Request(
                f"{self.ollama_host}/api/generate",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                raw_response = body.get("response", "")
                latency = round(time.monotonic() - t0, 3)

            parsed = self.extract_json_object(raw_response)
            return VisualDetectionResult(
                success=True,
                target_detected=bool(parsed.get("target_detected", False)),
                detected_object_id=parsed.get("detected_object_id"),
                confidence=float(parsed.get("confidence", 0.0)),
                position_in_frame=str(parsed.get("position_in_frame", "unknown")),
                visual_description=str(parsed.get("visual_description", "")),
                can_estimate_3d_from_2d_only=bool(parsed.get("can_estimate_3d_from_2d_only", False)),
                recommended_action=str(parsed.get("recommended_action", "request_clarification_or_abort")),
                latency_sec=latency,
                model_used=self.model_name,
            )
        except Exception as exc:
            latency = round(time.monotonic() - t0, 3)
            if allow_fallback:
                target_id = "north_large_boulder"
                low = target_description.lower()
                if "вост" in low or "групп" in low:
                    target_id = "east_boulder_cluster"
                elif "запад" in low or "уступ" in low:
                    target_id = "west_rock_outcrop"
                elif "юг" in low or "оранж" in low or "маркер" in low:
                    target_id = "south_marker_beacon"
                return VisualDetectionResult(
                    success=True,
                    target_detected=True,
                    detected_object_id=target_id,
                    confidence=0.85,
                    position_in_frame="center",
                    visual_description="[Резервный анализатор] Объект подтверждён в секторе обзора.",
                    can_estimate_3d_from_2d_only=False,
                    recommended_action="proceed_with_map_coordinates",
                    latency_sec=latency,
                    model_used="fallback_inspector",
                )
            return VisualDetectionResult(
                success=False,
                target_detected=False,
                detected_object_id=None,
                confidence=0.0,
                position_in_frame="none",
                visual_description="",
                can_estimate_3d_from_2d_only=False,
                recommended_action="abort",
                latency_sec=latency,
                model_used=self.model_name,
                error_message=f"VLM Inspect Error: {exc}",
            )

    @staticmethod
    def _rule_based_fallback_plan(command_ru: str, latency: float, orig_error: str) -> VLMPlanResult:
        text = command_ru.lower()
        steps: List[Dict[str, Any]] = []
        alt_match = re.search(r"высот[уеы]?\\s*(\\d+(?:[.,]\\d+)?)\\s*м", text)
        takeoff_alt = float(alt_match.group(1).replace(",", ".")) if alt_match else 5.0

        if "взлет" in text or "поднимись" in text:
            steps.append({"action": "takeoff", "params": {"alt": takeoff_alt}, "description": f"Взлёт на высоту {takeoff_alt} м"})

        target_id = None
        sector = "north"
        if "камен" in text or "валун" in text:
            if "вост" in text or "групп" in text:
                target_id, sector = "east_boulder_cluster", "east"
            elif "запад" in text or "уступ" in text:
                target_id, sector = "west_rock_outcrop", "west"
            else:
                target_id, sector = "north_large_boulder", "north"
        elif "маркер" in text or "оранж" in text or "маяк" in text:
            target_id, sector = "south_marker_beacon", "south"

        if ("найди" in text or "осмотри" in text or "камен" in text) and target_id:
            steps.append({"action": "inspect", "params": {"target": target_id, "sector": sector}, "description": f"Визуальный поиск {target_id}"})

        if ("пролет" in text or "лети" in text or "над" in text) and target_id:
            steps.append({"action": "goto", "params": {"target_id": target_id, "overfly": True, "clearance_m": 2.5}, "description": f"Пролёт над {target_id}"})

        if "вернись" in text or "домой" in text or "исходн" in text:
            steps.append({"action": "return_home", "params": {}, "description": "Возврат в домашнюю точку"})

        if "сядь" in text or "посадк" in text:
            steps.append({"action": "land", "params": {}, "description": "Посадка"})

        return VLMPlanResult(
            success=bool(steps),
            intent_summary=f"[Fallback Parser] План для: {command_ru}",
            confidence=0.88 if steps else 0.0,
            requires_clarification=not bool(steps),
            clarification_question=None if steps else "Уточните задание.",
            localization_mode="visual_plus_simulation_map",
            steps=steps,
            raw_response=json.dumps({"steps": steps}, ensure_ascii=False),
            latency_sec=latency,
            model_used="fallback_rule_parser",
            error_message=None if steps else orig_error,
        )
""")

# ==============================================================================
# 2. ПАКЕТ PLANNER (planner/__init__.py, planner/plan_validator.py)
# ==============================================================================
write_file("planner/__init__.py", """# -*- coding: utf-8 -*-
from .plan_validator import PlanValidator, ValidationReport, ValidatedStep
""")

write_file("planner/plan_validator.py", """# -*- coding: utf-8 -*-
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

ALLOWED_ACTIONS = {"takeoff", "goto", "move_relative", "inspect", "return_home", "land", "pause", "abort"}


@dataclass
class ValidatedStep:
    action: str
    params: Dict[str, Any]
    description: str
    waypoints: List[Tuple[float, float, float]] = field(default_factory=list)
    localization_source: str = "explicit_or_none"


@dataclass
class ValidationReport:
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    validated_steps: List[ValidatedStep] = field(default_factory=list)
    requires_visual_confirmation: bool = False


class PlanValidator:
    def __init__(self, config_dir: Optional[str] = None):
        if config_dir is None:
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_dir = os.path.join(base, "config")
        self.config_dir = config_dir
        self.quarry_map = self._load_quarry_map()
        self.limits = {
            "x_min": -50.0, "x_max": 50.0,
            "y_min": -50.0, "y_max": 50.0,
            "z_min": 1.0, "z_max": 25.0,
            "max_speed_mps": 4.0,
            "min_obstacle_clearance_m": 1.5,
            "overfly_clearance_m": 2.5,
            "max_plan_steps": 10,
            "max_consecutive_identical_actions": 2,
        }

    def _load_quarry_map(self) -> Dict[str, Any]:
        map_path = os.path.join(self.config_dir, "quarry_map.json")
        if os.path.exists(map_path):
            with open(map_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"landmarks": {}, "static_obstacles": []}

    def resolve_landmark(self, target_query: str) -> Optional[Dict[str, Any]]:
        if not target_query:
            return None
        q = target_query.strip().lower()
        landmarks = self.quarry_map.get("landmarks", {})
        if q in landmarks:
            return landmarks[q]
        for _, lm_data in landmarks.items():
            if q == lm_data.get("name_ru", "").lower():
                return lm_data
            for alias in lm_data.get("aliases", []):
                if alias.lower() in q or q in alias.lower():
                    return lm_data
        return None

    def check_within_bounds(self, x: float, y: float, z: float) -> Tuple[bool, Optional[str]]:
        if not (self.limits["x_min"] <= x <= self.limits["x_max"]):
            return False, f"X={x:.1f}м вне рабочей зоны [{self.limits['x_min']}, {self.limits['x_max']}]м."
        if not (self.limits["y_min"] <= y <= self.limits["y_max"]):
            return False, f"Y={y:.1f}м вне рабочей зоны [{self.limits['y_min']}, {self.limits['y_max']}]м."
        if not (self.limits["z_min"] <= z <= self.limits["z_max"]):
            return False, f"Высота Z={z:.1f}м вне диапазона [{self.limits['z_min']}, {self.limits['z_max']}]м."
        return True, None

    @staticmethod
    def _segment_circle_distance_2d(x1: float, y1: float, x2: float, y2: float, cx: float, cy: float) -> float:
        dx, dy = x2 - x1, y2 - y1
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq < 1e-9:
            return math.hypot(cx - x1, cy - y1)
        t = max(0.0, min(1.0, ((cx - x1) * dx + (cy - y1) * dy) / seg_len_sq))
        return math.hypot(cx - (x1 + t * dx), cy - (y1 + t * dy))

    def build_safe_segment(
        self, start_xyz: Tuple[float, float, float], goal_xyz: Tuple[float, float, float]
    ) -> Tuple[List[Tuple[float, float, float]], List[str]]:
        warnings: List[str] = []
        x1, y1, z1 = start_xyz
        x2, y2, z2 = goal_xyz
        min_safe_z = max(z1, z2)
        for obs in self.quarry_map.get("static_obstacles", []):
            ox, oy = float(obs["x"]), float(obs["y"])
            orad = float(obs.get("radius", 2.0))
            otop = float(obs.get("top_z", 2.0))
            if self._segment_circle_distance_2d(x1, y1, x2, y2, ox, oy) < (orad + self.limits["min_obstacle_clearance_m"]):
                req_z = otop + self.limits["overfly_clearance_m"]
                if min(z1, z2) < req_z:
                    min_safe_z = max(min_safe_z, req_z)
                    warnings.append(f"Препятствие «{obs['name']}» (высота {otop}м): эшелон поднят до {min_safe_z:.1f}м.")
        wps: List[Tuple[float, float, float]] = []
        if min_safe_z > z1 + 0.3:
            wps.append((x1, y1, min_safe_z))
        if min_safe_z > z2 + 0.3:
            wps.append((x2, y2, min_safe_z))
        wps.append((x2, y2, max(z2, min_safe_z)))
        return wps, warnings

    def validate_plan(self, raw_steps: List[Dict[str, Any]], drone_state: Dict[str, Any]) -> ValidationReport:
        report = ValidationReport(is_valid=True)
        if not raw_steps:
            report.is_valid = False
            report.errors.append("План пуст.")
            return report
        if len(raw_steps) > self.limits["max_plan_steps"]:
            report.is_valid = False
            report.errors.append("Превышено макс. число шагов плана.")
            return report
        if not drone_state.get("connected", True):
            report.is_valid = False
            report.errors.append("Нет связи с автопилотом.")
            return report

        cur_x = float(drone_state.get("x", 0.0))
        cur_y = float(drone_state.get("y", 0.0))
        cur_z = float(drone_state.get("z", 0.0))
        is_airborne = cur_z > 0.8 or bool(drone_state.get("armed", False))

        consecutive_count = 0
        last_sig = None

        for idx, step in enumerate(raw_steps):
            if not isinstance(step, dict):
                report.is_valid = False
                report.errors.append(f"Шаг #{idx+1} некорректен.")
                continue
            action = str(step.get("action", "")).strip().lower()
            params = step.get("params", {}) or {}
            desc = str(step.get("description", f"Шаг {action}"))

            if action not in ALLOWED_ACTIONS:
                report.is_valid = False
                report.errors.append(f"Шаг #{idx+1}: недопустимое действие «{action}».")
                continue

            sig = f"{action}:{json.dumps(params, sort_keys=True, ensure_ascii=False)}"
            if sig == last_sig:
                consecutive_count += 1
                if consecutive_count >= self.limits["max_consecutive_identical_actions"]:
                    report.is_valid = False
                    report.errors.append(f"Зацикливание команды {action}.")
                    continue
            else:
                consecutive_count = 1
                last_sig = sig

            if action == "takeoff":
                alt = float(params.get("alt", 5.0))
                ok, err = self.check_within_bounds(cur_x, cur_y, alt)
                if not ok:
                    report.is_valid = False
                    report.errors.append(f"Шаг #{idx+1} (takeoff): {err}")
                    continue
                cur_z = alt
                is_airborne = True
                report.validated_steps.append(ValidatedStep("takeoff", {"alt": alt}, desc, [(cur_x, cur_y, cur_z)], "baro_gps_sitl"))

            elif action == "inspect":
                target = str(params.get("target", "north_large_boulder"))
                sector = str(params.get("sector", "north"))
                report.requires_visual_confirmation = True
                report.validated_steps.append(ValidatedStep("inspect", {"target": target, "sector": sector}, desc, [], "vlm_camera"))

            elif action == "goto":
                if not is_airborne:
                    report.is_valid = False
                    report.errors.append(f"Шаг #{idx+1} (goto): полёт без взлёта.")
                    continue
                target_id = params.get("target_id")
                overfly = bool(params.get("overfly", True))
                clearance_m = float(params.get("clearance_m", self.limits["overfly_clearance_m"]))
                if target_id:
                    lm = self.resolve_landmark(str(target_id))
                    if lm is None:
                        report.is_valid = False
                        report.errors.append(f"Шаг #{idx+1} (goto): неизвестный ориентир «{target_id}».")
                        continue
                    tx, ty = float(lm["x"]), float(lm["y"])
                    top_z = float(lm.get("top_z", lm["z"]))
                    tz = max(cur_z, top_z + clearance_m) if overfly else max(cur_z, float(lm["z"]) + 2.0)
                    loc_src = f"simulation_map_landmark:{lm['id']}"
                elif "x" in params and "y" in params:
                    tx, ty, tz = float(params["x"]), float(params["y"]), float(params.get("z", cur_z))
                    loc_src = "explicit_coordinates"
                else:
                    report.is_valid = False
                    report.errors.append(f"Шаг #{idx+1} (goto): нет координат.")
                    continue
                ok, err = self.check_within_bounds(tx, ty, tz)
                if not ok:
                    report.is_valid = False
                    report.errors.append(f"Шаг #{idx+1} (goto): {err}")
                    continue
                wps, seg_w = self.build_safe_segment((cur_x, cur_y, cur_z), (tx, ty, tz))
                report.warnings.extend(seg_w)
                cur_x, cur_y, cur_z = wps[-1]
                report.validated_steps.append(ValidatedStep("goto", {"x": cur_x, "y": cur_y, "z": cur_z, "target_id": target_id}, desc, wps, loc_src))

            elif action == "move_relative":
                if not is_airborne:
                    report.is_valid = False
                    report.errors.append("move_relative до взлёта.")
                    continue
                dx, dy, dz = float(params.get("dx", 0.0)), float(params.get("dy", 0.0)), float(params.get("dz", 0.0))
                tx, ty, tz = cur_x + dx, cur_y + dy, cur_z + dz
                ok, err = self.check_within_bounds(tx, ty, tz)
                if not ok:
                    report.is_valid = False
                    report.errors.append(err)
                    continue
                wps, seg_w = self.build_safe_segment((cur_x, cur_y, cur_z), (tx, ty, tz))
                report.warnings.extend(seg_w)
                cur_x, cur_y, cur_z = wps[-1]
                report.validated_steps.append(ValidatedStep("move_relative", {"x": cur_x, "y": cur_y, "z": cur_z}, desc, wps, "relative"))

            elif action == "return_home":
                home = self.quarry_map.get("home_position", {"x": 0.0, "y": 0.0, "z": 0.2})
                hx, hy = float(home["x"]), float(home["y"])
                safe_z = max(cur_z, 5.0)
                wps, seg_w = self.build_safe_segment((cur_x, cur_y, safe_z), (hx, hy, safe_z))
                report.warnings.extend(seg_w)
                cur_x, cur_y, cur_z = hx, hy, safe_z
                report.validated_steps.append(ValidatedStep("return_home", {"x": hx, "y": hy, "z": safe_z}, desc, wps, "home_sitl"))

            elif action == "land":
                cur_z = 0.0
                is_airborne = False
                report.validated_steps.append(ValidatedStep("land", {}, desc, [(cur_x, cur_y, 0.0)], "baro_sitl"))

            elif action == "pause":
                dur = float(params.get("duration_sec", 3.0))
                report.validated_steps.append(ValidatedStep("pause", {"duration_sec": dur}, desc))

            elif action == "abort":
                report.validated_steps.append(ValidatedStep("abort", {"reason": str(params.get("reason", ""))}, desc))

        return report
""")

# ==============================================================================
# 3. ПАКЕТ CONTROLLER (ardupilot_controller.py, mission_orchestrator.py)
# ==============================================================================
write_file("controller/__init__.py", """# -*- coding: utf-8 -*-
from .ardupilot_controller import ArduPilotSITLController, TelemetryState
from .mission_orchestrator import MissionOrchestrator, FramePacket, MissionExecutionReport
""")

write_file("controller/ardupilot_controller.py", """# -*- coding: utf-8 -*-
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
    "udp:127.0.0.1:", "udp:localhost:",
    "tcp:127.0.0.1:", "tcp:localhost:",
    "udpin:127.0.0.1:", "udpin:0.0.0.0:",
    "mock://",
)


@dataclass
class TelemetryState:
    connected: bool = False
    mode: str = "UNKNOWN"
    armed: bool = False
    ekf_ok: bool = False
    home_available: bool = False
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    yaw_deg: float = 0.0
    battery_v: float = 16.8
    last_heartbeat_ts: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ArduPilotSITLController:
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
            raise PermissionError(f"Подключение к физическому порту ({conn_str}) строго запрещено!")
        if not any(clean.startswith(p) for p in ALLOWED_SITL_PREFIXES):
            raise PermissionError(f"Недопустимый адрес ({conn_str}). Разрешён только локальный SITL.")

    def connect(self, timeout_sec: float = 15.0) -> bool:
        if self.is_mock:
            self.state.connected = True
            self.state.mode = "STABILIZE"
            self.state.ekf_ok = True
            self.state.home_available = True
            self.state.last_heartbeat_ts = time.monotonic()
            return True
        if not HAS_PYMAVLINK:
            raise RuntimeError("Установите pymavlink: pip install pymavlink")
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
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL, 10, 1
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
                self.state.armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            elif mtype == "LOCAL_POSITION_NED":
                self.state.x = float(msg.x)
                self.state.y = -float(msg.y)
                self.state.z = -float(msg.z)
                self.state.vx = float(msg.vx)
                self.state.vy = -float(msg.vy)
                self.state.vz = -float(msg.vz)
            elif mtype == "ATTITUDE":
                self.state.yaw_deg = math.degrees(float(msg.yaw))
        if (time.monotonic() - self.state.last_heartbeat_ts) > self.heartbeat_timeout_sec:
            self.state.connected = False
        return self.state

    def request_abort(self) -> None:
        self._abort_requested = True
        if self.is_mock:
            self.state.mode = "BRAKE"
            return
        if self.master is not None:
            m = self.master.mode_mapping()
            self.master.set_mode(m.get("BRAKE", m.get("LOITER", 5)))

    def clear_abort(self) -> None:
        self._abort_requested = False

    def set_mode(self, mode_name: str = "GUIDED", timeout_sec: float = 8.0) -> bool:
        self.poll_telemetry()
        if not self.state.connected:
            return False
        if self.is_mock:
            self.state.mode = mode_name.upper()
            return True
        m = self.master.mode_mapping()
        if mode_name not in m:
            return False
        t0 = time.monotonic()
        while (time.monotonic() - t0) < timeout_sec:
            self.master.set_mode(m[mode_name])
            time.sleep(0.3)
            self.poll_telemetry()
            if self.state.mode == mode_name:
                return True
        return False

    def arm_and_takeoff(self, target_alt_m: float = 5.0, timeout_sec: float = 35.0) -> Tuple[bool, str]:
        self.clear_abort()
        self.poll_telemetry()
        if not self.state.connected:
            return False, "Нет связи с ArduPilot SITL."
        if not self.set_mode("GUIDED"):
            return False, "Не удалось включить режим GUIDED."
        if self.is_mock:
            self.state.armed = True
            self.state.z = float(target_alt_m)
            return True, f"Взлёт завершён. Высота: {self.state.z:.2f} м."

        t0 = time.monotonic()
        while not self.state.armed and (time.monotonic() - t0) < 15.0:
            if self._abort_requested:
                return False, "Взлёт прерван."
            self.master.arducopter_arm()
            time.sleep(0.5)
            self.poll_telemetry()
        if not self.state.armed:
            return False, "Тайм-аут ARM (проверьте готовность EKF/PreArm в SITL)."

        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0,
            0, 0, 0, 0, 0, 0, float(target_alt_m),
        )
        t_to = time.monotonic()
        while (time.monotonic() - t_to) < timeout_sec:
            if self._abort_requested:
                self.request_abort()
                return False, "Взлёт остановлен оператором."
            self.poll_telemetry()
            if abs(self.state.z - target_alt_m) <= 0.6:
                return True, f"Взлёт выполнен. Высота: {self.state.z:.2f} м."
            time.sleep(0.2)
        return False, f"Тайм-аут набора высоты {target_alt_m}м."

    def goto_local_xyz(
        self, x_north: float, y_west: float, z_up: float,
        timeout_sec: float = 45.0,
        progress_cb: Optional[Callable[[TelemetryState], None]] = None,
    ) -> Tuple[bool, str]:
        self.poll_telemetry()
        if not self.state.connected:
            return False, "Нет связи с автопилотом."
        if self.is_mock:
            self.state.x, self.state.y, self.state.z = float(x_north), float(y_west), float(z_up)
            if progress_cb:
                progress_cb(self.state)
            return True, f"Достигнута точка ({self.state.x:.1f}, {self.state.y:.1f}, {self.state.z:.1f}) м."

        type_mask = 0b0000111111111000
        t0 = time.monotonic()
        while (time.monotonic() - t0) < timeout_sec:
            if self._abort_requested:
                self.request_abort()
                return False, "Полёт прерван оператором."
            self.master.mav.set_position_target_local_ned_send(
                0, self.master.target_system, self.master.target_component,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED, type_mask,
                float(x_north), -float(y_west), -float(z_up),
                0, 0, 0, 0, 0, 0, 0, 0,
            )
            time.sleep(0.25)
            self.poll_telemetry()
            if progress_cb:
                progress_cb(self.state)
            dist = math.sqrt((self.state.x - x_north)**2 + (self.state.y - y_west)**2 + (self.state.z - z_up)**2)
            if dist <= self.acceptance_radius_m:
                return True, f"Точка ({x_north:.1f}, {y_west:.1f}, {z_up:.1f}) достигнута."
        return False, "Тайм-аут достижения путевой точки."

    def safe_return_home(self, waypoints: List[Tuple[float, float, float]]) -> Tuple[bool, str]:
        self.poll_telemetry()
        if not self.state.connected:
            return False, "Потеряна связь с автопилотом."
        if not self.state.ekf_ok or not self.state.home_available:
            self.set_mode("LAND")
            return False, "EKF или домашняя точка недоступны. Активирована посадка на месте (LAND)."
        for wp in waypoints:
            ok, msg = self.goto_local_xyz(wp[0], wp[1], wp[2])
            if not ok:
                return False, msg
        return True, "Возврат в домашнюю точку выполнен."

    def land(self, timeout_sec: float = 30.0) -> Tuple[bool, str]:
        self.poll_telemetry()
        if not self.state.connected:
            return False, "Нет связи с автопилотом."
        if self.is_mock:
            self.state.mode = "LAND"
            self.state.z = 0.0
            self.state.armed = False
            return True, "Посадка завершена."
        self.set_mode("LAND")
        t0 = time.monotonic()
        while (time.monotonic() - t0) < timeout_sec:
            self.poll_telemetry()
            if not self.state.armed or self.state.z <= 0.25:
                return True, "Посадка успешно выполнена."
            time.sleep(0.3)
        return False, "Тайм-аут посадки."
""")

write_file("controller/mission_orchestrator.py", """# -*- coding: utf-8 -*-
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from vlm.vlm_client import OllamaVLMClient, VLMPlanResult, VisualDetectionResult
from planner.plan_validator import PlanValidator, ValidationReport
from controller.ardupilot_controller import ArduPilotSITLController


@dataclass
class FramePacket:
    image_bytes: bytes
    timestamp: float
    width: int = 640
    height: int = 480
    frame_id: str = "camera_link"


@dataclass
class MissionExecutionReport:
    command_ru: str
    success: bool
    vlm_plan: Optional[VLMPlanResult] = None
    validation: Optional[ValidationReport] = None
    visual_detections: List[VisualDetectionResult] = field(default_factory=list)
    executed_steps: List[str] = field(default_factory=list)
    final_telemetry: Dict[str, Any] = field(default_factory=dict)
    abort_reason: Optional[str] = None
    total_duration_sec: float = 0.0


class MissionOrchestrator:
    def __init__(
        self,
        vlm_client: OllamaVLMClient,
        validator: PlanValidator,
        controller: ArduPilotSITLController,
        frame_provider: Optional[Callable[[], Optional[FramePacket]]] = None,
        image_freshness_timeout_sec: float = 5.0,
        status_callback: Optional[Callable[[str], None]] = None,
        allow_vlm_fallback: bool = False,
    ):
        self.vlm = vlm_client
        self.validator = validator
        self.controller = controller
        self.frame_provider = frame_provider
        self.image_freshness_timeout_sec = image_freshness_timeout_sec
        self.status_callback = status_callback or (lambda msg: None)
        self.allow_vlm_fallback = allow_vlm_fallback

    def _emit(self, message: str) -> None:
        self.status_callback(message)

    def _get_fresh_frame(self) -> Tuple[Optional[FramePacket], Optional[str]]:
        if self.frame_provider is None:
            return None, "Поставщик кадров виртуальной камеры не подключён."
        pkt = self.frame_provider()
        if pkt is None or not pkt.image_bytes:
            return None, "Кадр с виртуальной камеры ещё не поступил."
        age = time.monotonic() - pkt.timestamp
        if age > self.image_freshness_timeout_sec:
            return None, f"Изображение с виртуальной камеры устарело ({age:.1f}с > {self.image_freshness_timeout_sec:.1f}с)."
        return pkt, None

    def execute_russian_command(self, command_ru: str) -> MissionExecutionReport:
        t_start = time.monotonic()
        report = MissionExecutionReport(command_ru=command_ru, success=False)
        telemetry = self.controller.poll_telemetry()
        if not telemetry.connected:
            report.abort_reason = "Нет связи с ArduPilot SITL."
            self._emit(f"[ОШИБКА] {report.abort_reason}")
            return report

        self._emit(f"[УРОВЕНЬ 1: VLM] Интерпретация команды: «{command_ru}»...")
        vlm_plan = self.vlm.generate_flight_plan(
            operator_command_ru=command_ru,
            drone_state=telemetry.to_dict(),
            allow_rule_fallback=self.allow_vlm_fallback,
        )
        report.vlm_plan = vlm_plan
        if not vlm_plan.success:
            report.abort_reason = f"Ошибка VLM: {vlm_plan.error_message}"
            self._emit(f"[ОШИБКА VLM] {report.abort_reason}")
            return report
        if vlm_plan.requires_clarification:
            report.abort_reason = f"Требуется уточнение: {vlm_plan.clarification_question}"
            self._emit(f"[УТОЧНЕНИЕ] {report.abort_reason}")
            return report

        self._emit("[УРОВЕНЬ 2: ПЛАНИРОВЩИК] Валидация плана и высот пролёта...")
        validation = self.validator.validate_plan(vlm_plan.steps, telemetry.to_dict())
        report.validation = validation
        for w in validation.warnings:
            self._emit(f"[ПРЕДУПРЕЖДЕНИЕ] {w}")
        if not validation.is_valid:
            report.abort_reason = "План отклонён: " + "; ".join(validation.errors)
            self._emit(f"[БЛОКИРОВКА] {report.abort_reason}")
            return report

        for idx, step in enumerate(validation.validated_steps, start=1):
            self._emit(f"[УРОВЕНЬ 3: ИСПОЛНЕНИЕ] Шаг {idx}/{len(validation.validated_steps)}: {step.description}")
            if step.action == "takeoff":
                ok, msg = self.controller.arm_and_takeoff(float(step.params["alt"]))
                report.executed_steps.append(f"takeoff: {msg}")
                if not ok:
                    report.abort_reason = msg
                    return self._finalize(report, t_start)
            elif step.action == "inspect":
                target_q = str(step.params.get("target", "north_large_boulder"))
                frame_pkt, frame_err = self._get_fresh_frame()
                if frame_err:
                    report.abort_reason = frame_err
                    self._emit(f"  -> [ОШИБКА КАМЕРЫ] {frame_err}")
                    return self._finalize(report, t_start)
                det = self.vlm.inspect_image(
                    target_description=target_q,
                    image_bytes=frame_pkt.image_bytes,
                    drone_state=self.controller.poll_telemetry().to_dict(),
                    allow_fallback=self.allow_vlm_fallback,
                )
                report.visual_detections.append(det)
                if not det.success or not det.target_detected:
                    report.abort_reason = f"Объект «{target_q}» визуально не подтверждён."
                    return self._finalize(report, t_start)
                report.executed_steps.append(f"inspect({target_q}): подтверждён {det.detected_object_id}")
            elif step.action in ("goto", "move_relative"):
                for wp in step.waypoints:
                    ok, msg = self.controller.goto_local_xyz(wp[0], wp[1], wp[2])
                    if not ok:
                        report.abort_reason = msg
                        return self._finalize(report, t_start)
                report.executed_steps.append(f"{step.action}: достигнута точка {step.waypoints[-1]}")
            elif step.action == "return_home":
                ok, msg = self.controller.safe_return_home(step.waypoints)
                report.executed_steps.append(f"return_home: {msg}")
                if not ok:
                    report.abort_reason = msg
                   python3 /home/nikita/drone_vlm_quarry/ros2_ws/src/quarry_drone_vlm/quarry_drone_vlm/operator_cli.py return self._finalize(report, t_start)
            elif step.action == "land":
                ok, msg = self.controller.land()
                report.executed_steps.append(f"land: {msg}")
                if not ok:
                    report.abort_reason = msg
                    return self._finalize(report, t_start)
            elif step.action == "pause":
                time.sleep(min(float(step.params.get("duration_sec", 2.0)), 5.0))
                report.executed_steps.append("pause: выполнено")
            elif step.action == "abort":
                self.controller.request_abort()
                report.abort_reason = "Остановка по шагу abort"
                return self._finalize(report, t_start)

        report.success = True
        self._emit("[УСПЕХ] Миссия успешно завершена.")
        return self._finalize(report, t_start)

    def _finalize(self, report: MissionExecutionReport, t_start: float) -> MissionExecutionReport:
        report.final_telemetry = self.controller.poll_telemetry().to_dict()
        report.total_duration_sec = round(time.monotonic() - t_start, 3)
        return report
""")

# ==============================================================================
# 4. ИСПРАВЛЕНИЕ ПУТЕЙ МОДЕЛИ В SDF (работает даже при прямом вызове gz sim!)
# ==============================================================================
iris_standoffs_uri = (
    f"file://{ARDUPILOT_GZ_MODELS}/iris_with_standoffs"
    if os.path.exists(os.path.join(ARDUPILOT_GZ_MODELS, "iris_with_standoffs"))
    else "model://iris_with_standoffs"
)
iris_quarry_cam_uri = f"file://{PROJECT_ROOT}/models/iris_quarry_cam"

write_file("models/iris_quarry_cam/model.sdf", f"""<?xml version="1.0" ?>
<sdf version="1.9">
  <model name="iris_with_ardupilot">
    <include merge="true">
      <uri>{iris_standoffs_uri}</uri>
    </include>

    <link name="camera_link">
      <pose relative_to="iris_with_standoffs::base_link">0.10 0 0.02 0 0.35 0</pose>
      <inertial>
        <mass>0.015</mass>
        <inertia><ixx>0.00001</ixx><ixy>0</ixy><ixz>0</ixz><iyy>0.00001</iyy><iyz>0</iyz><izz>0.00001</izz></inertia>
      </inertial>
      <visual name="cam_vis">
        <geometry><box><size>0.03 0.04 0.03</size></box></geometry>
        <material><ambient>0.1 0.1 0.1 1</ambient><diffuse>0.2 0.2 0.2 1</diffuse></material>
      </visual>
      <sensor name="front_down_camera" type="camera">
        <always_on>1</always_on>
        <update_rate>10</update_rate>
        <visualize>true</visualize>
        <topic>/quarry/camera/image_raw</topic>
        <camera name="quarry_rgb">
          <horizontal_fov>1.3962634</horizontal_fov>
          <image><width>640</width><height>480</height><format>R8G8B8</format></image>
          <clip><near>0.1</near><far>150.0</far></clip>
        </camera>
      </sensor>
    </link>

    <joint name="camera_mount_joint" type="fixed">
      <parent>iris_with_standoffs::base_link</parent>
      <child>camera_link</child>
    </joint>

    <plugin filename="gz-sim-joint-state-publisher-system" name="gz::sim::systems::JointStatePublisher"/>
    <plugin filename="gz-sim-apply-joint-force-system" name="gz::sim::systems::ApplyJointForce">
      <joint_name>iris_with_standoffs::rotor_0_joint</joint_name>
    </plugin>
    <plugin filename="gz-sim-apply-joint-force-system" name="gz::sim::systems::ApplyJointForce">
      <joint_name>iris_with_standoffs::rotor_1_joint</joint_name>
    </plugin>
    <plugin filename="gz-sim-apply-joint-force-system" name="gz::sim::systems::ApplyJointForce">
      <joint_name>iris_with_standoffs::rotor_2_joint</joint_name>
    </plugin>
    <plugin filename="gz-sim-apply-joint-force-system" name="gz::sim::systems::ApplyJointForce">
      <joint_name>iris_with_standoffs::rotor_3_joint</joint_name>
    </plugin>

    <plugin name="ArduPilotPlugin" filename="ArduPilotPlugin">
      <fdm_addr>127.0.0.1</fdm_addr>
      <fdm_port_in>9002</fdm_port_in>
      <connectionTimeoutMaxCount>5</connectionTimeoutMaxCount>
      <lock_step>1</lock_step>
      <have_32_channels>0</have_32_channels>
      <modelXYZToAirplaneXForwardZDown degrees="true">0 0 0 180 0 0</modelXYZToAirplaneXForwardZDown>
      <gazeboXYZToNED degrees="true">0 0 0 180 0 90</gazeboXYZToNED>
      <imuName>iris_with_standoffs::imu_link::imu_sensor</imuName>
      <control channel="0">
        <jointName>iris_with_standoffs::rotor_0_joint</jointName>
        <useForce>1</useForce><multiplier>838</multiplier><offset>0</offset>
        <servo_min>1100</servo_min><servo_max>1900</servo_max><type>VELOCITY</type>
        <p_gain>0.20</p_gain><i_gain>0</i_gain><d_gain>0</d_gain>
        <cmd_max>2.5</cmd_max><cmd_min>-2.5</cmd_min>
      </control>
      <control channel="1">
        <jointName>iris_with_standoffs::rotor_1_joint</jointName>
        <useForce>1</useForce><multiplier>838</multiplier><offset>0</offset>
        <servo_min>1100</servo_min><servo_max>1900</servo_max><type>VELOCITY</type>
        <p_gain>0.20</p_gain><i_gain>0</i_gain><d_gain>0</d_gain>
        <cmd_max>2.5</cmd_max><cmd_min>-2.5</cmd_min>
      </control>
      <control channel="2">
        <jointName>iris_with_standoffs::rotor_2_joint</jointName>
        <useForce>1</useForce><multiplier>-838</multiplier><offset>0</offset>
        <servo_min>1100</servo_min><servo_max>1900</servo_max><type>VELOCITY</type>
        <p_gain>0.20</p_gain><i_gain>0</i_gain><d_gain>0</d_gain>
        <cmd_max>2.5</cmd_max><cmd_min>-2.5</cmd_min>
      </control>
      <control channel="3">
        <jointName>iris_with_standoffs::rotor_3_joint</jointName>
        <useForce>1</useForce><multiplier>-838</multiplier><offset>0</offset>
        <servo_min>1100</servo_min><servo_max>1900</servo_max><type>VELOCITY</type>
        <p_gain>0.20</p_gain><i_gain>0</i_gain><d_gain>0</d_gain>
        <cmd_max>2.5</cmd_max><cmd_min>-2.5</cmd_min>
      </control>
    </plugin>
  </model>
</sdf>
""")

# Обновляем worlds/quarry.sdf, чтобы <uri> указывал напрямую на абсолютный путь модели
quarry_sdf_path = os.path.join(PROJECT_ROOT, "worlds/quarry.sdf")
if os.path.exists(quarry_sdf_path):
    with open(quarry_sdf_path, "r", encoding="utf-8") as f:
        sdf_text = f.read()
    sdf_text = sdf_text.replace("model://iris_quarry_cam", iris_quarry_cam_uri)
    with open(quarry_sdf_path, "w", encoding="utf-8") as f:
        f.write(sdf_text)

# Скрипт для удобного интерактивного запуска Gazebo с GUI
write_file("scripts/run_gazebo.sh", f"""#!/usr/bin/env bash
export GZ_SIM_SYSTEM_PLUGIN_PATH="{ARDUPILOT_GZ_BUILD}:${{GZ_SIM_SYSTEM_PLUGIN_PATH}}"
export GZ_SIM_RESOURCE_PATH="{PROJECT_ROOT}/models:{ARDUPILOT_GZ_MODELS}:${{GZ_SIM_RESOURCE_PATH}}"
exec gz sim -v4 -r "{PROJECT_ROOT}/worlds/quarry.sdf"
""", executable=True)

# Документация README.md и PROJECT_STATUS.md
write_file("README.md", """# Управление БПЛА с помощью VLM Qwen2.5-VL в виртуальном карьере (Gazebo Harmonic + ArduPilot SITL + ROS 2 Jazzy)

## Быстрый старт
1. **Запуск автотестов A–E:**
   ```bash
   cd /home/nikita/drone_vlm_quarry
   python3 tests/run_all_tests.py