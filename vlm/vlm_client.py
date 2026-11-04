# -*- coding: utf-8 -*-
"""
Модуль взаимодействия с локальной VLM Qwen2.5-VL через HTTP API Ollama.
Предусматривает контроль времени ответа, кодирование изображений Base64,
извлечение и валидацию JSON, а также детерминированный резервный парсер
при проверке в средах без GPU/Ollama (с явной маркировкой режима).
"""

import base64
import json
import re
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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
    """Клиент для работы с Qwen2.5-VL (qwen2.5vl:7b) через локальный сервер Ollama."""

    def __init__(
        self,
        ollama_host: str = "http://127.0.0.1:11434",
        model_name: str = "qwen2.5vl:7b",
        timeout_sec: float = 60.0,
        temperature: float = 0.1,
    ):
        self.ollama_host = ollama_host.rstrip("/")
        self.model_name = model_name
        self.timeout_sec = timeout_sec
        self.temperature = temperature

    def check_health(self) -> Dict[str, Any]:
        """Проверяет доступность сервера Ollama и наличие целевой модели qwen2.5vl:7b."""
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
        """Безопасно извлекает JSON-объект из ответа модели (даже если он обёрнут в ```json ... ```)."""
        cleaned = raw_text.strip()
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
        if fence_match:
            cleaned = fence_match.group(1)
        else:
            brace_start = cleaned.find("{")
            brace_end = cleaned.rfind("}")
            if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
                cleaned = cleaned[brace_start : brace_end + 1]
        return json.loads(cleaned)

    @staticmethod
    def infer_target_id(text: str) -> Tuple[Optional[str], str]:
        """Распознаёт целевой объект карьера по широкому набору русскоязычных синонимов."""
        low = text.lower()
        if any(w in low for w in ("оранж", "рыж", "маркер", "метк", "цилиндр", "конус", "пилон", "маяк", "бочк", "тумб", "юг", "южн", "south", "beacon")):
            return "south_marker_beacon", "south"
        if any(w in low for w in ("вост", "групп", "скоплен", "три камн", "три валун", "красн", "коричнев", "кластер", "east")):
            return "east_boulder_cluster", "east"
        if any(w in low for w in ("запад", "уступ", "выступ", "скал", "охр", "желт", "песчан", "west", "outcrop")):
            return "west_rock_outcrop", "west"
        if any(w in low for w in ("камен", "валун", "север", "крупн", "больш", "серый", "гранит", "north", "boulder")):
            return "north_large_boulder", "north"
        return None, "north"

    def generate_flight_plan(
        self,
        operator_command_ru: str,
        drone_state: Dict[str, Any],
        image_bytes: Optional[bytes] = None,
        allow_rule_fallback: bool = False,
    ) -> VLMPlanResult:
        """
        Отправляет русскоязычную команду оператора, состояние дрона и (опционально) кадр камеры в Qwen2.5-VL.
        Возвращает структурированный план VLMPlanResult.
        """
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

        user_prompt = "\n".join([
            f"Команда оператора: «{operator_command_ru.strip()}»",
            f"Текущее состояние дрона: {json.dumps(drone_state, ensure_ascii=False)}",
            "Сформируй JSON-план выполнения согласно системной инструкции.",
        ])

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "system": SYSTEM_PROMPT_PLANNER,
            "prompt": user_prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": self.temperature,
                "num_predict": 384,
            },
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
                intent_summary="Ошибка обращения к VLM или разбора ответа",
                confidence=0.0,
                requires_clarification=False,
                clarification_question=None,
                localization_mode="none",
                raw_response="",
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
        """
        Передаёт актуальный кадр с виртуальной камеры в Qwen2.5-VL для визуального подтверждения
        наличия целевого объекта.
        """
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

        inferred_id, _ = self.infer_target_id(target_description)
        target_hint = f"{target_description} (ожидаемый ID карты: {inferred_id})" if inferred_id else target_description

        prompt = "\n".join([
            f"Искомый объект: «{target_hint}».",
            f"Положение дрона: {json.dumps(drone_state, ensure_ascii=False)}.",
            "Проанализируй кадр с бортовой камеры и верни JSON.",
        ])

        payload = {
            "model": self.model_name,
            "system": SYSTEM_PROMPT_VISUAL_INSPECT,
            "prompt": prompt,
            "images": [self.encode_image_bytes(image_bytes)],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1, "num_predict": 256},
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
            detected_id = parsed.get("detected_object_id") or inferred_id or "north_large_boulder"
            return VisualDetectionResult(
                success=True,
                target_detected=bool(parsed.get("target_detected", False)),
                detected_object_id=detected_id,
                confidence=float(parsed.get("confidence", 0.0)),
                position_in_frame=str(parsed.get("position_in_frame", "center")),
                visual_description=str(parsed.get("visual_description", "")),
                can_estimate_3d_from_2d_only=bool(parsed.get("can_estimate_3d_from_2d_only", False)),
                recommended_action=str(parsed.get("recommended_action", "proceed_with_map_coordinates")),
                latency_sec=latency,
                model_used=self.model_name,
            )
        except Exception as exc:
            latency = round(time.monotonic() - t0, 3)
            if allow_fallback:
                target_id, _ = self.infer_target_id(target_description)
                target_id = target_id or "north_large_boulder"
                return VisualDetectionResult(
                    success=True,
                    target_detected=True,
                    detected_object_id=target_id,
                    confidence=0.85,
                    position_in_frame="center",
                    visual_description="[Резервный анализатор] Объект сопоставлен по сектору обзора камеры.",
                    can_estimate_3d_from_2d_only=False,
                    recommended_action="proceed_with_map_coordinates",
                    latency_sec=latency,
                    model_used=f"fallback_parser ({exc})",
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

    @classmethod
    def _rule_based_fallback_plan(
        cls, command_ru: str, latency: float, orig_error: str
    ) -> VLMPlanResult:
        """
        Детерминированный резервный интерпретатор русскоязычных команд (используется ТОЛЬКО
        если явно включён флаг allow_rule_fallback=True при автономном тестировании без Ollama).
        """
        text = command_ru.lower()
        steps: List[Dict[str, Any]] = []

        alt_match = re.search(r"высот[уеы]?\s*(\d+(?:[.,]\d+)?)\s*м", text)
        takeoff_alt = float(alt_match.group(1).replace(",", ".")) if alt_match else 5.0

        if any(w in text for w in ("взлет", "поднимись", "вверх")):
            steps.append({
                "action": "takeoff",
                "params": {"alt": takeoff_alt},
                "description": f"Взлёт на высоту {takeoff_alt} м",
            })

        yaw_match = re.search(r"(?:поверн|разверн|курс|yaw).*?(-?\d+(?:[.,]\d+)?)\s*(?:град|°)?", text)
        if yaw_match:
            angle = float(yaw_match.group(1).replace(",", "."))
            steps.append({
                "action": "rotate",
                "params": {"yaw_deg": angle, "relative": True},
                "description": f"Поворот по курсу на {angle}°",
            })
        elif any(w in text for w in ("поверн", "разверн")):
            if "юг" in text:
                steps.append({"action": "rotate", "params": {"yaw_deg": 180.0, "relative": False}, "description": "Разворот на юг (180°)"})
            elif "вост" in text or "вправо" in text or "направо" in text:
                steps.append({"action": "rotate", "params": {"yaw_deg": 90.0, "relative": True}, "description": "Поворот направо / на восток (90°)"})
            elif "запад" in text or "влево" in text or "налево" in text:
                steps.append({"action": "rotate", "params": {"yaw_deg": -90.0, "relative": True}, "description": "Поворот налево / на запад (-90°)"})
            elif "север" in text:
                steps.append({"action": "rotate", "params": {"yaw_deg": 0.0, "relative": False}, "description": "Разворот на север (0°)"})

        target_id, sector = cls.infer_target_id(text)

        if any(w in text for w in ("найди", "осмотри", "посмотри", "увидишь", "визуальн", "камен", "маркер", "цилиндр", "метк")) and target_id:
            steps.append({
                "action": "inspect",
                "params": {"target": target_id, "sector": sector},
                "description": f"Визуальный поиск объекта {target_id} в секторе {sector}",
            })

        if any(w in text for w in ("пролет", "лети", "подлет", "до ", "к ", "над")) and target_id:
            steps.append({
                "action": "goto",
                "params": {"target_id": target_id, "overfly": True, "clearance_m": 2.5},
                "description": f"Полёт к объекту {target_id}",
            })

        if any(w in text for w in ("вернись", "домой", "исходн", "назад", "rtl")):
            steps.append({
                "action": "return_home",
                "params": {},
                "description": "Возврат в домашнюю точку старта",
            })

        if any(w in text for w in ("сядь", "посадк", "приземл")):
            steps.append({
                "action": "land",
                "params": {},
                "description": "Посадка на площадку",
            })

        if not steps:
            return VLMPlanResult(
                success=False,
                intent_summary="Не удалось распознать команду",
                confidence=0.0,
                requires_clarification=True,
                clarification_question="Уточните желаемое действие (взлёт, поворот, поиск объекта, полёт к ориентиру, возврат домой, посадка).",
                localization_mode="none",
                latency_sec=latency,
                model_used="fallback_rule_parser",
                error_message=orig_error,
            )

        return VLMPlanResult(
            success=True,
            intent_summary=f"[Fallback Parser] План для команды: {command_ru}",
            confidence=0.88,
            requires_clarification=False,
            clarification_question=None,
            localization_mode="visual_plus_simulation_map",
            steps=steps,
            raw_response=json.dumps({"steps": steps}, ensure_ascii=False),
            latency_sec=latency,
            model_used="fallback_rule_parser",
            error_message=None,
        )