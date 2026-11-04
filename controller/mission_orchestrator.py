# -*- coding: utf-8 -*-
"""
Сквозной оркестратор миссии управления квадрокоптером в виртуальном карьере.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from vlm.vlm_client import OllamaVLMClient, VLMPlanResult, VisualDetectionResult
from planner.plan_validator import PlanValidator, ValidationReport, ValidatedStep
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
    """Координатор выполнения русскоязычных полётных заданий."""

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

    def _get_fresh_frame(self, wait_sec: float = 4.0) -> Tuple[Optional[FramePacket], Optional[str]]:
        if self.frame_provider is None:
            return None, "Поставщик кадров виртуальной камеры не подключён."

        t0 = time.monotonic()
        pkt = self.frame_provider()
        while (pkt is None or not pkt.image_bytes) and (time.monotonic() - t0) < wait_sec:
            time.sleep(0.2)
            pkt = self.frame_provider()

        if pkt is None or not pkt.image_bytes:
            return None, "Кадр с виртуальной камеры ещё не поступил."
        age = time.monotonic() - pkt.timestamp
        if age > self.image_freshness_timeout_sec:
            return (
                None,
                f"Изображение с виртуальной камеры устарело (возраст кадра {age:.1f}с > лимит {self.image_freshness_timeout_sec:.1f}с).",
            )
        return pkt, None

    def execute_russian_command(self, command_ru: str) -> MissionExecutionReport:
        t_start = time.monotonic()
        report = MissionExecutionReport(command_ru=command_ru, success=False)

        telemetry = self.controller.poll_telemetry()
        if not telemetry.connected:
            report.abort_reason = "Нет связи с автопилотом ArduPilot SITL."
            self._emit(f"[ОШИБКА] {report.abort_reason}")
            return report

        self._emit(f"[УРОВЕНЬ 1: VLM] Отправка команды в Qwen2.5-VL: «{command_ru}»...")
        vlm_plan = self.vlm.generate_flight_plan(
            operator_command_ru=command_ru,
            drone_state=telemetry.to_dict(),
            image_bytes=None,
            allow_rule_fallback=self.allow_vlm_fallback,
        )
        report.vlm_plan = vlm_plan

        if not vlm_plan.success:
            report.abort_reason = f"VLM не смогла сформировать план: {vlm_plan.error_message}"
            self._emit(f"[ОШИБКА VLM] {report.abort_reason}")
            return report

        if vlm_plan.requires_clarification:
            report.abort_reason = f"Требуется уточнение оператора: {vlm_plan.clarification_question}"
            self._emit(f"[ЗАПРОС УТОЧНЕНИЯ] {report.abort_reason}")
            return report

        self._emit(
            f"[УРОВЕНЬ 1: VLM] Получен план ({len(vlm_plan.steps)} шагов, модель: {vlm_plan.model_used}, "
            f"время: {vlm_plan.latency_sec}с): {vlm_plan.intent_summary}"
        )

        self._emit("[УРОВЕНЬ 2: ПЛАНИРОВЩИК] Проверка ограничений рабочей зоны, высот и препятствий...")
        validation = self.validator.validate_plan(vlm_plan.steps, telemetry.to_dict())
        report.validation = validation

        for warn in validation.warnings:
            self._emit(f"[ПРЕДУПРЕЖДЕНИЕ ПЛАНИРОВЩИКА] {warn}")

        if not validation.is_valid:
            report.abort_reason = "План отклонён модулем безопасности: " + "; ".join(validation.errors)
            self._emit(f"[БЛОКИРОВКА ПЛАНА] {report.abort_reason}")
            return report

        self._emit(f"[УРОВЕНЬ 2: ПЛАНИРОВЩИК] План одобрен ({len(validation.validated_steps)} проверенных шагов).")

        has_subsequent_goto = any(s.action == "goto" for s in validation.validated_steps)

        for idx, step in enumerate(validation.validated_steps, start=1):
            self._emit(f"[УРОВЕНЬ 3: ИСПОЛНЕНИЕ] Шаг {idx}/{len(validation.validated_steps)}: {step.description} ({step.action})")

            if step.action == "takeoff":
                alt = float(step.params["alt"])
                ok, msg = self.controller.arm_and_takeoff(target_alt_m=alt)
                report.executed_steps.append(f"takeoff({alt}m): {msg}")
                self._emit(f"  -> {msg}")
                if not ok:
                    report.abort_reason = msg
                    return self._finalize(report, t_start)

            elif step.action == "rotate":
                yaw_deg = float(step.params.get("yaw_deg", 0.0))
                relative = bool(step.params.get("relative", False))
                ok, msg = self.controller.rotate_yaw(yaw_deg=yaw_deg, relative=relative)
                report.executed_steps.append(f"rotate({yaw_deg}°, rel={relative}): {msg}")
                self._emit(f"  -> {msg}")
                if not ok:
                    report.abort_reason = msg
                    return self._finalize(report, t_start)

            elif step.action == "inspect":
                target_query = str(step.params.get("target", "north_large_boulder"))
                frame_pkt, frame_err = self._get_fresh_frame(wait_sec=3.0)
                if frame_err is not None:
                    report.abort_reason = f"Невозможно выполнить визуальный поиск: {frame_err}"
                    self._emit(f"  -> [ОШИБКА КАМЕРЫ] {report.abort_reason}")
                    return self._finalize(report, t_start)

                self._emit(f"  -> Анализ актуального кадра ({frame_pkt.width}x{frame_pkt.height}) в Qwen2.5-VL для поиска «{target_query}»...")
                det = self.vlm.inspect_image(
                    target_description=target_query,
                    image_bytes=frame_pkt.image_bytes,
                    drone_state=self.controller.poll_telemetry().to_dict(),
                    allow_fallback=self.allow_vlm_fallback,
                )
                report.visual_detections.append(det)

                if not det.success or not det.target_detected:
                    # Если объект далеко и впереди есть шаг подлёта goto — продолжаем подлёт к ориентиру по карте
                    if has_subsequent_goto and self.validator.resolve_landmark(target_query) is not None:
                        warn_msg = (
                            f"С текущей дистанции объект «{target_query}» ещё далеко или вне центра обзора "
                            f"({det.visual_description or 'дальний план'}). Выполняется подлёт к сектору ориентира по карте."
                        )
                        self._emit(f"  -> [ПОДЛЁТ К СЕКТОРУ] {warn_msg}")
                        report.executed_steps.append(f"inspect({target_query}): {warn_msg}")
                        continue

                    report.abort_reason = (
                        f"Целевой объект «{target_query}» не подтверждён визуально на кадре с камеры "
                        f"({det.visual_description or det.error_message}). Выполнение остановлено для безопасности."
                    )
                    self._emit(f"  -> [ОБЪЕКТ НЕ ОБНАРУЖЕН] {report.abort_reason}")
                    return self._finalize(report, t_start)

                info_msg = (
                    f"Визуально подтверждён объект «{det.detected_object_id}» (уверенность {det.confidence:.2f}, "
                    f"положение в кадре: {det.position_in_frame})."
                )
                report.executed_steps.append(f"inspect({target_query}): {info_msg}")
                self._emit(f"  -> {info_msg}")

            elif step.action in ("goto", "move_relative"):
                for wp_idx, wp in enumerate(step.waypoints, start=1):
                    self._emit(
                        f"  -> Следование к путевой точке {wp_idx}/{len(step.waypoints)}: "
                        f"X={wp[0]:.1f}м, Y={wp[1]:.1f}м, Z={wp[2]:.1f}м [источник: {step.localization_source}]"
                    )
                    ok, msg = self.controller.goto_local_xyz(wp[0], wp[1], wp[2])
                    if not ok:
                        report.abort_reason = msg
                        self._emit(f"  -> [ОШИБКА ДВИЖЕНИЯ] {msg}")
                        return self._finalize(report, t_start)
                report.executed_steps.append(
                    f"{step.action}: достигнута цель ({step.waypoints[-1][0]:.1f}, {step.waypoints[-1][1]:.1f}, {step.waypoints[-1][2]:.1f})м"
                )

            elif step.action == "return_home":
                ok, msg = self.controller.safe_return_home(step.waypoints)
                report.executed_steps.append(f"return_home: {msg}")
                self._emit(f"  -> {msg}")
                if not ok:
                    report.abort_reason = msg
                    return self._finalize(report, t_start)

            elif step.action == "land":
                ok, msg = self.controller.land()
                report.executed_steps.append(f"land: {msg}")
                self._emit(f"  -> {msg}")
                if not ok:
                    report.abort_reason = msg
                    return self._finalize(report, t_start)

            elif step.action == "pause":
                dur = float(step.params.get("duration_sec", 2.0))
                time.sleep(min(dur, 5.0))
                report.executed_steps.append(f"pause({dur}s): выполнено")

            elif step.action == "abort":
                self.controller.request_abort()
                report.abort_reason = f"Миссия остановлена по команде abort: {step.params.get('reason', '')}"
                self._emit(f"  -> {report.abort_reason}")
                return self._finalize(report, t_start)

        report.success = True
        self._emit("[УСПЕХ] Все этапы полётного задания успешно выполнены.")
        return self._finalize(report, t_start)

    def _finalize(self, report: MissionExecutionReport, t_start: float) -> MissionExecutionReport:
        report.final_telemetry = self.controller.poll_telemetry().to_dict()
        report.total_duration_sec = round(time.monotonic() - t_start, 3)
        return report