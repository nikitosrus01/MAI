# -*- coding: utf-8 -*-
"""
Уровень 2: Проверка планов VLM и построение безопасных маршрутов в виртуальном карьере.
Выполняет:
- проверку допустимости действий из расширенного белого списка (включая повороты по yaw, смену высоты, ожидание и др.);
- проверку параметров высоты, рабочей зоны, скорости;
- защиту от бесконечного повторения одних и тех же команд;
- интеллектуальное сопоставление ориентиров и их синонимов с картой симуляции;
- проверку отрезков маршрута на пересечение с препятствиями и генерацию путевых точек облёта/пролёта сверху.
"""

import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


ALLOWED_ACTIONS = {
    "takeoff",
    "goto",
    "fly_to",
    "navigate",
    "approach",
    "move_relative",
    "move",
    "inspect",
    "scan",
    "search",
    "observe",
    "rotate",
    "set_yaw",
    "yaw",
    "turn",
    "look_at",
    "spin",
    "orbit",
    "hover",
    "wait",
    "pause",
    "change_altitude",
    "ascend",
    "descend",
    "climb",
    "return_home",
    "rtl",
    "land",
    "abort",
    "stop",
}

SECTOR_YAW_DEG = {
    "north": 0.0,
    "север": 0.0,
    "east": 90.0,
    "восток": 90.0,
    "south": 180.0,
    "юг": 180.0,
    "west": -90.0,
    "запад": -90.0,
}


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
    """Модуль валидации планов и геометрического планирования маршрутов в карьере."""

    def __init__(self, config_dir: Optional[str] = None):
        if config_dir is None:
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_dir = os.path.join(base, "config")
        self.config_dir = config_dir
        self.quarry_map = self._load_quarry_map()
        self.limits = {
            "x_min": -50.0,
            "x_max": 50.0,
            "y_min": -50.0,
            "y_max": 50.0,
            "z_min": 1.0,
            "z_max": 25.0,
            "max_speed_mps": 4.0,
            "min_obstacle_clearance_m": 1.5,
            "overfly_clearance_m": 2.5,
            "max_plan_steps": 12,
            "max_consecutive_identical_actions": 3,
        }

    def _load_quarry_map(self) -> Dict[str, Any]:
        map_path = os.path.join(self.config_dir, "quarry_map.json")
        if os.path.exists(map_path):
            with open(map_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"landmarks": {}, "static_obstacles": []}

    @staticmethod
    def _normalize_ru(text: str) -> str:
        t = text.strip().lower().replace("ё", "е")
        t = re.sub(r"[^a-zа-я0-9_\s-]", " ", t)
        return re.sub(r"\s+", " ", t).strip()

    def resolve_landmark(self, target_query: str) -> Optional[Dict[str, Any]]:
        """
        Ищет объект в карте виртуального мира по ID, русскоязычному названию, синонимам или ключевым корням слов.
        Понимает вариации: «оранжевый цилиндр» == «оранжевая метка» == «маркер на юге» и т.д.
        Никогда не выдумывает координаты, если объект совершенно неизвестен.
        """
        if not target_query:
            return None
        q = self._normalize_ru(str(target_query))
        landmarks = self.quarry_map.get("landmarks", {})
        if q in landmarks:
            return landmarks[q]

        # 1. Прямое совпадение по имени или списку aliases
        for lm_id, lm_data in landmarks.items():
            if q == self._normalize_ru(lm_data.get("name_ru", "")):
                return lm_data
            for alias in lm_data.get("aliases", []):
                norm_alias = self._normalize_ru(alias)
                if norm_alias and (norm_alias in q or q in norm_alias):
                    return lm_data

        # 2. Интеллектуальное сопоставление по корням слов (морфология русского языка)
        if any(stem in q for stem in ("оранж", "рыж", "маркер", "метк", "цилиндр", "конус", "пилон", "маяк", "бочк", "тумб", "южн", "на юге", "beacon")):
            if "south_marker_beacon" in landmarks:
                return landmarks["south_marker_beacon"]

        if any(stem in q for stem in ("групп", "скоплен", "три камн", "три валун", "восточн", "на востоке", "красн", "коричнев", "кластер")):
            if "east_boulder_cluster" in landmarks:
                return landmarks["east_boulder_cluster"]

        if any(stem in q for stem in ("уступ", "выступ", "западн", "на западе", "охр", "желт", "песчан", "скал")):
            if "west_rock_outcrop" in landmarks:
                return landmarks["west_rock_outcrop"]

        if any(stem in q for stem in ("крупн", "больш", "северн", "на севере", "серый", "гранит", "главн", "камен", "валун", "boulder")):
            if "north_large_boulder" in landmarks:
                return landmarks["north_large_boulder"]

        return None

    def check_within_bounds(self, x: float, y: float, z: float) -> Tuple[bool, Optional[str]]:
        if not (self.limits["x_min"] <= x <= self.limits["x_max"]):
            return False, f"Координата X={x:.1f}м выходит за пределы рабочей зоны [{self.limits['x_min']}, {self.limits['x_max']}]м."
        if not (self.limits["y_min"] <= y <= self.limits["y_max"]):
            return False, f"Координата Y={y:.1f}м выходит за пределы рабочей зоны [{self.limits['y_min']}, {self.limits['y_max']}]м."
        if not (self.limits["z_min"] <= z <= self.limits["z_max"]):
            return False, f"Высота Z={z:.1f}м выходит за безопасный диапазон [{self.limits['z_min']}, {self.limits['z_max']}]м."
        return True, None

    @staticmethod
    def _segment_circle_distance_2d(
        x1: float, y1: float, x2: float, y2: float, cx: float, cy: float
    ) -> float:
        dx = x2 - x1
        dy = y2 - y1
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq < 1e-9:
            return math.hypot(cx - x1, cy - y1)
        t = max(0.0, min(1.0, ((cx - x1) * dx + (cy - y1) * dy) / seg_len_sq))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        return math.hypot(cx - proj_x, cy - proj_y)

    def build_safe_segment(
        self,
        start_xyz: Tuple[float, float, float],
        goal_xyz: Tuple[float, float, float],
    ) -> Tuple[List[Tuple[float, float, float]], List[str]]:
        """
        Проверяет прямой отрезок маршрута на пересечение со статическими препятствиями карьера.
        Если высота полёта ниже верхушки препятствия + запас безопасности, автоматически поднимает
        маршрут на безопасный эшелон пролёта над препятствием.
        """
        warnings: List[str] = []
        x1, y1, z1 = start_xyz
        x2, y2, z2 = goal_xyz
        min_safe_z = max(z1, z2)

        obstacles = self.quarry_map.get("static_obstacles", [])
        clearance = self.limits["min_obstacle_clearance_m"]

        for obs in obstacles:
            ox = float(obs["x"])
            oy = float(obs["y"])
            orad = float(obs.get("radius", 2.0))
            otop = float(obs.get("top_z", 2.0))

            dist_2d = self._segment_circle_distance_2d(x1, y1, x2, y2, ox, oy)
            if dist_2d < (orad + clearance):
                required_z = otop + self.limits["overfly_clearance_m"]
                if min(z1, z2) < required_z:
                    min_safe_z = max(min_safe_z, required_z)
                    warnings.append(
                        f"Маршрут проходит вблизи препятствия «{obs['name']}» (высота {otop}м). "
                        f"Безопасная высота эшелона скорректирована до {min_safe_z:.1f}м."
                    )

        waypoints: List[Tuple[float, float, float]] = []
        if min_safe_z > z1 + 0.3:
            waypoints.append((x1, y1, min_safe_z))
        if min_safe_z > z2 + 0.3:
            waypoints.append((x2, y2, min_safe_z))
        waypoints.append((x2, y2, max(z2, min_safe_z)))
        return waypoints, warnings

    def validate_plan(
        self,
        raw_steps: List[Dict[str, Any]],
        drone_state: Dict[str, Any],
    ) -> ValidationReport:
        """
        Комплексная проверка плана перед отправкой в исполнительный контроллер ArduPilot SITL.
        """
        report = ValidationReport(is_valid=True)

        if not raw_steps:
            report.is_valid = False
            report.errors.append("План не содержит ни одного шага (пустой список steps).")
            return report

        if len(raw_steps) > self.limits["max_plan_steps"]:
            report.is_valid = False
            report.errors.append(
                f"Слишком длинный план ({len(raw_steps)} шагов > макс. {self.limits['max_plan_steps']})."
            )
            return report

        if not drone_state.get("connected", True):
            report.is_valid = False
            report.errors.append("Отсутствует связь с автопилотом ArduPilot SITL.")
            return report

        cur_x = float(drone_state.get("x", 0.0))
        cur_y = float(drone_state.get("y", 0.0))
        cur_z = float(drone_state.get("z", 0.0))
        is_airborne = cur_z > 0.8 or bool(drone_state.get("armed", False))

        consecutive_count = 0
        last_signature = None

        for idx, step in enumerate(raw_steps):
            if not isinstance(step, dict):
                report.is_valid = False
                report.errors.append(f"Шаг #{idx + 1} не является словарём.")
                continue

            action = str(step.get("action", "")).strip().lower()
            params = step.get("params", {}) or {}
            if not isinstance(params, dict):
                params = {}
            desc = str(step.get("description", f"Шаг {action}"))

            if action not in ALLOWED_ACTIONS:
                report.is_valid = False
                report.errors.append(
                    f"Шаг #{idx + 1}: недопустимое действие «{action}». Разрешены только: {sorted(ALLOWED_ACTIONS)}."
                )
                continue

            # Защита от бесконечного зацикливания одинаковых команд
            sig = f"{action}:{json.dumps(params, sort_keys=True, ensure_ascii=False)}"
            if sig == last_signature:
                consecutive_count += 1
                if consecutive_count >= self.limits["max_consecutive_identical_actions"]:
                    report.is_valid = False
                    report.errors.append(
                        f"Обнаружено зацикливание (повтор команды {action} более {self.limits['max_consecutive_identical_actions']} раз подряд)."
                    )
                    continue
            else:
                consecutive_count = 1
                last_signature = sig

            if action == "takeoff":
                alt = float(params.get("alt", params.get("z", step.get("alt", 5.0))))
                ok, err = self.check_within_bounds(cur_x, cur_y, alt)
                if not ok:
                    report.is_valid = False
                    report.errors.append(f"Шаг #{idx + 1} (takeoff): {err}")
                    continue
                cur_z = alt
                is_airborne = True
                report.validated_steps.append(
                    ValidatedStep(
                        action="takeoff",
                        params={"alt": alt},
                        description=desc,
                        waypoints=[(cur_x, cur_y, cur_z)],
                        localization_source="baro_gps_sitl",
                    )
                )

            elif action in ("rotate", "set_yaw", "yaw", "turn", "look_at", "spin", "orbit"):
                if not is_airborne:
                    # Автоматический взлёт на безопасную высоту, если VLM забыла поставить takeoff перед поворотом
                    default_alt = 5.0
                    cur_z = default_alt
                    is_airborne = True
                    report.warnings.append(
                        f"Шаг #{idx + 1} ({action}): добавлен автоматический взлёт на {default_alt}м перед выполнением разворота."
                    )
                    report.validated_steps.append(
                        ValidatedStep(
                            action="takeoff",
                            params={"alt": default_alt},
                            description=f"Автоматический взлёт на {default_alt}м перед поворотом",
                            waypoints=[(cur_x, cur_y, cur_z)],
                            localization_source="auto_takeoff_safety",
                        )
                    )

                raw_yaw = params.get("yaw_deg", params.get("angle", params.get("yaw", step.get("yaw_deg", step.get("angle")))))
                relative = bool(params.get("relative", step.get("relative", False)))
                target_ref = params.get("target", params.get("target_id", step.get("target")))
                sector_ref = str(params.get("sector", step.get("sector", ""))).strip().lower()

                yaw_deg = 0.0
                if raw_yaw is not None:
                    yaw_deg = float(raw_yaw)
                elif target_ref:
                    lm = self.resolve_landmark(str(target_ref))
                    if lm is not None:
                        dx = float(lm["x"]) - cur_x
                        dy_east = -(float(lm["y"]) - cur_y)
                        yaw_deg = math.degrees(math.atan2(dy_east, dx)) % 360.0
                        relative = False
                elif sector_ref in SECTOR_YAW_DEG:
                    yaw_deg = SECTOR_YAW_DEG[sector_ref]
                    relative = False
                elif action == "spin" or action == "orbit":
                    yaw_deg = 360.0
                    relative = True

                report.validated_steps.append(
                    ValidatedStep(
                        action="rotate",
                        params={"yaw_deg": yaw_deg, "relative": relative},
                        description=desc,
                        waypoints=[],
                        localization_source="compass_attitude_sitl",
                    )
                )

            elif action in ("inspect", "scan", "search", "observe"):
                if not is_airborne:
                    report.warnings.append(
                        f"Шаг #{idx + 1} (inspect): выполняется до взлёта, обзор камеры может быть ограничен."
                    )
                raw_target = str(params.get("target", params.get("target_id", step.get("target", "north_large_boulder"))))
                lm = self.resolve_landmark(raw_target)
                resolved_target = lm["id"] if lm else raw_target
                sector = str(params.get("sector", lm.get("sector", "north") if lm else "north"))
                report.requires_visual_confirmation = True
                report.validated_steps.append(
                    ValidatedStep(
                        action="inspect",
                        params={"target": resolved_target, "raw_query": raw_target, "sector": sector},
                        description=desc,
                        waypoints=[],
                        localization_source="vlm_camera_verification",
                    )
                )

            elif action in ("goto", "fly_to", "navigate", "approach"):
                if not is_airborne:
                    report.is_valid = False
                    report.errors.append(
                        f"Шаг #{idx + 1} (goto): попытка горизонтального полёта без предварительного взлёта (takeoff)."
                    )
                    continue

                target_id = params.get("target_id", params.get("target", step.get("target_id", step.get("target"))))
                overfly = bool(params.get("overfly", True))
                clearance_m = float(params.get("clearance_m", self.limits["overfly_clearance_m"]))

                if target_id:
                    lm = self.resolve_landmark(str(target_id))
                    if lm is None:
                        report.is_valid = False
                        report.errors.append(
                            f"Шаг #{idx + 1} (goto): координаты объекта «{target_id}» неизвестны. Выдумывание координат запрещено."
                        )
                        continue
                    tx = float(lm["x"])
                    ty = float(lm["y"])
                    top_z = float(lm.get("top_z", lm["z"]))
                    tz = max(cur_z, top_z + clearance_m) if overfly else max(cur_z, float(lm["z"]) + 2.0)
                    loc_src = f"simulation_map_landmark:{lm['id']} (visual_confirmation_required={report.requires_visual_confirmation})"
                elif "x" in params and "y" in params:
                    tx = float(params["x"])
                    ty = float(params["y"])
                    tz = float(params.get("z", cur_z))
                    loc_src = "explicit_coordinates"
                else:
                    report.is_valid = False
                    report.errors.append(
                        f"Шаг #{idx + 1} (goto): не указан ни известный target_id, ни явные координаты (x, y)."
                    )
                    continue

                ok, err = self.check_within_bounds(tx, ty, tz)
                if not ok:
                    report.is_valid = False
                    report.errors.append(f"Шаг #{idx + 1} (goto): {err}")
                    continue

                wps, seg_warnings = self.build_safe_segment((cur_x, cur_y, cur_z), (tx, ty, tz))
                report.warnings.extend(seg_warnings)
                cur_x, cur_y, cur_z = wps[-1]

                report.validated_steps.append(
                    ValidatedStep(
                        action="goto",
                        params={"x": cur_x, "y": cur_y, "z": cur_z, "target_id": lm["id"] if target_id and lm else target_id, "overfly": overfly},
                        description=desc,
                        waypoints=wps,
                        localization_source=loc_src,
                    )
                )

            elif action in ("move_relative", "move", "change_altitude", "ascend", "descend", "climb"):
                if not is_airborne:
                    report.is_valid = False
                    report.errors.append(f"Шаг #{idx + 1} ({action}): дрон ещё не взлетел.")
                    continue
                dx = float(params.get("dx", params.get("x", 0.0)))
                dy = float(params.get("dy", params.get("y", 0.0)))
                if "alt" in params or "z" in params:
                    target_z = float(params.get("alt", params.get("z", cur_z)))
                    dz = target_z - cur_z
                else:
                    dz = float(params.get("dz", 0.0))
                    if action == "descend" and dz > 0:
                        dz = -dz
                tx, ty, tz = cur_x + dx, cur_y + dy, cur_z + dz
                ok, err = self.check_within_bounds(tx, ty, tz)
                if not ok:
                    report.is_valid = False
                    report.errors.append(f"Шаг #{idx + 1} ({action}): {err}")
                    continue
                wps, seg_warnings = self.build_safe_segment((cur_x, cur_y, cur_z), (tx, ty, tz))
                report.warnings.extend(seg_warnings)
                cur_x, cur_y, cur_z = wps[-1]
                report.validated_steps.append(
                    ValidatedStep(
                        action="move_relative",
                        params={"dx": dx, "dy": dy, "dz": dz, "x": cur_x, "y": cur_y, "z": cur_z},
                        description=desc,
                        waypoints=wps,
                        localization_source="relative_odometry",
                    )
                )

            elif action in ("return_home", "rtl"):
                home = self.quarry_map.get("home_position", {"x": 0.0, "y": 0.0, "z": 0.2})
                hx, hy = float(home["x"]), float(home["y"])
                safe_return_z = max(cur_z, 5.0)
                wps, seg_warnings = self.build_safe_segment((cur_x, cur_y, safe_return_z), (hx, hy, safe_return_z))
                report.warnings.extend(seg_warnings)
                cur_x, cur_y, cur_z = hx, hy, safe_return_z
                report.validated_steps.append(
                    ValidatedStep(
                        action="return_home",
                        params={"x": hx, "y": hy, "z": safe_return_z},
                        description=desc,
                        waypoints=wps,
                        localization_source="home_position_sitl",
                    )
                )

            elif action == "land":
                cur_z = 0.0
                is_airborne = False
                report.validated_steps.append(
                    ValidatedStep(
                        action="land",
                        params={},
                        description=desc,
                        waypoints=[(cur_x, cur_y, 0.0)],
                        localization_source="baro_sitl",
                    )
                )

            elif action in ("pause", "hover", "wait"):
                dur = float(params.get("duration_sec", params.get("duration", params.get("sec", 3.0))))
                if dur < 0 or dur > 60.0:
                    report.is_valid = False
                    report.errors.append(f"Шаг #{idx + 1} (pause): недопустимая длительность паузы {dur}с.")
                    continue
                report.validated_steps.append(
                    ValidatedStep(action="pause", params={"duration_sec": dur}, description=desc)
                )

            elif action in ("abort", "stop"):
                report.validated_steps.append(
                    ValidatedStep(
                        action="abort",
                        params={"reason": str(params.get("reason", "Остановка по плану"))},
                        description=desc,
                    )
                )

        return report