from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import numpy as np

from .se3 import invert_transform, is_valid_rotation, make_transform


@dataclass
class CameraCalibration:
    camera_id: str
    K: np.ndarray
    dist: np.ndarray
    T_w_c: np.ndarray
    image_size: tuple[int, int]


def _find_calib_json(dataset_root: Path, preferred_name: str) -> Path:
    candidate = dataset_root / preferred_name
    if candidate.exists():
        return candidate

    matches = sorted(dataset_root.glob("*camera*params*.json"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        names = "\n".join(str(p) for p in matches)
        raise FileNotFoundError(f"Multiple candidate calibration json files found:\n{names}")
    raise FileNotFoundError(f"No calibration json found under {dataset_root}")


def _extract_intrinsics(entry: dict, camera_id: str) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    try:
        intr = entry["RGB"]["intrinsic"]
        dist = entry["RGB"]["distortion"]
    except KeyError as exc:
        raise KeyError(f"Missing RGB intrinsic/distortion in camera {camera_id}") from exc

    fx = float(intr["fx"])
    fy = float(intr["fy"])
    cx = float(intr["cx"])
    cy = float(intr["cy"])
    width = int(intr.get("width", entry["RGB"].get("width", 0)))
    height = int(intr.get("height", entry["RGB"].get("height", 0)))

    K = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    dist_vec = np.array(
        [
            float(dist.get("k1", 0.0)),
            float(dist.get("k2", 0.0)),
            float(dist.get("p1", 0.0)),
            float(dist.get("p2", 0.0)),
            float(dist.get("k3", 0.0)),
        ],
        dtype=np.float64,
    )
    return K, dist_vec, (width, height)


def _extract_twc(entry: dict, camera_id: str, use_mm_to_m_auto_scale: bool, extrinsics_are_twc: bool) -> np.ndarray:
    if "rotation" not in entry or "translation" not in entry:
        raise KeyError(f"Missing rotation/translation for camera {camera_id}")

    R = np.array(entry["rotation"], dtype=np.float64).reshape(3, 3)
    t = np.array(entry["translation"], dtype=np.float64).reshape(3)

    if use_mm_to_m_auto_scale and np.linalg.norm(t) > 10.0:
        t = t / 1000.0

    if not is_valid_rotation(R):
        raise ValueError(f"Invalid rotation matrix in camera {camera_id}")

    T = make_transform(R, t)
    if extrinsics_are_twc:
        return T
    return invert_transform(T)


def _resolve_camera_entry(data: dict, cam_id: str) -> dict:
    """Resolve camera key robustly for ids like 07/7."""
    if cam_id in data:
        return data[cam_id]

    try:
        cam_num = int(cam_id)
    except ValueError:
        cam_num = None

    candidates = []
    if cam_num is not None:
        candidates.extend([str(cam_num), f"{cam_num:02d}"])

    for key in candidates:
        if key in data:
            return data[key]

    raise KeyError(f"Camera id {cam_id} not found in calibration json")


def load_calibrations(
    dataset_root: Path,
    camera_ids: Iterable[str],
    calibration_filename: str,
    use_mm_to_m_auto_scale: bool,
    fixed_extrinsics_are_twc: bool,
) -> tuple[Path, Dict[str, CameraCalibration]]:
    calib_json_path = _find_calib_json(dataset_root, calibration_filename)
    with calib_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    out: Dict[str, CameraCalibration] = {}
    for cam_id in camera_ids:
        entry = _resolve_camera_entry(data, cam_id)
        K, dist, size = _extract_intrinsics(entry, cam_id)
        T_w_c = _extract_twc(entry, cam_id, use_mm_to_m_auto_scale, fixed_extrinsics_are_twc)
        out[cam_id] = CameraCalibration(camera_id=cam_id, K=K, dist=dist, T_w_c=T_w_c, image_size=size)
    return calib_json_path, out
