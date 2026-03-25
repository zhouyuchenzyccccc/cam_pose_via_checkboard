from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import cv2

from .calib_io import CameraCalibration
from .chessboard import solve_board_pnp
from .config import RuntimeConfig
from .fusion import BoardCandidate, fuse_board_pose
from .se3 import compose, invert_transform


def _discover_frames(dataset_root: Path, camera_id: str) -> set[str]:
    rgb_dir = dataset_root / camera_id / "RGB"
    if not rgb_dir.exists():
        return set()
    return {p.stem for p in rgb_dir.glob("*.jpg")}


def _collect_frame_indices(dataset_root: Path, cfg: RuntimeConfig) -> List[str]:
    target_frames = _discover_frames(dataset_root, cfg.target_camera_id)
    fixed_sets = [_discover_frames(dataset_root, cid) for cid in cfg.fixed_camera_ids]

    if cfg.frame_policy == "target_primary":
        return sorted(target_frames)

    if not fixed_sets:
        return sorted(target_frames)

    inter = set(target_frames)
    for s in fixed_sets:
        inter &= s
    return sorted(inter)


def _read_image(dataset_root: Path, cam_id: str, frame_index: str):
    path = dataset_root / cam_id / "RGB" / f"{frame_index}.jpg"
    if not path.exists():
        return None
    return cv2.imread(str(path))


def run_pipeline(dataset_root: Path, cfg: RuntimeConfig, calibrations: Dict[str, CameraCalibration]) -> List[dict]:
    logger = logging.getLogger("pipeline")
    frames = _collect_frame_indices(dataset_root, cfg)
    logger.info("Discovered %d frame indices", len(frames))

    rows: List[dict] = []
    for frame in frames:
        row = {
            "frame_index": frame,
            "success": False,
            "reason": "",
            "visible_fixed": [],
            "used_fixed": [],
            "inlier_fixed": [],
            "target_inliers": 0,
            "target_rmse": float("inf"),
            "T_w_c07": None,
        }

        candidates: List[BoardCandidate] = []
        for cam_id in cfg.fixed_camera_ids:
            cam = calibrations.get(cam_id)
            if cam is None:
                continue
            image = _read_image(dataset_root, cam_id, frame)
            res = solve_board_pnp(
                image=image,
                K=cam.K,
                dist=cam.dist,
                cols=cfg.board_cols,
                rows=cfg.board_rows,
                square_size_m=cfg.square_size_m,
                reproj_error_px=cfg.pnp_reproj_error_px,
                pnp_iterations=cfg.pnp_iterations,
                min_inliers=cfg.min_inliers,
            )
            if not res.success or res.T_c_b is None:
                continue
            row["visible_fixed"].append(cam_id)
            T_w_b = compose(cam.T_w_c, res.T_c_b)
            candidates.append(BoardCandidate(cam_id, T_w_b, res.reproj_rmse, res.inliers))

        if len(candidates) < cfg.min_fixed_observations:
            row["reason"] = "insufficient_fixed_observations"
            rows.append(row)
            continue

        fused = fuse_board_pose(
            candidates,
            trans_thresh_m=cfg.fusion_ransac_trans_thresh_m,
            rot_thresh_deg=cfg.fusion_ransac_rot_thresh_deg,
        )
        row["used_fixed"] = fused.used_camera_ids
        row["inlier_fixed"] = fused.inlier_camera_ids
        if not fused.success or fused.T_w_b is None:
            row["reason"] = f"fusion_failed:{fused.reason}"
            rows.append(row)
            continue

        target_cam = calibrations[cfg.target_camera_id]
        image_target = _read_image(dataset_root, cfg.target_camera_id, frame)
        target_res = solve_board_pnp(
            image=image_target,
            K=target_cam.K,
            dist=target_cam.dist,
            cols=cfg.board_cols,
            rows=cfg.board_rows,
            square_size_m=cfg.square_size_m,
            reproj_error_px=cfg.pnp_reproj_error_px,
            pnp_iterations=cfg.pnp_iterations,
            min_inliers=cfg.min_inliers,
        )
        if not target_res.success or target_res.T_c_b is None:
            row["reason"] = f"target_failed:{target_res.reason}"
            rows.append(row)
            continue

        T_w_c07 = compose(fused.T_w_b, invert_transform(target_res.T_c_b))
        row["success"] = True
        row["reason"] = ""
        row["target_inliers"] = target_res.inliers
        row["target_rmse"] = target_res.reproj_rmse
        row["T_w_c07"] = T_w_c07
        rows.append(row)

    logger.info("Pipeline finished: success=%d / total=%d", sum(int(r["success"]) for r in rows), len(rows))
    return rows
