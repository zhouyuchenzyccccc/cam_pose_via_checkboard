#!/usr/bin/env python3
"""
对 trajectory_tw_c07.txt 输出的相机位姿做后处理，消除大跳变。

位姿格式：每行  frame_index  m00 m01 ... m33  （4×4 T_w_c 按行展开，16 个数字）
平移 XYZ 直接从矩阵第 4 列读取；旋转转换为 RPY 后平滑，再转回旋转矩阵。

流程：
  XYZ：异常帧检测（>jump_thresh_mm）→ 插值 → 平滑
  RPY：XYZ 异常帧同步插值 + 可选独立 RPY 跳变检测 → unwrap → 平滑 → wrap

用法（先分析分布，再决定阈值）：
  python scripts/smooth_poses.py --input outputs/trajectory_tw_c07.txt --analyze

用法（平滑单文件）：
  python scripts/smooth_poses.py \\
      --input  outputs/trajectory_tw_c07.txt \\
      --output outputs/trajectory_tw_c07_smooth.txt

用法（批量目录）：
  python scripts/smooth_poses.py \\
      --input_dir  /path/to/poses \\
      --output_dir /path/to/smooth \\
      --jump_thresh_mm 30 --rpy_jump_thresh_deg 20
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter


# ── I/O ─────────────────────────────────────────────────────────────────────

def load_trajectory(path: Path) -> tuple[list[str], np.ndarray]:
    """
    读取 trajectory_tw_c07.txt。
    返回 (frame_ids, poses)：
      frame_ids : list[str]，长度 T
      poses     : float64 (T, 4, 4)
    跳过成功帧之外的行（即只保留有完整 16 个数字的行）。
    """
    frame_ids: list[str] = []
    matrices: list[np.ndarray] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) != 17:
                continue
            frame_ids.append(parts[0])
            matrices.append(np.array(parts[1:], dtype=np.float64).reshape(4, 4))
    if not matrices:
        raise ValueError(f"文件中没有有效位姿行: {path}")
    return frame_ids, np.stack(matrices, axis=0)


def save_trajectory(path: Path, frame_ids: list[str], poses: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for fid, T in zip(frame_ids, poses):
            flat = " ".join(f"{x:.9f}" for x in T.reshape(-1))
            f.write(f"{fid} {flat}\n")


# ── 旋转矩阵 <-> RPY ─────────────────────────────────────────────────────────

def rot_to_rpy(R: np.ndarray) -> np.ndarray:
    """ZYX 顺序 RPY（roll-pitch-yaw），返回弧度，shape (3,)。"""
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        roll  = np.arctan2( R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw   = np.arctan2( R[1, 0], R[0, 0])
    else:
        roll  = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw   = 0.0
    return np.array([roll, pitch, yaw], dtype=np.float64)


def rpy_to_rot(rpy: np.ndarray) -> np.ndarray:
    """ZYX 顺序 RPY（弧度）→ 旋转矩阵 (3,3)。"""
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll),  np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),   np.sin(yaw)
    Rx = np.array([[1, 0,   0  ], [0, cr, -sr], [0, sr,  cr]])
    Ry = np.array([[cp, 0, sp  ], [0,  1,   0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0 ], [sy, cy,  0], [0,   0,  1]])
    return Rz @ Ry @ Rx


def poses_to_xyz_rpy(poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(T,4,4) → xyz (T,3), rpy (T,3)"""
    xyz = poses[:, :3, 3]
    rpy = np.stack([rot_to_rpy(T[:3, :3]) for T in poses], axis=0)
    return xyz.astype(np.float64), rpy.astype(np.float64)


def xyz_rpy_to_poses(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    """xyz (T,3), rpy (T,3) → (T,4,4)"""
    T_count = len(xyz)
    poses = np.tile(np.eye(4, dtype=np.float64), (T_count, 1, 1))
    for i in range(T_count):
        poses[i, :3, :3] = rpy_to_rot(rpy[i])
        poses[i, :3,  3] = xyz[i]
    return poses


# ── 核心后处理 ───────────────────────────────────────────────────────────────

def detect_jump_outliers(signal: np.ndarray, jump_thresh: float,
                         context: int = 1) -> np.ndarray:
    """检测帧间跳变异常帧，返回 bool mask（True = 异常）。"""
    T = len(signal)
    bad = np.zeros(T, dtype=bool)
    if T < 2:
        return bad
    diff = np.diff(signal, axis=0)
    jumps = np.linalg.norm(diff, axis=1) if diff.ndim == 2 else np.abs(diff)
    for i in np.where(jumps > jump_thresh)[0]:
        lo = max(0, i + 1 - context)
        hi = min(T - 1, i + 1 + context)
        bad[lo:hi + 1] = True
    return bad


def interpolate_bad(signal: np.ndarray, bad: np.ndarray) -> np.ndarray:
    """用前后有效帧线性插值替换异常帧。"""
    result = signal.copy()
    good = ~bad
    vi = np.where(good)[0]
    if len(vi) < 2:
        return result
    ii = np.where(bad)[0]
    for d in range(signal.shape[1]):
        result[ii, d] = np.interp(ii, vi, signal[vi, d])
    return result


def smooth_xyz(xyz: np.ndarray, method: str, window: int) -> np.ndarray:
    T = len(xyz)
    win = min(window, T if T % 2 == 1 else T - 1)
    if win < 5 or method == "none":
        return xyz
    s = xyz.copy()
    for d in range(3):
        if method == "savgol":
            s[:, d] = savgol_filter(xyz[:, d], win, 3)
        elif method == "median_then_savgol":
            s[:, d] = median_filter(xyz[:, d], size=5)
            s[:, d] = savgol_filter(s[:, d], win, 3)
    return s


def smooth_rpy(rpy: np.ndarray, method: str, window: int) -> np.ndarray:
    """unwrap → 平滑 → wrap 回 [-π, π]。"""
    T = len(rpy)
    win = min(window, T if T % 2 == 1 else T - 1)
    if win < 5 or method == "none":
        return rpy
    unwrapped = np.unwrap(rpy, axis=0)
    s = unwrapped.copy()
    for d in range(3):
        if method == "savgol":
            s[:, d] = savgol_filter(unwrapped[:, d], win, 3)
        elif method == "median_then_savgol":
            s[:, d] = median_filter(unwrapped[:, d], size=5)
            s[:, d] = savgol_filter(s[:, d], win, 3)
    return (s + np.pi) % (2 * np.pi) - np.pi


def postprocess_poses(
    poses: np.ndarray,
    jump_thresh_m: float,
    smooth_method: str,
    smooth_window: int,
    context: int,
    rpy_jump_thresh_rad: float | None = None,
) -> tuple[np.ndarray, int, int]:
    """
    处理 (T,4,4) 位姿序列，返回 (smoothed_poses, n_xyz_fixed, n_rpy_fixed)。
    """
    xyz, rpy = poses_to_xyz_rpy(poses)

    # XYZ 跳变检测与插值
    xyz_bad = detect_jump_outliers(xyz, jump_thresh_m, context)
    n_xyz_fixed = int(xyz_bad.sum())
    if n_xyz_fixed > 0:
        xyz = interpolate_bad(xyz, xyz_bad)
        rpy = interpolate_bad(rpy, xyz_bad)  # XYZ 异常帧旋转同样不可信

    # RPY 独立跳变检测
    n_rpy_fixed = 0
    if rpy_jump_thresh_rad is not None:
        rpy_diff = np.diff(rpy, axis=0)
        rpy_diff = (rpy_diff + np.pi) % (2 * np.pi) - np.pi
        rpy_bad = np.zeros(len(rpy), dtype=bool)
        for i in np.where(np.linalg.norm(rpy_diff, axis=1) > rpy_jump_thresh_rad)[0]:
            lo = max(0, i + 1 - context)
            hi = min(len(rpy) - 1, i + 1 + context)
            rpy_bad[lo:hi + 1] = True
        n_rpy_fixed = int((rpy_bad & ~xyz_bad).sum())
        if rpy_bad.any():
            rpy = interpolate_bad(rpy, rpy_bad)

    # 平滑
    xyz = smooth_xyz(xyz, smooth_method, smooth_window)
    rpy = smooth_rpy(rpy, smooth_method, smooth_window)

    return xyz_rpy_to_poses(xyz, rpy), n_xyz_fixed, n_rpy_fixed


def process_file(input_path: Path, output_path: Path, args) -> dict:
    frame_ids, poses = load_trajectory(input_path)
    jump_thresh_m = args.jump_thresh_mm / 1000.0
    rpy_thresh_rad = (args.rpy_jump_thresh_deg * np.pi / 180.0
                      if args.rpy_jump_thresh_deg is not None else None)
    smoothed, n_xyz, n_rpy = postprocess_poses(
        poses, jump_thresh_m, args.smooth_method,
        args.smooth_window, args.context, rpy_thresh_rad,
    )
    if not args.dry_run:
        save_trajectory(output_path, frame_ids, smoothed)
    return {"frames": len(poses), "xyz_fixed": n_xyz, "rpy_fixed": n_rpy}


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="相机位姿轨迹后处理：去除跳变 + 平滑",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input",     type=Path, help="单个轨迹文件路径")
    src.add_argument("--input_dir", type=Path, help="包含多个轨迹 .txt 文件的目录")

    dst = ap.add_mutually_exclusive_group()
    dst.add_argument("--output",     type=Path, help="单文件输出路径（与 --input 配合）")
    dst.add_argument("--output_dir", type=Path, help="批量输出目录（与 --input_dir 配合）")
    dst.add_argument("--inplace",    action="store_true", help="直接覆盖原文件")

    ap.add_argument("--jump_thresh_mm", type=float, default=50.0,
                    help="平移帧间跳变阈值（mm），超过则视为异常帧（默认 50mm）")
    ap.add_argument("--rpy_jump_thresh_deg", type=float, default=None,
                    help="旋转帧间跳变阈值（deg），不设则不单独检测旋转异常帧")
    ap.add_argument("--context", type=int, default=1,
                    help="异常帧前后额外标记的帧数（默认 1）")
    ap.add_argument("--smooth_method", default="median_then_savgol",
                    choices=["savgol", "median_then_savgol", "none"],
                    help="平滑方法（默认 median_then_savgol）")
    ap.add_argument("--smooth_window", type=int, default=11,
                    help="Savitzky-Golay 窗口大小，奇数（默认 11）")
    ap.add_argument("--dry_run", action="store_true",
                    help="只统计，不写文件")
    ap.add_argument("--analyze", action="store_true",
                    help="打印跳变分布，帮助选择合适阈值")

    args = ap.parse_args()
    if args.smooth_window % 2 == 0:
        args.smooth_window += 1

    # 收集输入文件
    if args.input:
        input_files = [args.input]
    else:
        input_files = sorted(args.input_dir.rglob("*.txt"))
        if not input_files:
            raise FileNotFoundError(f"在 {args.input_dir} 下未找到任何 .txt 文件")

    print(f"找到 {len(input_files)} 个文件")
    rpy_info = f"  rpy_thresh={args.rpy_jump_thresh_deg}deg" if args.rpy_jump_thresh_deg else ""
    print(f"参数: jump_thresh={args.jump_thresh_mm}mm{rpy_info}"
          f"  smooth={args.smooth_method}  window={args.smooth_window}  context=±{args.context}帧")
    if args.dry_run:
        print("[dry_run] 不写文件\n")

    # --analyze
    if args.analyze:
        all_xyz_jumps: list[np.ndarray] = []
        all_rpy_jumps: list[np.ndarray] = []
        for fp in input_files:
            try:
                _, poses = load_trajectory(fp)
            except ValueError:
                continue
            if len(poses) < 2:
                continue
            xyz, rpy = poses_to_xyz_rpy(poses)
            all_xyz_jumps.append(np.linalg.norm(np.diff(xyz, axis=0), axis=1) * 1000.0)
            rpy_diff = np.diff(rpy, axis=0)
            rpy_diff = (rpy_diff + np.pi) % (2 * np.pi) - np.pi
            all_rpy_jumps.append(np.linalg.norm(rpy_diff, axis=1) * 180.0 / np.pi)

        if all_xyz_jumps:
            j = np.concatenate(all_xyz_jumps)
            print(f"\n=== XYZ 跳变分布（mm）  样本数: {len(j)} ===")
            print(f"  P50={np.percentile(j,50):.2f}  P90={np.percentile(j,90):.2f}"
                  f"  P95={np.percentile(j,95):.2f}  P99={np.percentile(j,99):.2f}"
                  f"  max={j.max():.2f}")
            for t in [10, 20, 30, 50, 100]:
                cnt = int(np.sum(j > t))
                print(f"  >{t:3d}mm: {cnt:5d} 帧 ({100*cnt/len(j):.2f}%)")

        if all_rpy_jumps:
            j = np.concatenate(all_rpy_jumps)
            print(f"\n=== RPY 跳变分布（deg）  样本数: {len(j)} ===")
            print(f"  P50={np.percentile(j,50):.2f}  P90={np.percentile(j,90):.2f}"
                  f"  P95={np.percentile(j,95):.2f}  P99={np.percentile(j,99):.2f}"
                  f"  max={j.max():.2f}")
            for t in [5, 10, 20, 30, 45]:
                cnt = int(np.sum(j > t))
                print(f"  >{t:2d}deg: {cnt:5d} 帧 ({100*cnt/len(j):.2f}%)")
        print()
        if not (args.dry_run or args.inplace or args.output or args.output_dir):
            return

    # 主循环
    total_xyz = total_rpy = 0
    for fp in input_files:
        if args.input:
            out = args.output if args.output else fp.with_name(fp.stem + "_smooth.txt")
        elif args.inplace:
            out = fp
        else:
            out = args.output_dir / fp.name

        r = process_file(fp, out, args)
        total_xyz += r["xyz_fixed"]
        total_rpy += r["rpy_fixed"]
        arrow = f" -> {out}" if out != fp else " [inplace]"
        print(f"  {fp.name}{arrow}  xyz={r['xyz_fixed']} rpy={r['rpy_fixed']}/{r['frames']}")

    print(f"\n汇总: XYZ修复={total_xyz}  RPY修复={total_rpy}")


if __name__ == "__main__":
    main()
