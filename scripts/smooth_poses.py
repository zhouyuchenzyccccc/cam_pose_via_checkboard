#!/usr/bin/env python3
"""
对相机位姿估计输出目录做后处理：消除大跳变 + 平滑。

支持直接传入 main.py 的输出目录（含 trajectory_*.txt / matrices/ / plots/），
同步更新三类文件：
  1. trajectory_*.txt        逐行位姿文件（原地覆盖或写到新目录）
  2. matrices/frame_*.txt    每帧单独 4×4 矩阵
  3. plots/                  用平滑后数据重新生成图表

流程：
  XYZ：异常帧检测（>jump_thresh_mm）→ 插值 → 平滑
  RPY：XYZ 异常帧同步插值 + 可选独立跳变检测 → unwrap → 平滑 → wrap

用法（先分析跳变分布）：
  python scripts/smooth_poses.py --input_dir outputs/ --analyze

用法（原地平滑，覆盖输出目录）：
  python scripts/smooth_poses.py --input_dir outputs/ --inplace --jump_thresh_mm 30

用法（写到新目录）：
  python scripts/smooth_poses.py --input_dir outputs/ --output_dir outputs_smooth/ --jump_thresh_mm 30
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter


# ── I/O ─────────────────────────────────────────────────────────────────────

def load_trajectory(path: Path) -> tuple[list[str], np.ndarray]:
    """读取 trajectory_*.txt，返回 (frame_ids, poses (T,4,4))。"""
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


def save_matrices(matrices_dir: Path, frame_ids: list[str], poses: np.ndarray) -> None:
    matrices_dir.mkdir(parents=True, exist_ok=True)
    for fid, T in zip(frame_ids, poses):
        np.savetxt(matrices_dir / f"frame_{fid}.txt", T, fmt="%.9f")


def save_plots(plots_dir: Path, frame_ids: list[str], poses: np.ndarray) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    xs = poses[:, 0, 3]
    ys = poses[:, 1, 3]

    # 轨迹俯视图
    plt.figure(figsize=(6, 6))
    plt.plot(xs, ys, "-o", markersize=2)
    plt.title("Camera trajectory top view (X-Y)")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "traj_topview.png", dpi=150)
    plt.close()

    # XYZ 平移曲线
    plt.figure(figsize=(10, 4))
    t = range(len(frame_ids))
    plt.plot(t, xs, label="X")
    plt.plot(t, ys, label="Y")
    plt.plot(t, poses[:, 2, 3], label="Z")
    plt.title("Camera translation (smoothed)")
    plt.xlabel("Frame idx")
    plt.ylabel("m")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "translation_curve.png", dpi=150)
    plt.close()


# ── 旋转矩阵 <-> RPY ─────────────────────────────────────────────────────────

def rot_to_rpy(R: np.ndarray) -> np.ndarray:
    """ZYX 顺序 RPY，返回弧度 (3,)。"""
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy >= 1e-6:
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
    Rx = np.array([[1, 0,   0 ], [0, cr, -sr], [0,   sr, cr]])
    Ry = np.array([[cp, 0, sp ], [0,  1,   0], [-sp,  0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy,  0], [0,    0,  1]])
    return Rz @ Ry @ Rx


def poses_to_xyz_rpy(poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xyz = poses[:, :3, 3].copy().astype(np.float64)
    rpy = np.stack([rot_to_rpy(T[:3, :3]) for T in poses], axis=0)
    return xyz, rpy


def xyz_rpy_to_poses(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    out = np.tile(np.eye(4, dtype=np.float64), (len(xyz), 1, 1))
    for i in range(len(xyz)):
        out[i, :3, :3] = rpy_to_rot(rpy[i])
        out[i, :3,  3] = xyz[i]
    return out


# ── 核心后处理 ───────────────────────────────────────────────────────────────

def detect_jump_outliers(signal: np.ndarray, thresh: float, context: int) -> np.ndarray:
    T = len(signal)
    bad = np.zeros(T, dtype=bool)
    if T < 2:
        return bad
    diff = np.diff(signal, axis=0)
    jumps = np.linalg.norm(diff, axis=1) if diff.ndim == 2 else np.abs(diff)
    for i in np.where(jumps > thresh)[0]:
        bad[max(0, i + 1 - context): min(T, i + 2 + context)] = True
    return bad


def interpolate_bad(signal: np.ndarray, bad: np.ndarray) -> np.ndarray:
    result = signal.copy()
    vi = np.where(~bad)[0]
    if len(vi) < 2:
        return result
    for d in range(signal.shape[1]):
        result[np.where(bad)[0], d] = np.interp(np.where(bad)[0], vi, signal[vi, d])
    return result


def smooth_signal(sig: np.ndarray, method: str, window: int) -> np.ndarray:
    T = len(sig)
    win = min(window, T if T % 2 == 1 else T - 1)
    if win < 5 or method == "none":
        return sig
    s = sig.copy()
    for d in range(sig.shape[1]):
        if method == "savgol":
            s[:, d] = savgol_filter(sig[:, d], win, 3)
        elif method == "median_then_savgol":
            s[:, d] = median_filter(sig[:, d], size=5)
            s[:, d] = savgol_filter(s[:, d], win, 3)
    return s


def smooth_rpy(rpy: np.ndarray, method: str, window: int) -> np.ndarray:
    unwrapped = np.unwrap(rpy, axis=0)
    smoothed = smooth_signal(unwrapped, method, window)
    return (smoothed + np.pi) % (2 * np.pi) - np.pi


def postprocess_poses(
    poses: np.ndarray,
    jump_thresh_m: float,
    smooth_method: str,
    smooth_window: int,
    context: int,
    rpy_jump_thresh_rad: float | None = None,
) -> tuple[np.ndarray, int, int]:
    """返回 (smoothed_poses, n_xyz_fixed, n_rpy_fixed)。"""
    xyz, rpy = poses_to_xyz_rpy(poses)

    xyz_bad = detect_jump_outliers(xyz, jump_thresh_m, context)
    n_xyz_fixed = int(xyz_bad.sum())
    if n_xyz_fixed > 0:
        xyz = interpolate_bad(xyz, xyz_bad)
        rpy = interpolate_bad(rpy, xyz_bad)

    n_rpy_fixed = 0
    if rpy_jump_thresh_rad is not None:
        rpy_diff = np.diff(rpy, axis=0)
        rpy_diff = (rpy_diff + np.pi) % (2 * np.pi) - np.pi
        rpy_bad = np.zeros(len(rpy), dtype=bool)
        for i in np.where(np.linalg.norm(rpy_diff, axis=1) > rpy_jump_thresh_rad)[0]:
            rpy_bad[max(0, i + 1 - context): min(len(rpy), i + 2 + context)] = True
        n_rpy_fixed = int((rpy_bad & ~xyz_bad).sum())
        if rpy_bad.any():
            rpy = interpolate_bad(rpy, rpy_bad)

    xyz = smooth_signal(xyz, smooth_method, smooth_window)
    rpy = smooth_rpy(rpy, smooth_method, smooth_window)

    return xyz_rpy_to_poses(xyz, rpy), n_xyz_fixed, n_rpy_fixed


# ── 输出目录级别处理 ─────────────────────────────────────────────────────────

def _find_trajectory(input_dir: Path) -> Path | None:
    """在目录下找 trajectory_*.txt，优先精确匹配常见命名。"""
    for name in ("trajectory_tw_c07.txt", "trajectory_tw_c08.txt"):
        p = input_dir / name
        if p.exists():
            return p
    candidates = sorted(input_dir.glob("trajectory_*.txt"))
    return candidates[0] if candidates else None


def process_output_dir(
    input_dir: Path,
    output_dir: Path,
    args,
    dry_run: bool = False,
) -> dict:
    """
    处理一个完整的输出目录（trajectory + matrices + plots）。
    output_dir 可与 input_dir 相同（inplace）。
    返回统计字典。
    """
    traj_path = _find_trajectory(input_dir)
    if traj_path is None:
        return {"skipped": True, "reason": "no_trajectory_file"}

    frame_ids, poses = load_trajectory(traj_path)
    jump_thresh_m = args.jump_thresh_mm / 1000.0
    rpy_thresh_rad = (args.rpy_jump_thresh_deg * np.pi / 180.0
                      if args.rpy_jump_thresh_deg is not None else None)

    smoothed, n_xyz, n_rpy = postprocess_poses(
        poses, jump_thresh_m, args.smooth_method,
        args.smooth_window, args.context, rpy_thresh_rad,
    )

    if not dry_run:
        # 若输出到新目录，先复制 diagnostics.csv 等非位姿文件
        if output_dir != input_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            for f in input_dir.iterdir():
                if f.is_file() and not f.name.startswith("trajectory_"):
                    shutil.copy2(f, output_dir / f.name)

        # 1. trajectory
        out_traj = output_dir / traj_path.name
        save_trajectory(out_traj, frame_ids, smoothed)

        # 2. matrices
        matrices_out = output_dir / "matrices"
        save_matrices(matrices_out, frame_ids, smoothed)

        # 3. plots
        plots_out = output_dir / "plots"
        save_plots(plots_out, frame_ids, smoothed)

    return {"skipped": False, "frames": len(poses),
            "xyz_fixed": n_xyz, "rpy_fixed": n_rpy}


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="相机位姿轨迹后处理：去除跳变 + 平滑（同步更新 trajectory / matrices / plots）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    ap.add_argument("--input_dir", type=Path, required=True,
                    help="输出目录路径（含 trajectory_*.txt / matrices/ / plots/）")

    dst = ap.add_mutually_exclusive_group()
    dst.add_argument("--output_dir", type=Path,
                     help="写到新目录（不传则原地覆盖）")
    dst.add_argument("--inplace", action="store_true",
                     help="原地覆盖（默认行为，与不传 --output_dir 等价）")

    ap.add_argument("--jump_thresh_mm", type=float, default=50.0,
                    help="平移跳变阈值（mm），默认 50")
    ap.add_argument("--rpy_jump_thresh_deg", type=float, default=None,
                    help="旋转跳变阈值（deg），不设则不单独检测")
    ap.add_argument("--context", type=int, default=1,
                    help="异常帧前后额外标记帧数，默认 1")
    ap.add_argument("--smooth_method", default="median_then_savgol",
                    choices=["savgol", "median_then_savgol", "none"],
                    help="平滑方法，默认 median_then_savgol")
    ap.add_argument("--smooth_window", type=int, default=11,
                    help="Savitzky-Golay 窗口（奇数），默认 11")
    ap.add_argument("--dry_run", action="store_true",
                    help="只统计，不写文件")
    ap.add_argument("--analyze", action="store_true",
                    help="打印跳变分布，帮助选择阈值")

    args = ap.parse_args()
    if args.smooth_window % 2 == 0:
        args.smooth_window += 1

    input_dir: Path = args.input_dir
    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    # 判断是单个输出目录还是包含多个子目录
    traj = _find_trajectory(input_dir)
    if traj is not None:
        # 直接就是一个输出目录
        target_dirs = [input_dir]
    else:
        # 当作父目录，搜索含 trajectory_*.txt 的子目录
        target_dirs = sorted(
            d for d in input_dir.iterdir()
            if d.is_dir() and _find_trajectory(d) is not None
        )
        if not target_dirs:
            raise FileNotFoundError(
                f"在 {input_dir} 及其子目录下未找到任何 trajectory_*.txt 文件"
            )

    rpy_info = f"  rpy_thresh={args.rpy_jump_thresh_deg}deg" if args.rpy_jump_thresh_deg else ""
    print(f"找到 {len(target_dirs)} 个输出目录")
    print(f"参数: jump_thresh={args.jump_thresh_mm}mm{rpy_info}"
          f"  smooth={args.smooth_method}  window={args.smooth_window}  context=±{args.context}帧")
    if args.dry_run:
        print("[dry_run] 不写文件\n")

    # --analyze
    if args.analyze:
        all_xyz: list[np.ndarray] = []
        all_rpy: list[np.ndarray] = []
        for d in target_dirs:
            tp = _find_trajectory(d)
            if tp is None:
                continue
            try:
                _, poses = load_trajectory(tp)
            except ValueError:
                continue
            if len(poses) < 2:
                continue
            xyz, rpy = poses_to_xyz_rpy(poses)
            all_xyz.append(np.linalg.norm(np.diff(xyz, axis=0), axis=1) * 1000.0)
            rd = np.diff(rpy, axis=0)
            rd = (rd + np.pi) % (2 * np.pi) - np.pi
            all_rpy.append(np.linalg.norm(rd, axis=1) * 180.0 / np.pi)

        if all_xyz:
            j = np.concatenate(all_xyz)
            print(f"\n=== XYZ 跳变分布（mm）  样本数: {len(j)} ===")
            print(f"  P50={np.percentile(j,50):.2f}  P90={np.percentile(j,90):.2f}"
                  f"  P95={np.percentile(j,95):.2f}  P99={np.percentile(j,99):.2f}"
                  f"  max={j.max():.2f}")
            for t in [10, 20, 30, 50, 100]:
                cnt = int(np.sum(j > t))
                print(f"  >{t:3d}mm: {cnt:5d} 帧 ({100*cnt/len(j):.2f}%)")

        if all_rpy:
            j = np.concatenate(all_rpy)
            print(f"\n=== RPY 跳变分布（deg）  样本数: {len(j)} ===")
            print(f"  P50={np.percentile(j,50):.2f}  P90={np.percentile(j,90):.2f}"
                  f"  P95={np.percentile(j,95):.2f}  P99={np.percentile(j,99):.2f}"
                  f"  max={j.max():.2f}")
            for t in [5, 10, 20, 30, 45]:
                cnt = int(np.sum(j > t))
                print(f"  >{t:2d}deg: {cnt:5d} 帧 ({100*cnt/len(j):.2f}%)")
        print()
        if not (args.dry_run or args.output_dir) and not args.inplace:
            return

    # 主处理循环
    total_xyz = total_rpy = 0
    for d in target_dirs:
        if args.output_dir:
            # 保持相对结构：若有多个子目录则在 output_dir 下建同名子目录
            rel = d.relative_to(input_dir) if d != input_dir else Path(".")
            out_d = args.output_dir / rel if rel != Path(".") else args.output_dir
        else:
            out_d = d  # inplace

        r = process_output_dir(d, out_d, args, dry_run=args.dry_run)
        if r.get("skipped"):
            print(f"  {d.name}  [跳过: {r['reason']}]")
            continue

        total_xyz += r["xyz_fixed"]
        total_rpy += r["rpy_fixed"]
        arrow = f" -> {out_d}" if out_d != d else " [inplace]"
        print(f"  {d.name}{arrow}  xyz={r['xyz_fixed']} rpy={r['rpy_fixed']}/{r['frames']}")

    print(f"\n汇总: XYZ修复={total_xyz}  RPY修复={total_rpy}")


if __name__ == "__main__":
    main()
