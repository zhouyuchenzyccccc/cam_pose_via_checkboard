# cam_pose_via_checkboard

使用多台固定相机(00~05)与棋盘格观测，估计目标相机07每一帧在 cam00 世界系下的 `T_w_c07`。

## 输入约定

只需要传入 `dataset_root`。程序会自动读取：

- 内参文件：优先 `${dataset_root}/camera_params.json`，若不存在会搜索 `*camera*params*.json` 与 `*camera*parms*.json`
- 外参文件：优先 `${dataset_root}/extrinsics.json`，若不存在会自动尝试 `${dataset_root}/extrinsic.json` 或搜索 `*extrinsic*.json`
- 图像目录：`${dataset_root}/{cam_id}/RGB/{frame_index}.jpg`

其中 `cam_id` 默认使用：固定相机 `00~05` + 目标相机 `07`。

## 参数字段映射

从 `camera_params.json` 中读取：

- RGB内参：`[cam_id].RGB.intrinsic.{fx,fy,cx,cy,width,height}`
- RGB畸变：`[cam_id].RGB.distortion.{k1,k2,p1,p2,k3}`

外参优先读取：

- `extrinsics.json/extrinsic.json` 中的 `[cam_id].rotation` 与 `[cam_id].translation`
- 若内参文件本身也包含 `rotation/translation`，同样可直接使用

注意：默认只要求固定相机 `00~05` 在外参文件中存在，目标相机 `07` 可以不在外参文件里。

## 输出

默认输出到 `outputs/`：

- `trajectory_tw_c07.txt`：每行 `frame_index + 16个矩阵元素(按行展开)`
- `matrices/frame_{frame_index}.txt`：每帧一个4x4矩阵
- `diagnostics.csv`：每帧质量信息与失败原因
- `plots/*.png`：RMSE曲线、可见相机数曲线、07轨迹俯视图

## Ubuntu 安装与运行

```bash
cd cam_pose_via_checkboard
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
python3 -m src.main --dataset_root /home/ubuntu/orbbec/src/sync/test/test/zyc --config configs/default.yaml
```

Ubuntu 一键脚本：

```bash
cd cam_pose_via_checkboard
chmod +x scripts/run_example.sh
./scripts/run_example.sh /home/ubuntu/orbbec/src/sync/test/test/zyc
```

说明：

- 建议始终在 `cam_pose_via_checkboard` 目录下运行命令。
- 如果系统里默认 `python` 不是 Python 3，请统一使用 `python3`。

## 常见问题

- 角点检测失败较多：优先检查棋盘是否完整可见、运动模糊、曝光。
- 位姿尺度异常：确认外参 `translation` 单位，必要时启用 `use_mm_to_m_auto_scale`。
- 姿态方向看起来翻转：尝试切换 `fixed_extrinsics_are_twc`（外参方向可能与配置不一致）。

## AprilTag Workflow (Auto, no pre-defined world map)

This project now supports `target_type: apriltag`.
In this mode, you do **not** need to pre-fill each tag's world pose in a config file.
The world pose of each visible AprilTag is estimated automatically from fixed cameras per frame.

### 1) Config example

Edit `configs/default.yaml`:

```yaml
target_type: "apriltag"
apriltag_family: "tag36h11"   # tag16h5 | tag25h9 | tag36h10 | tag36h11
apriltag_default_size_m: 0.10   # physical tag size (meter)
apriltag_min_tags: 1
apriltag_min_inliers: 4
```

Notes:

- `apriltag_default_size_m` must match your real printed tag size.
- Current implementation assumes all tags use the same size.

### 2) Pipeline logic (per frame)

1. For each fixed camera (`00~05`), detect visible AprilTags.
2. For each detected tag, solve single-tag PnP in camera frame (`T_c_tag`).
3. Convert to world frame using fixed-camera extrinsic (`T_w_tag = T_w_c_fixed * T_c_tag`).
4. If one tag is seen by multiple fixed cameras, fuse those `T_w_tag` candidates.
5. Build this frame's world-tag map from fused tags.
6. For target camera (`07`), detect AprilTags and run joint PnP using:
   - image points from camera-07,
   - world 3D corners from this frame's world-tag map.
7. Invert the result to output `T_w_c07`.

### 3) Run

```bash
python -m src.main --dataset_root <your_dataset_root> --config configs/default.yaml
```

### 4) Failure reasons in diagnostics.csv

Common reasons for AprilTag mode:

- `insufficient_fixed_observations`: too few fixed cameras with valid detections.
- `insufficient_world_tags`: fixed-camera fusion did not produce enough world tags.
- `target_failed:tags_not_found`: target camera cannot detect tags in that frame.
- `target_failed:few_tags`: target camera sees fewer tags than `apriltag_min_tags`.
- `fixed_failed:opencv_has_no_aruco`: OpenCV build lacks `aruco` (install `opencv-contrib-python`).

## AprilTag 使用说明（自动建图，无需预设世界坐标）

当前已支持 `target_type: apriltag`。
在该模式下，不需要手工预先填写每个 tag 的世界坐标与位姿。
系统会在每一帧中基于固定相机自动估计可见 AprilTag 的世界位姿。

### 1）配置示例

请在 `configs/default.yaml` 中设置：

```yaml
target_type: "apriltag"
apriltag_family: "tag36h11"   # tag16h5 | tag25h9 | tag36h10 | tag36h11
apriltag_default_size_m: 0.10   # 标签物理边长（米）
apriltag_min_tags: 1
apriltag_min_inliers: 4
```

说明：

- `apriltag_default_size_m` 必须与实际打印标签尺寸一致，否则尺度会不准。
- 当前实现默认所有 tag 尺寸一致。

### 2）每帧处理流程

1. 对每台固定相机（`00~05`）检测当前帧可见的 AprilTag。
2. 对每个检测到的 tag 做单 tag PnP，得到相机系下 `T_c_tag`。
3. 结合固定相机外参，转换到世界系：`T_w_tag = T_w_c_fixed * T_c_tag`。
4. 若同一 tag 被多台固定相机同时看到，对该 tag 的多个 `T_w_tag` 候选进行融合。
5. 得到该帧的 world-tag map（本帧可用 tag 的世界坐标与位姿）。
6. 对头戴相机（`07`）检测 AprilTag，并使用：
   - 头戴相机图像上的 tag 角点（2D）
   - 本帧 world-tag map 的对应 3D 角点
   做联合 PnP。
7. 对结果取逆，得到头戴相机在世界系下位姿 `T_w_c07`。

### 3）运行命令

```bash
python -m src.main --dataset_root <your_dataset_root> --config configs/default.yaml
```

### 4）diagnostics.csv 常见失败原因

- `insufficient_fixed_observations`：可用固定相机数量不足。
- `insufficient_world_tags`：固定相机融合后可用于建图的 tag 数不足。
- `target_failed:tags_not_found`：头戴相机该帧未检测到 tag。
- `target_failed:few_tags`：头戴相机该帧可用 tag 数小于 `apriltag_min_tags`。
- `fixed_failed:opencv_has_no_aruco`：OpenCV 不含 `aruco` 模块（请安装 `opencv-contrib-python`）。
