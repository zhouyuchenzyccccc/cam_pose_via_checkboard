# cam_pose_via_checkboard

使用多台固定相机(00~05)与棋盘格观测，估计目标相机07每一帧在 cam00 世界系下的 `T_w_c07`。

## 输入约定

只需要传入 `dataset_root`。程序会自动读取：

- 参数文件：优先 `${dataset_root}/camera_params.json`，若不存在会在根目录搜索 `*camera*params*.json`
- 图像目录：`${dataset_root}/{cam_id}/RGB/{frame_index}.jpg`

其中 `cam_id` 默认使用：固定相机 `00~05` + 目标相机 `07`。

## 参数字段映射

从 `camera_params.json` 中读取：

- 外参：`[cam_id].rotation`、`[cam_id].translation`
- RGB内参：`[cam_id].RGB.intrinsic.{fx,fy,cx,cy,width,height}`
- RGB畸变：`[cam_id].RGB.distortion.{k1,k2,p1,p2,k3}`

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
