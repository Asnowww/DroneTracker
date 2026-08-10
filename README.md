# AirSim + YOLO 无人机视觉制导追踪

Tracker 无人机仅凭机载前视相机，用 YOLO 识别 Target 无人机，通过运动预测补偿链路延迟，
再由视觉伺服控制把目标保持在画面中心。在 AirSim 中验证，架构上为迁移实机预留了替换点。

完整方案见 **[docs/development_plan.md](docs/development_plan.md)**。
算法原理见 [docs/auto_tracking_algorithm.tex](docs/auto_tracking_algorithm.tex)。

## 快速开始

```powershell
# 1. 体检（模拟器需已启动）
python scripts\check_airsim_ready.py

# 2. 采集数据集 —— 用分割掩码自动标注，零人工标框
python scripts\collect_dataset.py --samples 3000

# 3. 训练（从 Hugging Face 无人机权重热启动）
python scripts\fetch_hf_assets.py --weights --which yolov11x
python scripts\train_yolo.py --data datasets\airsim_drone\data.yaml `
  --base weights\hf\yolov11x\weight\best.pt --name drone_sim_ft

# 4. 全自动 A/B 回归：3 种目标轨迹 x 3 种预测配置
python scripts\auto_test.py --plan config\auto_test_plan.json
```

不需要模拟器的离线验证：

```powershell
python tests\run_all.py
```

## 数据流

```
Tracker 相机 -> YOLO 检测 -> 运动预测 -> 视觉伺服控制 -> AirSim 速度/yaw/云台指令 -> 下一帧
```

只有 `src/drone_tracker/airsim_io.py` 直接接触仿真器。迁移实机时替换这一个文件为 MAVLink 实现，
检测 / 预测 / 控制三层不动。

## 目录

| 路径 | 内容 |
|---|---|
| `src/drone_tracker/` | 检测、预测、控制、自动标注、AirSim IO |
| `scripts/` | 体检、采集、训练、追踪、自动化测试 |
| `config/` | 追踪 / 数据集 / 测试计划配置 |
| `tests/` | 离线测试，无需 AirSim、GPU 或训练权重 |
| `docs/` | 开发方案与算法文档 |

## 运动预测

两种可选，配置里切换 `prediction.model`：

- `kalman` — 8 维常速度 Kalman，已在 AirSim 闭环验证的基线
- `imm` — 12 维 IMM，并行 CV / CA / 机动三个模型按后验概率融合

`tests/test_prediction_benchmark.py` 实测（24 s @ 20 Hz，检测噪声 σ=4 px，丢检率 8%）：

| 轨迹 | 无预测 | kalman | imm |
|---|---|---|---|
| 平滑巡航 RMSE | 15.55 px | 7.08 px | **6.20 px** |
| 急转机动 RMSE | 38.00 px | 35.09 px | **27.20 px** |
| 急转机动 P95 | 77.76 px | 61.73 px | **48.14 px** |

## 验收阈值

`config/auto_test_plan.json`：

```json
{ "visible_rate_min": 0.98, "lost_count_max": 0, "center_error_p95_px_max": 67.71 }
```
