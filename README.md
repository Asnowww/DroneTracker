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

## 检测模型

四级递进训练（`scripts/train_yolo.py`），当前完成到第 2 级：

| 级 | 权重 | 数据 |
|---|---|---|
| 0 | `yolov8s.pt` | COCO |
| 1 | `runs/detect/real_prior_v8s` | Seraphim 4 万张真实无人机图 |
| 2 | `runs/detect/mixed_ft_v8s` ← **当前使用** | 1.8 万真实 + 2503 AirSim（7.19:1） |
| 3 | 待做 | 你自己平台的空对空实拍，首飞前必需 |

分域评估（`scripts/eval_domains.py`）：

| 模型 | 参数 | 真实域 mAP50 | 真实域 mAP50-95 | 仿真域 mAP50 |
|---|---|---|---|---|
| **mixed_ft_v8s** | 11.1M | 0.9185 | 0.6277 | **0.9950** |
| real_prior_v8s | 11.1M | 0.9287 | 0.6291 | 0.9656 |

混合微调只损失 0.0014 的真实域精度，换来仿真域接近满分。

## 运动预测

配置里切换 `prediction.model`：`kalman`（8 维常速度）或 `imm`（12 维多模型，**默认**）。

`tests/test_prediction_benchmark.py` 离线实测（24 s @ 20 Hz，检测噪声 σ=4 px，丢检率 8%）：

| 轨迹 | 无预测 | kalman | imm |
|---|---|---|---|
| 平滑巡航 RMSE | 15.55 px | 7.08 px | **6.20 px** |
| 急转机动 RMSE | 38.00 px | 35.09 px | **27.20 px** |

## 闭环回归结果

`runs/auto_test/20260810-194356`，**9/9 通过**（90 s × 3 轨迹 × 3 预测配置）：

| 轨迹 / 配置 | 可见率 | 丢失 | 中心误差 p95 |
|---|---|---|---|
| front_sweep / imm | 1.000 | 0 | **43.3 px** |
| lateral_dash / imm | 1.000 | 0 | **23.0 px** |
| figure8 / imm（压力项） | 1.000 | 0 | 97.9 px |

检测器从 YOLOv11x 换到 YOLOv8s 后，控制环从 5.9 Hz 提升到 **8.1 Hz**，未改动任何控制增益即让所有指标改善——figure8 更是从可见率 0.70 / 7 次丢失变为满分。**环路速率是比增益更有效的杠杆。**

验收阈值见 `config/auto_test_plan.json`：标准轨迹 `p95 ≤ 67.71 / visible ≥ 0.98 / lost = 0`；
`figure8` 为压力层，阈值放宽至 160 px。
