# AirSim + YOLO 无人机视觉制导：完整开发方案

目标：Tracker 无人机仅靠机载前视相机，用 YOLO 持续识别 Target 无人机，并把它保持在画面中心附近；
加入运动预测提升追踪精度与响应速度；在 AirSim 跑通后迁移到实机。

---

## 0. 系统架构

```
Tracker 相机 (RGB / Depth)
   -> YOLO 检测器          detector.py        输出 Detection(xyxy, conf, cls)
   -> 运动预测器            predictor.py / imm_predictor.py
   -> 视觉伺服控制器        controller.py      输出 vx / vz / yaw_rate
   -> AirSim 执行           airsim_io.py       moveByVelocityBodyFrameAsync
   -> 下一帧
```

设计上只有 `airsim_io.py` 直接接触仿真器。**迁移到实机时只需要换掉这一个文件**（见第 6 节），
检测、预测、控制三层完全不动。这是整套方案能落地的关键。

| 文件 | 职责 |
|---|---|
| `src/drone_tracker/airsim_io.py` | AirSim RPC 封装：连接、起飞、取图、速度指令、云台、分割、天气 |
| `src/drone_tracker/detector.py` | YOLO 推理封装 |
| `src/drone_tracker/predictor.py` | 常速度 Kalman 预测器（基线） |
| `src/drone_tracker/imm_predictor.py` | IMM 多模型预测器（升级项） |
| `src/drone_tracker/prediction.py` | 预测器工厂，按配置选型 |
| `src/drone_tracker/controller.py` | 视觉伺服控制律 + 云台 |
| `src/drone_tracker/labeling.py` | 自动标注：分割掩码 + 三维投影 |
| `src/drone_tracker/target_policy.py` | 目标机的脚本化轨迹 |
| `src/drone_tracker/metrics.py` | CSV / summary 指标 |

---

## 1. 阶段 0：环境准备

### 1.1 AirSim 设置

`~/Documents/AirSim/settings.json` 必须包含两台无人机和两种图像类型：

```json
{
  "SettingsVersion": 1.2,
  "SimMode": "Multirotor",
  "ClockSpeed": 1,
  "Vehicles": {
    "Tracker": { "VehicleType": "SimpleFlight", "X": 0,  "Y": 0, "Z": 0, "Yaw": 0 },
    "Target":  { "VehicleType": "SimpleFlight", "X": 10, "Y": 0, "Z": 0, "Yaw": 180 }
  },
  "CameraDefaults": {
    "CaptureSettings": [
      { "ImageType": 0, "Width": 1280, "Height": 720, "FOV_Degrees": 35 },
      { "ImageType": 2, "Width": 1280, "Height": 720, "FOV_Degrees": 35 }
    ]
  }
}
```

`ImageType 0` = Scene，`ImageType 2` = DepthPlanar。改完必须**重启模拟器**才生效。

### 1.2 依赖

```bash
pip install ultralytics opencv-python huggingface_hub
pip install airsim
```

Cosys-AirSim 用户改为 `pip install -e <Cosys-AirSim>/PythonClient`（模块名是 `cosysairsim`，
`airsim_io.py` 会自动优先使用它）。

### 1.3 体检

```bash
python scripts/check_airsim_ready.py
```

逐项检查 CUDA、ultralytics、RPC 连接、双机存在、图像尺寸与配置是否一致、深度图是否可用、
分割 ID 是否能挂到目标网格上。任何一项 FAIL 都会给出具体修法并以非零码退出。

---

## 2. 阶段 1：数据集采集（零人工标注）

### 2.1 为什么不手工标注

手标 3000 张图要几十小时，而 AirSim 知道每个物体的真实位置。我们用两条互相独立的路径生成标签：

**路径 A：分割掩码（首选，像素级精确）**

1. `simSetSegmentationObjectID("[\w]*", 0, True)` 把整个场景刷成 ID 0；
2. `simSetSegmentationObjectID("Target[\w]*", 25, True)` 只把目标机刷成 ID 25；
3. 抓分割图，**取图像边缘环的众数颜色作为背景色**，其余像素即目标；
4. 用 `cv2.connectedComponentsWithStats` 取最大连通域，去掉散点；
5. 掩码的外接矩形就是精确的 bbox。

用边缘环而不是全图众数，是为了让目标占满画面时仍然判对——这是近距离样本的常见情况。

**路径 B：三维包围盒投影（交叉校验）**

用 `simGetCameraInfo` 的相机位姿和 `simGetVehiclePose` 的目标位姿，把目标的 8 个立方体角点
按针孔模型投影到像平面：

```
p_cam = R_cam^T (p_world - t_cam)          AirSim 相机系：+x 前，+y 右，+z 下
f     = (W/2) / tan(FOV/2)
u     = W/2 + f * y_cam / x_cam
v     = H/2 + f * z_cam / x_cam
```

**采集器同时跑两条路径，IoU < 0.25 就丢弃这一帧。** 这能自动过滤掉遮挡、分割 ID 串到别的物体、
目标半出画等脏样本。这个交叉校验是数据质量的唯一防线——因为标注错了模型不会报错，只会悄悄变差。

### 2.2 域随机化

每一帧都随机化，让模型学到"无人机的结构"而不是"某个固定背景里的某个色块"：

| 维度 | 范围 | 意图 |
|---|---|---|
| 距离 | 3–45 m | 覆盖从占满画面到只有十几像素 |
| 方位角 / 俯仰角 | ±15° / ±8.5° | 覆盖画面中心到边缘（FOV 35° 的边界内） |
| 目标姿态 | yaw ±180°，roll/pitch ±25° | 各种机动姿态下的外形 |
| Tracker 位置 | XY ±120 m，高度 4–50 m | 天空 / 地面 / 建筑等不同背景 |
| 天气 | 25% 概率，雨雪雾尘 ≤0.45 | 能见度退化 |
| 时间 | 35% 概率，6–20 时 | 光照和阴影方向 |
| 负样本 | 4% | 把目标挪到相机背后，纯背景帧，压低误检 |

采集时调用 `simPause(True)` 冻结物理，位姿设定后画面不会因为重力下坠而漂移。

### 2.3 执行

```bash
python scripts/list_scene_objects.py --regex ".*(Target|Drone|Flying).*"
python scripts/collect_dataset.py --config config/dataset_config.json --samples 3000
```

第一条先确认目标网格的真实名字，把它填进 `config/dataset_config.json` 的
`segmentation.target_regex`。默认 `Target[\w]*` 在多数环境下可用，但不同 UE 场景命名会变。

采集完**务必打开 `datasets/airsim_drone/preview/` 目视检查那 24 张带框图**。这一步花两分钟，
能挡住后面几小时的无效训练。同时看 `collection_stats.json` 里的拒绝计数：

- `rejected_cross_check` 偏高 → 两条标注路径不一致，多半是 `target_extents_m` 不对或分割 ID 串了；
- `rejected_no_mask` 偏高 → 分割正则没匹配到目标；
- `rejected_small` 偏高 → 距离上限设得太远。

3000 张在 20 Hz 左右的采集速率下大约 5–8 分钟。

---

## 3. 阶段 2：模型训练（含 Hugging Face 开源模型）

### 3.1 四级递进

每一级都从上一级热启动，而不是各练各的：

| 级 | 权重来源 | 学到什么 |
|---|---|---|
| 0 | `yolov8s.pt` | COCO 通用视觉特征 |
| 1 | HF 开源无人机模型 / Seraphim 数据集 | "无人机"这个类别的泛化先验 |
| 2 | AirSim 采集帧 | 仿真域里这架目标机的具体外形 |
| 3 | 实飞录像 | 真实域（迁移实机时做，见第 6 节） |

### 3.2 Hugging Face 资源（已核实文件与许可）

**数据集（价值最高）**

- [`lgrzybowski/seraphim-drone-detection-dataset`](https://huggingface.co/datasets/lgrzybowski/seraphim-drone-detection-dataset)
  — 83,483 张（75,134 train / 8,349 test），单类 `drone`，**YOLO 格式**，640×640，**CC BY 4.0**。
  由 23 个开源数据集整合而成，直接可训。
  注意：混有营销图和 CG 合成图，官方自己也提示这会影响真实场景泛化。**适合做第 1 级先验，不适合当终点。**

**预训练权重**

- [`doguilmak/Drone-Detection-YOLOv11x`](https://huggingface.co/doguilmak/Drone-Detection-YOLOv11x)
  — **MIT 许可**，文件 `weight/best.pt`。P 0.922 / R 0.831 / mAP50 0.905 / mAP50-95 0.546，~8.9 ms/img。
- [`doguilmak/Drone-Detection-YOLOv8x`](https://huggingface.co/doguilmak/Drone-Detection-YOLOv8x)
  — 文件同为 `weight/best.pt`，**仓库未声明许可证**，商用或对外发布前需要先确认授权。

```bash
python scripts/fetch_hf_assets.py --weights --which yolov11x
python scripts/fetch_hf_assets.py --dataset --dataset-dir E:/datasets/seraphim   # 数十 GB
```

### 3.3 为什么不直接用 HF 权重收工

AirSim 的渲染域和真实照片域差距很大（材质、光照模型、纹理细节、运动模糊特性都不同），
现成权重在 AirSim 画面上召回率通常明显下降。反过来，只用 AirSim 数据训练又会过拟合到那一架
特定型号的目标机。所以两者都要：**HF 给类别泛化，AirSim 给域适配。**

另外 HF 权重还有个务实用途：拿它对 AirSim 采集帧跑一遍推理，和自动标签比 IoU。
两者高度一致说明标注管线没问题，这是一次几乎免费的独立验证。

### 3.4 训练

```bash
python scripts/train_yolo.py \
  --data datasets/airsim_drone/data.yaml \
  --base weights/hf/yolov11x/weight/best.pt \
  --name drone_sim_ft --epochs 120 --imgsz 960
```

关键超参和理由：

- `imgsz 960` — 远距离目标只有十几像素，降到 640 会直接丢掉小目标召回；
- `flipud 0.0` — 无人机是重力对齐的，上下翻转会造出物理上不存在的样本；
- `scale 0.6` — 尺度增强放宽，因为真实追踪中目标尺寸跨度就是这么大；
- `close_mosaic 10` — 最后 10 轮关掉 mosaic，让模型在接近推理时的真实构图上收敛。

训练结束会写出 `train_summary.json`（mAP50 / mAP50-95 / P / R）和 `best.pt` 路径。

---

## 4. 阶段 3：追踪闭环

### 4.1 控制律

归一化中心误差，`W`/`H` 为画面宽高，`(ĉx, ĉy)` 为**预测框**中心（不是原始检测框）：

```
e_x = (ĉx - W/2) / (W/2)          e_y = (ĉy - H/2) / (H/2)
```

三个通道解耦：

| 通道 | 控制量 | 公式 |
|---|---|---|
| 水平 | yaw rate | `ω = clip(k_yaw · e_x, ±ω_max)` |
| 垂直 | 机体 z 速度 | `v_z = clip(k_z · e_y, ±v_z_max)`（NED，+z 向下） |
| 前后 | 机体 x 速度 | `v_x = clip(k_size · (w_desired − ŵ/W), −v_back, +v_fwd)` |

距离控制默认**用框宽而不是深度图**（`use_depth_distance: false`）。原因很实际：螺旋桨和细支架
让深度中值极不稳定，框宽反而更平滑。深度路径保留着，取框内中心 1/4 区域的中位数并剔除天空的
超大平面深度值，需要时可以打开。

中心附近设 `center_deadzone = 0.035` 死区，否则控制器会围着中心持续抖动。

### 4.2 三层丢失恢复

1. **短时丢检**（< `max_prediction_gap_s` = 0.6 s）：预测器继续输出预测框，控制器照常追。
   绝大多数"丢失"其实只是单帧置信度掉到阈值以下，根本不该触发搜索。
2. **预测超时**：预测器返回 `None`，历史状态不再可信。
3. **扫描搜索**：云台或机体朝**目标最后出现的方向**扫描，直到重新检出。

`lost_grace_s = 0.3` 是第 1 层和第 3 层之间的缓冲：刚丢的一瞬间保持姿态不动，比立刻甩头找更容易接回。

### 4.3 运行

```bash
python scripts/run_tracking.py --config config/tracking_config.json \
  --weights runs/detect/drone_sim_ft/weights/best.pt --seconds 90
```

---

## 5. 阶段 4：运动预测

### 5.1 为什么需要

从"光子进相机"到"无人机改变速度"，链路上串着采集延迟、YOLO 推理延迟、Python 控制循环延迟、
以及 AirSim 速度指令的执行延迟。加起来 0.15–0.25 s。不补偿的话，控制器永远在追目标 0.2 秒前的位置，
目标一动就慢半拍，而且这个滞后会被反馈放大成振荡。

做法：控制器追的是 `horizon_s`（默认 0.2 s）之后的**预测位置**。

### 5.2 基线：常速度 Kalman

状态 8 维 `[cx, cy, w, h, vx, vy, vw, vh]`，观测 `[cx, cy, w, h]`。
把框宽高也纳入状态，是因为它们的变化率直接反映距离变化，能让距离通道也获得前馈。

### 5.3 升级：IMM 交互多模型

单一常速度模型有个绕不开的矛盾：过程噪声调小则急转时过冲，调大则跟着检测抖动跑。
IMM 并行跑三个假设，按后验似然加权融合：

| 模型 | 用途 | 过程噪声倍率 |
|---|---|---|
| `cv` | 平稳巡航 | 1.0 |
| `ca` | 加速 / 俯冲 | 4.0 |
| `maneuver` | 急转弯 | 25.0 |

统一用 12 维状态 `[位置4, 速度4, 加速度4]`，三个模型只是转移矩阵和过程噪声不同，
这样标准 IMM 的混合步骤可以直接套用，不需要跨维度的状态转换。

标准四步：模型条件混合 → 各模型 Kalman 预测更新 → 按似然更新模型概率 → 加权融合。
外推时**先各自外推再按模型概率融合**，而不是融合后再外推——否则加速度假设会污染 CV 的外推结果。

### 5.4 离线基准（已实测）

`tests/test_prediction_benchmark.py` 用合成轨迹 + 检测噪声（σ=4 px）+ 8% 丢检率，
评价"`horizon_s` 后的预测中心"与"真实位置"的误差。24 秒 @ 20 Hz，固定随机种子：

| 轨迹 | 无预测 | CV Kalman | IMM |
|---|---|---|---|
| 平滑巡航 RMSE | 15.55 px | 7.08 px | **6.20 px** |
| 平滑巡航 P95 | 24.90 px | 11.52 px | **10.39 px** |
| 急转机动 RMSE | 38.00 px | 35.09 px | **27.20 px** |
| 急转机动 P95 | 77.76 px | 61.73 px | **48.14 px** |

结论：预测在两类轨迹上都显著优于不预测；IMM 相对 CV 的增益主要体现在机动场景
（RMSE −22%，P95 −22%），平滑场景也没有付出代价（反而略好）。

### 5.5 选型

配置里切换，不改代码：

```json
"prediction": { "enabled": true, "model": "imm", "horizon_s": 0.2 }
```

默认仍是 `kalman`，因为它是已经在 AirSim 闭环中验证过的基线（p95 = 67.71 px）。
`imm` 需要在你的环境里跑完 `auto_test.py` 的 A/B 确认收益后再提为默认。

调参：目标平滑但控制慢半拍 → `horizon_s` 调到 0.25；急转时过冲 → 降到 0.15；
YOLO 框抖 → 增大 `measurement_noise`；跟不上机动 → 增大 `process_noise`。

### 5.6 后续方向

图像平面预测适合补偿短时延迟，不适合长时间无观测外推。往下可以做：
用框宽 + 相机内参估计相对距离形成 3D EKF；用 AirSim 轨迹日志训练 GRU 处理复杂机动模式；
把预测器接进 MPC 做前瞻轨迹规划。

---

## 6. 阶段 5：自动化测试

### 6.1 为什么是脚本而不是点击

AirSim 完全由 RPC 驱动。脚本能做到人工点击做不到的三件事：目标轨迹**逐帧可复现**、
同一条件下跑**开/关预测的 A/B**、以及**回归门禁**（阈值不过就非零退出）。

### 6.2 一条命令

```bash
python scripts/auto_test.py --plan config/auto_test_plan.json
```

先跑 `check_airsim_ready.py` 体检，然后跑 `3 种目标轨迹 × 3 种预测配置 = 9` 组 90 秒闭环：

- 轨迹：`front_sweep`（平滑）、`figure8`（连续转向）、`lateral_dash`（急停急转，最难）
- 配置：`no_prediction` / `kalman` / `imm`

输出 `report.md`（带相对基线的百分比变化）和 `report.json`。

空工程可以一条命令走完全流程：

```bash
python scripts/auto_test.py --plan config/auto_test_plan.json --collect --train
```

### 6.3 验收阈值

```json
{ "visible_rate_min": 0.98, "lost_count_max": 0, "center_error_p95_px_max": 67.71 }
```

`67.71` 是既有基线的实测值，用它做回归红线：新改动只要把 p95 抬上去就算退化。

### 6.4 离线测试（不需要模拟器）

```bash
python tests/run_all.py
```

覆盖：预测器行为、控制律符号与限幅、丢失恢复、自动标注数学、预测器 A/B 基准。
CI 里跑这个，AirSim 闭环留给本地。

---

## 7. 阶段 6：迁移到实机

### 7.1 架构上已经准备好的部分

只有 `airsim_io.py` 接触仿真器。实机上写一个 `mavlink_io.py` 实现同样的函数签名即可：

| 仿真 | 实机 |
|---|---|
| `connect()` | `mavutil.mavlink_connection('/dev/ttyTHS1', baud=921600)` |
| `get_scene_and_depth()` | GStreamer / V4L2 取帧（深度通常没有，传 `None`，控制器会自动走框宽路径） |
| `move_body_velocity()` | MAVLink `SET_POSITION_TARGET_LOCAL_NED`（`MAV_FRAME_BODY_NED`，只启用速度位 + yaw_rate） |
| `arm_and_takeoff()` | `MAV_CMD_COMPONENT_ARM_DISARM` + `MAV_CMD_NAV_TAKEOFF` |
| `set_camera_gimbal()` | MAVLink `MAV_CMD_DO_MOUNT_CONTROL` 或云台 SDK |

检测、预测、控制三层一行都不用改。

### 7.2 必须重做的部分

1. **模型域适应**。仿真训练的权重直接上真机会掉点。用真机录像重新做第 3 级微调，
   数据可以先用 HF 的 Seraphim 兜底，再补自己场地的实拍。
2. **推理加速**。`model.export(format='engine', half=True)` 导出 TensorRT。
   Jetson Orin NX 上 YOLOv8s @ 960 大约能到 30–45 FPS；YOLOv11x 太重，实机应降到 s/m 级。
3. **相机标定**。真实镜头有畸变，`focal_px()` 的理想针孔假设不再成立，
   需要用棋盘格标出内参和畸变系数，先去畸变再送检测。
4. **时间同步**。仿真里 `now_s()` 就够了，实机上图像时间戳和飞控时间戳是两个时钟，
   Kalman 的 `dt` 必须用图像采集时刻，否则预测会系统性偏移。

### 7.3 安全（不可省略）

- 地理围栏 + 最大距离限制，超出立即切回 LOITER；
- 速度限幅在**飞控侧**也设一份，不能只靠 Python 层的 `clip`；
- 看门狗：控制循环超过 200 ms 没有新指令，飞控自动悬停；
- 失控保护：连续丢检超过 `max_prediction_gap_s` 且扫描 N 秒无果，退出追踪模式；
- 遥控器随时可一键夺回控制权，这一条优先级最高。

### 7.4 验证顺序

```
离线回归 (tests/run_all.py)
  -> AirSim 闭环 A/B (auto_test.py)
  -> HIL：真飞控 + 仿真图像，验证 MAVLink 链路
  -> 系留测试：绳系限位，只验证检测和指令方向
  -> 开阔场地低速：限速 2 m/s，限距 15 m
  -> 正式测试
```

每一级都用同一套 `metrics.py` 指标，这样仿真和实机的数字可以直接对比，
一旦某级掉点就能定位到是哪一层引入的差异。

---

## 8. 命令速查

```bash
# 体检
python scripts/check_airsim_ready.py

# 找目标网格名（分割正则匹配不上时）
python scripts/list_scene_objects.py --regex ".*(Target|Drone|Flying).*"

# 拉 HF 资源
python scripts/fetch_hf_assets.py --weights --which yolov11x

# 采集数据集（自动标注）
python scripts/collect_dataset.py --samples 3000

# 训练
python scripts/train_yolo.py --data datasets/airsim_drone/data.yaml \
  --base weights/hf/yolov11x/weight/best.pt --name drone_sim_ft

# 单次追踪
python scripts/run_tracking.py --config config/tracking_config.json \
  --weights runs/detect/drone_sim_ft/weights/best.pt --seconds 90

# 全自动 A/B 回归
python scripts/auto_test.py --plan config/auto_test_plan.json

# 离线测试（无需 AirSim）
python tests/run_all.py
```
