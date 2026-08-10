# 路线 B 详细方案：FiGS 式 3DGS 无人机追踪模拟器（Claude 全程自主执行版）

> 本文档是执行契约：每个阶段都有机器可判定的验收门禁，Claude 按阶段推进，
> 门禁不过不进入下一阶段。用户预期操作次数：0–1 次（见 §12）。

---

## 1. 目标与最终验收

构建 `GS-Sim`：gsplat 光栅化真实重建场景 + RotorPy 四旋翼动力学 + 现有
检测/预测/控制栈，全部运行在**单 Python 进程**内（零 RPC）。

最终验收（与现有 AirSim 回归同口径，直接可比）：

| 指标 | 门槛 |
|---|---|
| 控制环路频率 | ≥ 20 Hz（目标 30 Hz） |
| front_sweep / lateral_dash | visible ≥ 0.98，lost = 0，p95 ≤ 67.71 px |
| **figure8（标准门禁，不再是压力项）** | visible ≥ 0.98，lost = 0，p95 ≤ 67.71 px |
| 场景 | ≥ 2 个真实重建场景 + Blocks 对照 |
| 检测模型 | 在 GS 域重训后 mAP50 ≥ 0.98 |

figure8 达标的依据：现有失败根因是 6 Hz 采样 + 55°/s 偏航限幅的角速度余量不足；
30 Hz 下同样的控制律余量翻 5 倍，且 IMM 在 6 Hz 时已达 visible 0.979。

## 2. 总体架构

```
┌────────────────────── 单 Python 进程 ──────────────────────┐
│  RotorPy 动力学 (400Hz 内步进)                              │
│    ├─ Tracker 四旋翼（速度指令接口 + 风扰/参数随机化）        │
│    └─ Target 四旋翼（follow_target 参考轨迹，同款动力学）     │
│  gsplat 渲染器                                             │
│    ├─ 场景高斯（静态，1–2M）                                │
│    └─ 目标机高斯（50–200k，按 Target 位姿刚体变换后拼接）——   │
│        单次光栅化，遮挡关系自动正确，无需独立合成器            │
│  gs_sim_io.py  ←— 与 airsim_io.py 同名同签名接口             │
│  ────────────── 以下四层零改动 ──────────────               │
│  detector.py / prediction(kalman+IMM) / controller.py       │
│  run_tracking.py / collect_dataset.py / auto_test.py        │
└────────────────────────────────────────────────────────────┘
```

**接口契约**（`gs_sim_io.py` 必须实现的函数，签名与 `airsim_io.py` 一致）：
`connect() -> GsSimClient`、`require_vehicles`、`arm_and_takeoff`、`hover_lock`、
`teleport_still`、`get_scene_and_depth(with_depth)`、`get_scene_and_segmentation`、
`move_body_velocity`、`set_camera_gimbal`、`vehicle_pitch_deg`、`pose_xyz_yaw`、
`get_target_instance_colors`（GS 域返回渲染器直出的目标掩码，精度高于颜色匹配）。
`GsSimClient` 鸭子类型模拟被用到的 client 方法（`simGetVehiclePose`、
`simSetCameraFov`、`simEnableWeather` 空实现等）。后端切换：
`tracking_config.json` 增加 `"backend": "airsim" | "gs"`，`run_tracking` 按此选择导入。

**关键设计决策：目标机也用高斯表示**。动态物体 = 场景高斯 ∪ T(pose)·无人机高斯，
拼接后一次光栅化。避开 mesh 渲染器（nvdiffrast）+ 深度合成 + 光照匹配三个难点；
200k 高斯的每帧刚体变换在 GPU 上 <1 ms。

## 3. 全自主执行设计

- **幂等脚本**：每步可重复执行，已完成的产物跳过（哈希/存在性检查）。
- **机器可判定门禁**：每阶段一个 `scripts/gs/verify_phaseN.py`，输出
  `PHASE_N_PASS=true|false` + 指标 JSON，非零退出即阻断。
- **状态文件**：`gs_sim/STATE.json` 记录各阶段状态与产物路径；
  Claude 任何新会话读此文件即可续跑（配合项目记忆）。
- **视觉 QC 自动化**：需要人眼的检查（渲染质量、合成痕迹、floater）由脚本生成
  拼图 → Claude 读图判定 → 结论写回 STATE.json。此模式已在数据集审计中验证。
- **长任务**：训练/下载全部后台运行 + Monitor 监控关键行（loss、PSNR、ETA、OOM）。
- **失败回退**：§11 的每个风险都预置了自动回退动作，Claude 按表执行不请示。

## 4. 阶段 0：工具链自检与安装（预计 0.5–1 天）

1. 检测 MSVC 工具链（`cl.exe`）与 CUDA Toolkit nvcc；缺失 → `winget install`
   VS 2022 Build Tools（**唯一可能的 UAC 弹窗，见 §12**）。
2. `pip install gsplat` —— JIT 编译对 torch 2.11+cu128；失败回退链：
   ① 指定官方预编译 wheel 索引 → ② 降 gsplat 版本 → ③ WSL2 环境。
3. `pip install rotorpy trimesh`（纯 Python，无编译风险）。
4. 冒烟基准：合成 10 万随机高斯渲染 720p，记录 FPS 基线。
5. **门禁 0**：gsplat 渲染出图 + FPS ≥ 100 + rotorpy 悬停仿真 10 s 高度误差 < 0.1 m。

## 5. 阶段 1：场景资产自动获取（预计 1 天，与阶段 0 并行下载）

双源策略，全程无需拍摄：

- **快线**：Inria 官方**预训练模型包**（Mip-NeRF360 等场景的现成 .ply）——
  下载即用，跳过重建训练，当天出第一个真实场景。
- **慢线**：Mip-NeRF360 数据集（自带 COLMAP 位姿，**无需运行 COLMAP**）→
  gsplat `simple_trainer` 自训 2 个场景（garden/bicycle 级别，16 GB 显存
  降采样到 1080p 可训）→ 验证我们掌握完整重建管线（后续实地场景要用）。
- 清理：floater 自动剔除（按不透明度/尺度阈值 + 离群聚类），生成前后对比拼图供 Claude 判读。
- **门禁 1**：≥2 个场景在自研渲染器里 PSNR ≥ 26（对留出视角），720p 渲染 ≥ 80 FPS。

## 6. 阶段 2：目标机高斯化（预计 1–2 天）

用现有 AirSim 当"扫描台"，全自动：

1. 启动 Blocks（已有脚本），Target 悬停于高空纯天空背景；
2. 相机以已知位姿环绕采样 120–200 视角（球面均匀 + 上下半球），
   同帧抓 RGB + 实例分割掩码（管线现成）；
3. 掩码外像素置白 → 带已知位姿的目标物数据集（无需 COLMAP）；
4. gsplat 训练小模型（目标 50–150k 高斯），中心化 + 尺度归一到真实尺寸 0.43 m；
5. 合成检验：无人机高斯变换到场景中 10 个随机位姿渲染，Claude 读拼图判定
   遮挡正确性与视觉融合度。
- **门禁 2**：合成渲染中目标清晰可辨、遮挡关系正确、渲染掩码与投影 bbox IoU ≥ 0.6。

## 7. 阶段 3：GS-Sim 核心开发（预计 3–5 天，核心工作量）

1. `src/drone_tracker/gs_renderer.py`：场景加载、无人机高斯变换拼接、
   相机模型（FOV 35°、云台偏转 = 相机外参旋转，含俯仰稳像语义）、
   RGB/深度/目标掩码三输出。
2. `src/drone_tracker/gs_dynamics.py`：RotorPy 封装。速度指令 → SE3 控制器；
   实现 `hover_lock`/`teleport_still`/`arm_and_takeoff` 语义（GS 域内传送即瞬移，
   无 Cosys 的渲染脱钩问题）。**机体俯仰角来自动力学真值**——俯仰-相机耦合
   在这里是自然涌现的（真实倾转产生真实画面偏移），稳像逻辑照旧工作。
3. `src/drone_tracker/gs_sim_io.py`：按 §2 契约拼装。
4. 时钟设计：动力学 400 Hz 定步长内循环，感知环按需步进（保持确定性回放能力，
   支持 seed 完全复现——比 AirSim 的实时时钟更适合回归）。
5. 单元测试平移：控制律/预测器测试本就离线；新增 `test_gs_sim.py`
   （动力学悬停/阶跃响应、渲染确定性、掩码正确性）。
- **门禁 3**：`run_tracking.py --backend gs` 跑通 45 s front_sweep，
  环路 ≥ 20 Hz，visible ≥ 0.9（用现 Blocks 模型，允许域差掉点）。

## 8. 阶段 4：数据与模型平移（预计 1–2 天）

1. `collect_dataset.py --backend gs`：域随机化平移（位姿、距离对数均匀、
   多场景轮换；天气随机化替换为光照增益/噪声/运动模糊增广）。
   标注 = 渲染器直出掩码，四层质量防线中的交叉校验自动满足。
2. 每场景 1500 帧 × 3 场景，YOLOv11x 继续微调（Blocks 权重热启动）。
3. **门禁 4**：GS 域验证集 mAP50 ≥ 0.98；混合验证集（GS + Blocks）不低于 0.95。

## 9. 阶段 5：性能调优与 figure8 达标（预计 2–3 天）

预算表（720p，RTX 5060 Ti）：

| 环节 | 预算 |
|---|---|
| gsplat 渲染（场景+目标） | ≤ 8 ms |
| YOLO 640 half | ≤ 20 ms |
| 动力学步进 + 控制律 | ≤ 2 ms |
| 合计 → 环路 | ≤ 33 ms ≈ 30 Hz |

调优杠杆（按序启用直到达标）：渲染分辨率 960×540、YOLO TensorRT 导出
（`model.export(format='engine')`，实机部署也要用）、高斯 LOD 裁剪、CUDA stream
重叠渲染与推理。随后按 30 Hz 重整定增益（k 值×采样率补偿，预期只需微调）。
- **门禁 5**：figure8 × 3 预测器全部达到标准门禁（§1 表）。

## 10. 阶段 6：最终回归与交付（预计 1 天）

1. `auto_test.py --backend gs`：3 场景 × 3 轨迹 × 3 预测器（27 组，90 s）；
2. AirSim(Blocks) 同版本对照矩阵，量化"视觉域升级对追踪指标的影响"；
3. 交付：回归报告、GS-Sim 使用文档、`development_plan.md` 更新实机迁移章
   （GS 场景 = 实飞场地数字孪生的操作指引）、录屏样例帧。

## 11. 风险与自动回退矩阵

| 风险 | 探测方式 | 自动回退 |
|---|---|---|
| gsplat Windows 编译失败 | pip 安装门禁 | wheel 索引 → 降版本 → WSL2 |
| VS Build Tools 需 UAC | winget 返回码 | 暂停并明确告知用户点一次"是"（§12） |
| 预训练包链接失效/网速 | 下载超时重试 3 次 | 切换镜像 → 慢线自训场景顶替 |
| 16 GB 训练大场景 OOM | 训练监控 OOM 行 | 降采样 → 减高斯上限 → 换小场景 |
| 目标机 splat 质量差（薄结构） | 门禁 2 拼图判读 | 增视角数 → 提训练迭代 → 回退 trimesh 程序化 mesh + nvdiffrast 路线 |
| RotorPy 接口不符预期 | 门禁 0 悬停测试 | 自研二阶动力学 + 一阶姿态环（半天工作量，规格已定义） |
| 30 Hz 不达 | 阶段 5 预算表逐项 profiling | 杠杆序列用尽后接受 20 Hz 并按 20 Hz 重整定（figure8 余量仍 ×3.3） |

## 12. 用户触点清单

| 触点 | 概率 | 内容 |
|---|---|---|
| UAC 弹窗点"是" | 仅当机器无 MSVC 工具链 | VS Build Tools 安装授权，一次性 |
| 其余全部 | — | 无。下载、训练、开发、测试、QC 判读、回归、文档全部由 Claude 执行 |

## 13. 里程碑总览

| 里程碑 | 累计工期 | 标志 |
|---|---|---|
| M0 工具链就绪 | 1 天 | 门禁 0 |
| M1 首个真实场景渲染 | 2 天 | 门禁 1 |
| M2 目标机合成可用 | 3–4 天 | 门禁 2 |
| M3 GS 闭环首飞 | 7–9 天 | 门禁 3 |
| M4 模型域适配完成 | 9–11 天 | 门禁 4 |
| M5 figure8 达标 | 12–14 天 | 门禁 5 |
| M6 27 组回归交付 | 13–15 天 | 最终报告 |
