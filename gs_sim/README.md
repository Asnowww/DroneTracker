# gs_sim — 3DGS 世界模型模拟器工作区

本目录存放 GS-Sim（gsplat 渲染 + RotorPy 动力学）的**资产与阶段产物**。
完整方案见 [../docs/gs_sim_plan.md](../docs/gs_sim_plan.md)，执行进度见 [STATE.json](STATE.json)。

## 目录约定

| 路径 | 内容 |
|---|---|
| `assets/scenes/` | 场景高斯模型 `.ply`（真实重建场景） |
| `assets/drone/` | 目标无人机高斯模型 `target.ply` |
| `scripts/` | 各阶段执行脚本 + `verify_phaseN.py` 门禁脚本 |
| `cache/` | 下载与中间产物（可安全删除，会自动重建） |
| `runs/` | GS-Sim 的追踪日志与回归报告 |

**代码不放这里。** 三个新模块作为 `airsim_io.py` 的同级兄弟放进
`src/drone_tracker/`，这样后端切换才干净：

```
src/drone_tracker/
├── airsim_io.py      现有后端
├── gs_renderer.py    新增：gsplat 渲染（场景 ∪ 目标机高斯，单次光栅化）
├── gs_dynamics.py    新增：RotorPy 四旋翼动力学
└── gs_sim_io.py      新增：与 airsim_io 同名同签名的接口层
```

## 迁移边界（实测，非估计）

`src/drone_tracker/` 各模块对 AirSim 的引用次数：

| 模块 | airsim 引用 | 迁移工作量 |
|---|---|---|
| `controller.py` · `predictor.py` · `imm_predictor.py` · `prediction.py` · `detector.py` · `metrics.py` · `utils.py` · `config.py` | **0** | 零改动 |
| `labeling.py` | 1（仅文档字符串） | 零改动 |
| `target_policy.py` | 5（2 处函数内惰性导入） | 轨迹数学零改动，仅执行器需适配 |
| `airsim_io.py` | 47 | 由 `gs_sim_io.py` 平行替代 |

离线测试套件 `tests/run_all.py` 全程不需要模拟器即可通过——这是控制/预测栈
与仿真后端完全解耦的既有证据。

## 后端切换

`config/tracking_config.json` 增加：

```json
{ "backend": "gs" }
```

`run_tracking.py` / `collect_dataset.py` / `auto_test.py` 据此选择导入
`airsim_io` 或 `gs_sim_io`，其余逻辑与指标口径完全一致，两个后端的
回归数字可直接对比。

## 恢复执行

任何新会话中，Claude 读取 `STATE.json` 的 `current_phase` 与各阶段
`status`，从第一个非 `passed` 的阶段继续。门禁脚本是幂等的，重复运行安全。
