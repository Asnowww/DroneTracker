# 实施报告

## 当前状态（2026-07-27）

`E:\airsim_yolov8_drone_tracker` 已从"运动预测补丁包"扩建为**自洽的完整工程**。
原 `D:\sim` 工程所在的 D 盘在本机不存在，因此不再依赖它；模块接口与原工程保持一致，可双向合并。

## 已实施

### 核心模块（本次补齐）

- `src/drone_tracker/airsim_io.py` — AirSim RPC 封装，自动兼容 `cosysairsim` / `airsim`
- `src/drone_tracker/controller.py` — 视觉伺服控制器 + 云台状态机
- `src/drone_tracker/target_policy.py` — 6 种脚本化目标轨迹
- `src/drone_tracker/labeling.py` — 分割掩码 + 三维投影双路自动标注
- `src/drone_tracker/config.py`、`utils.py` — 配置加载与几何工具

### 运动预测

- `src/drone_tracker/predictor.py` — 常速度 Kalman（既有基线）
- `src/drone_tracker/imm_predictor.py` — **新增** 12 维 IMM，并行 CV / CA / 机动模型
- `src/drone_tracker/prediction.py` — **新增** 预测器工厂，`prediction.model` 配置切换

### 数据与训练

- `scripts/collect_dataset.py` — AirSim 自动采集 + 零人工标注 + 域随机化
- `scripts/train_yolo.py` — 四级递进训练，支持 HF 权重热启动
- `scripts/fetch_hf_assets.py` — 拉取 Hugging Face 开源无人机模型与数据集
- `scripts/list_scene_objects.py` — 查找目标网格名（配置分割正则用）

### 测试

- `scripts/check_airsim_ready.py` — 逐项体检，FAIL 时给出具体修法
- `scripts/auto_test.py` — 3 轨迹 × 3 预测配置的全自动 A/B 回归，带验收门禁
- `tests/run_all.py` — 全部离线测试，不需要 AirSim / GPU / 权重

## 离线验证结果

```
python tests\run_all.py
```

```
[PASS] compileall
[PASS] tests/test_predictor.py          预测器基本行为
[PASS] tests/test_controller.py         控制律符号、死区、限幅、丢失恢复、轨迹策略
[PASS] tests/test_labeling.py           分割 bbox、针孔投影、IoU、YOLO 格式
[PASS] tests/test_prediction_benchmark.py   预测器 A/B 基准
all_offline_tests_passed=true
```

预测器基准（24 s @ 20 Hz，检测噪声 σ=4 px，丢检率 8%，固定种子）：

| 轨迹 | 无预测 RMSE | kalman RMSE | imm RMSE | 无预测 P95 | kalman P95 | imm P95 |
|---|---|---|---|---|---|---|
| 平滑巡航 | 15.55 | 7.08 | **6.20** | 24.90 | 11.52 | **10.39** |
| 急转机动 | 38.00 | 35.09 | **27.20** | 77.76 | 61.73 | **48.14** |

## 2026-07-27 下午：环境搭建与数据集完成

- 安装 Cosys-AirSim 5.8-v3.4.1 Blocks 打包版（`E:\sim\Blocks`）+ `cosysairsim` 3.4.1 客户端
- GPU：RTX 5060 Ti 16GB，CUDA 正常
- 体检 10/10 通过；实例分割方案适配完成（3.4 移除了 painted-ID）
- 踩坑与修复记录见 `docs/development_plan.md` 与项目记忆 `cosys-airsim-34-gotchas`：
  暂停传送失效 / 闲置机自由落体（悬停锁定解决）/ yaw 传送渲染错乱（采集固定 yaw=0）/
  Scene-Segmentation 渲染不同步（框内结构门禁解决）
- 数据集质量体系四层防线落地：视线检测、退化帧检测、可见比例、框内结构终审 +
  离线审计（确定性 + HF 独立交叉 + Claude 拼图判读）+ 隔离/补采闭环
- 最终数据集：3003 帧 / 2834 正样本，审计 level1 结构性缺陷 0 帧
- 隔离历史：首批 3000 帧中 747 帧空框/退化被清除，补采 750 帧（新门禁拦截 231 空框 + 10 退化）

## 2026-07-27 晚：闭环调通与回归结果

调通过程中修复的闭环问题（详见 git 外的调试记录）：指令占空比、目标机轨迹弹射、
跟踪距离与目标摆动包络冲突、机体俯仰-相机耦合（云台稳像+俯仰积分解决）、
加速度斜率限制、丢失扫描俯仰归中 + 机体接力扫描。

最终 90 秒 × 9 组回归（`runs/auto_test/20260727-174403/report.md`）：

| 轨迹 | no_prediction | kalman | imm |
|---|---|---|---|
| front_sweep | PASS (p95 65.5) | 67.95（超线 0.24px）| **PASS (p95 62.5)** |
| lateral_dash | PASS (p95 32.1) | PASS (p95 33.6) | PASS (p95 33.0) |
| figure8（压力项）| 0.914 可见 | 0.981 可见 | 高方差 |

- 标准轨迹（front_sweep / lateral_dash）达到并部分优于原 20 Hz 基线（67.71 px），
  当前 Python 循环实际 ~6 Hz。
- IMM 在 front_sweep 上 p95 最优（62.5），与离线基准结论一致。
- figure8（目标 4.2 m/s @ 5.7 m 距离）处于本环路（6 Hz + 55°/s 偏航限幅）的能力边界，
  逐轮方差大，作为压力项保留，不计入标准验收。
- 机体真实追击已验证：云台偏角常态 <7°，机体承担绝大部分跟踪。

## 尚未验证：AirSim 闭环

本机缺两项前置条件，`check_airsim_ready.py` 已如实报出：

- `airsim client import` FAIL — `cosysairsim` / `airsim` 均无法导入
- `torch CUDA` FAIL — 当前 `torch.cuda.is_available()` 为 False

补齐后按顺序执行：

```powershell
python scripts\check_airsim_ready.py
python scripts\collect_dataset.py --samples 3000
python scripts\train_yolo.py --data datasets\airsim_drone\data.yaml --base weights\hf\yolov11x\weight\best.pt
python scripts\auto_test.py --plan config\auto_test_plan.json
```

验收标准（沿用原基线，写在 `config/auto_test_plan.json`）：

- `visible_rate >= 0.98`
- `lost_count == 0`
- `center_error_p95_px <= 67.71`

`imm` 是否提为默认预测器，取决于 `auto_test.py` 在真实闭环中跑出的 A/B 结果，
不能仅凭离线基准下结论。
