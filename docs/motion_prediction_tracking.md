# 运动预测增强追踪说明

## 算法选择

本版本使用图像平面 Kalman Filter，而不是直接使用 LSTM/Transformer。原因是当前追踪闭环的真实输入就是 YOLOv8 检测框，图像平面预测能直接补偿检测、采集和控制延迟，参数少、可解释、调试成本低。

状态量：

```text
[cx, cy, w, h, vx, vy, vw, vh]
```

其中 `cx/cy` 是检测框中心，`w/h` 是框宽高，后四个量是对应速度。每帧先根据时间差预测，再用 YOLO 检测框更新；控制器最终追踪 `horizon_s` 秒后的预测框。

## 配置项

`config/tracking_config.json` 中新增：

```json
"prediction": {
  "enabled": true,
  "horizon_s": 0.2,
  "max_prediction_gap_s": 0.6,
  "process_noise": 80.0,
  "measurement_noise": 25.0
}
```

- `enabled`：是否启用预测，关闭后用于 A/B 对照。
- `horizon_s`：向未来预测的时间，默认 `0.2s`。
- `max_prediction_gap_s`：短时丢检时继续预测的最长时间，超过后返回 `None`，让原搜索逻辑接管。
- `process_noise`：目标运动变化噪声，越大越相信目标可能突然加速。
- `measurement_noise`：检测框观测噪声，越大越不容易被单帧抖动带偏。

## 接入方式

生产追踪脚本中的顺序应为：

```text
YOLO detect
  -> predictor.step(...)
  -> controller.command(prediction.detection, ...)
  -> move Tracker
  -> write CSV prediction fields
```

`predictor.step()` 返回的 `PredictionResult.detection` 类型仍然是 `Detection`，所以现有控制器不需要改接口。

## 日志字段

CSV 增加：

- `prediction_used`
- `predicted_center_x`
- `predicted_center_y`
- `raw_center_x`
- `raw_center_y`
- `prediction_age_s`

这些字段用于判断当前控制是基于原始检测，还是基于短时预测。

## 验证方法

静态检查：

```powershell
cd E:\airsim_yolov8_drone_tracker
python -m compileall src scripts tests
python tests\test_predictor.py
```

在完整 AirSim 项目中验证：

```powershell
python scripts\check_airsim_ready.py
python scripts\run_tracking.py --config config\tracking_config.json --weights D:/sim/runs/drone_tracker/detect/drone_centered_yolov8s_v2/weights/best.pt --seconds 90
```

A/B 对照：把 `prediction.enabled` 改为 `false` 再跑同样 90 秒，比较：

- `visible_rate`
- `lost_count`
- `center_error_mean_px`
- `center_error_p95_px`

验收目标：

- 启用预测后 `visible_rate >= 0.98`
- `lost_count == 0`
- `center_error_p95_px` 不高于原基线 `67.71px`

