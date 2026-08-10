from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "auto_tracking_algorithm.pdf"
FONT = Path(r"C:\Windows\Fonts\Deng.ttf")
BOLD_FONT = Path(r"C:\Windows\Fonts\Dengb.ttf")


def register_fonts() -> tuple[str, str]:
    if not FONT.exists() or not BOLD_FONT.exists():
        raise RuntimeError("Chinese fonts not found under C:\\Windows\\Fonts")
    pdfmetrics.registerFont(TTFont("CN", str(FONT)))
    pdfmetrics.registerFont(TTFont("CN-Bold", str(BOLD_FONT)))
    return "CN", "CN-Bold"


def p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def bullets(items: list[str], style: ParagraphStyle) -> ListFlowable:
    return ListFlowable(
        [ListItem(Paragraph(item, style), bulletColor=colors.HexColor("#2f5d8c")) for item in items],
        bulletType="bullet",
        start="circle",
        leftIndent=16,
    )


def code(text: str, style: ParagraphStyle) -> Preformatted:
    return Preformatted(text.strip(), style)


def soft_break(text: str) -> str:
    return (
        text.replace("/", "/ ")
        .replace("\\", "\\ ")
        .replace("_", "_ ")
        .replace(".", ". ")
    )


def table(rows: list[list[str]], widths: list[float], font: str, bold: str) -> Table:
    header_style = ParagraphStyle("TableHeader", fontName=bold, fontSize=8.8, leading=11, textColor=colors.HexColor("#1f3b57"))
    cell_style = ParagraphStyle("TableCell", fontName=font, fontSize=8.3, leading=10.5)
    wrapped = []
    for row_index, row in enumerate(rows):
        style = header_style if row_index == 0 else cell_style
        wrapped.append([Paragraph(soft_break(str(cell)), style) for cell in row])
    t = Table(wrapped, colWidths=widths, repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), font),
                ("FONTNAME", (0, 0), (-1, 0), bold),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eaf1f8")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1f3b57")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#b8c5d1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return t


def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("CN", 8)
    canvas.setFillColor(colors.HexColor("#6d7782"))
    canvas.drawString(2.0 * cm, 1.2 * cm, "AirSim + YOLOv8 无人机自动追踪算法")
    canvas.drawRightString(A4[0] - 2.0 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


def build() -> None:
    font, bold = register_fonts()
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "TitleCN",
        parent=styles["Title"],
        fontName=bold,
        fontSize=22,
        leading=28,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#17324d"),
        spaceAfter=14,
    )
    subtitle = ParagraphStyle(
        "SubtitleCN",
        parent=styles["Normal"],
        fontName=font,
        fontSize=10,
        leading=15,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#5d6975"),
        spaceAfter=18,
    )
    h1 = ParagraphStyle(
        "H1CN",
        parent=styles["Heading1"],
        fontName=bold,
        fontSize=16,
        leading=21,
        textColor=colors.HexColor("#17324d"),
        spaceBefore=12,
        spaceAfter=8,
    )
    h2 = ParagraphStyle(
        "H2CN",
        parent=styles["Heading2"],
        fontName=bold,
        fontSize=12.5,
        leading=17,
        textColor=colors.HexColor("#244b70"),
        spaceBefore=9,
        spaceAfter=5,
    )
    body = ParagraphStyle(
        "BodyCN",
        parent=styles["BodyText"],
        fontName=font,
        fontSize=9.6,
        leading=15.5,
        alignment=TA_LEFT,
        spaceAfter=6,
    )
    mono = ParagraphStyle(
        "MonoCN",
        parent=styles["Code"],
        fontName="Courier",
        fontSize=8.2,
        leading=11,
        backColor=colors.HexColor("#f5f7fa"),
        borderColor=colors.HexColor("#d7dde5"),
        borderWidth=0.5,
        borderPadding=6,
        leftIndent=0,
        spaceBefore=4,
        spaceAfter=8,
    )

    story = [
        p("AirSim + YOLOv8 无人机自动追踪算法原理与实现", title),
        p(
            "本文档说明双无人机自动追踪系统的算法原理和工程实现。系统以 Tracker 无人机前视相机为输入，"
            "通过 YOLOv8 锁定 Target 无人机，再使用图像平面 Kalman 预测器估计未来短时位置，最后由视觉伺服控制器"
            "驱动 Tracker 保持目标位于画面中心。",
            body,
        ),
        p("1. 系统目标与架构", h1),
        bullets(
            [
                "Tracker 是主控无人机，负责采集图像和执行追踪控制。",
                "Target 是被追踪无人机，是检测器唯一需要识别的 drone 类别。",
                "闭环目标是 Target 飞到哪里，Tracker 就追到哪里，同时 Target 不脱离视线。",
                "控制输入来自相机画面中的检测框，而不是直接使用 AirSim 真值位置。",
            ],
            body,
        ),
        code(
            """
Tracker camera frame
  -> YOLOv8 detector
  -> ImagePlaneKalmanPredictor
  -> VisualServoController
  -> AirSim body velocity / yaw / gimbal command
  -> next camera frame
            """,
            mono,
        ),
        p("2. YOLOv8 目标锁定", h1),
        p(
            "YOLOv8 对 Tracker 相机图像进行推理，输出目标无人机边界框 Detection。边界框使用 xyxy 像素格式，"
            "并携带置信度 confidence 和类别 class_id。检测框中心和尺寸是后续预测与控制的唯一视觉状态。",
            body,
        ),
        code(
            """
cx = (x1 + x2) / 2
cy = (y1 + y2) / 2
w  = x2 - x1
h  = y2 - y1
            """,
            mono,
        ),
        p("重新采集数据的原因", h2),
        p(
            "旧数据可能包含非无人机目标、目标过小或标注策略不一致。当前方案要求目标就是 Target 无人机，"
            "采集时让目标更大、更靠近画面中心，并尽量露出机架和螺旋桨，使模型学习真正的无人机外观特征。",
            body,
        ),
        p("3. 图像平面 Kalman 运动预测", h1),
        p(
            "预测器使用常速度模型，不直接预测三维真值坐标，而是在图像平面预测检测框。这样能直接补偿采集、"
            "推理、控制命令和机体响应带来的延迟，也能在短时丢检时继续给控制器一个合理目标。",
            body,
        ),
        table(
            [
                ["状态量", "含义"],
                ["cx, cy", "目标框中心像素坐标"],
                ["w, h", "目标框宽高，反映目标距离和尺度"],
                ["vx, vy", "目标中心在图像平面的速度"],
                ["vw, vh", "目标框尺寸变化速度"],
            ],
            [4.0 * cm, 11.5 * cm],
            font,
            bold,
        ),
        Spacer(1, 6),
        code(
            """
state = [cx, cy, w, h, vx, vy, vw, vh]^T
measurement = [cx, cy, w, h]^T
x(k|k-1) = F(dt) x(k-1|k-1)
x_future = F(horizon_s) x(k|k)
            """,
            mono,
        ),
        p("状态转移模型", h2),
        p(
            "预测器采用常速度模型。每个控制周期根据真实时间差 dt 更新位置和尺寸，速度项保持不变。"
            "这等价于假设目标在很短时间内近似匀速运动。对 0.2 秒量级的前馈预测，这个假设比复杂神经网络更可靠，也更容易调参。",
            body,
        ),
        code(
            """
F(dt) =
[1 0 0 0 dt 0  0  0]
[0 1 0 0 0  dt 0  0]
[0 0 1 0 0  0  dt 0]
[0 0 0 1 0  0  0  dt]
[0 0 0 0 1  0  0  0]
[0 0 0 0 0  1  0  0]
[0 0 0 0 0  0  1  0]
[0 0 0 0 0  0  0  1]
            """,
            mono,
        ),
        p("Kalman 更新步骤", h2),
        p(
            "YOLO 给出新检测框时，预测器用观测 z=[cx,cy,w,h] 修正状态。"
            "如果检测框有轻微抖动，Kalman 增益会在历史运动趋势和当前观测之间做平衡。",
            body,
        ),
        code(
            """
y = z - H x_pred
S = H P_pred H^T + R
K = P_pred H^T S^-1
x = x_pred + K y
P = (I - K H) P_pred
            """,
            mono,
        ),
        p(
            "其中 Q 由 process_noise 控制，表示目标运动模型的不确定性；R 由 measurement_noise 控制，表示 YOLO 观测的不确定性。"
            "Q 越大，预测器越愿意相信目标可能突然改变速度；R 越大，预测器越不容易被单帧检测噪声拉偏。",
            body,
        ),
        p("关键参数", h2),
        table(
            [
                ["参数", "默认值", "作用"],
                ["horizon_s", "0.2", "预测未来 0.2 秒的位置，用于补偿视觉和控制延迟"],
                ["max_prediction_gap_s", "0.6", "丢检后继续使用预测的最长时间"],
                ["process_noise", "80.0", "越大越允许目标突然加速或转向"],
                ["measurement_noise", "25.0", "越大越不容易被 YOLO 单帧抖动带偏"],
            ],
            [4.2 * cm, 2.4 * cm, 8.9 * cm],
            font,
            bold,
        ),
        p("短时丢检策略", h2),
        p(
            "如果 YOLO 暂时没有检测到 Target，且距离最近一次真实检测没有超过 max_prediction_gap_s，"
            "预测器继续输出预测框。超过阈值后返回 None，原有搜索扫描逻辑接管。",
            body,
        ),
        code(
            """
if raw_detection is not None:
    predict(dt)
    update(raw_detection)
    return predict(horizon_s)

if raw_detection is None and age <= max_prediction_gap_s:
    predict(dt)
    return predict(horizon_s)

return None
            """,
            mono,
        ),
        p("4. 视觉伺服控制", h1),
        p(
            "控制器追踪的是预测框中心。设图像宽高为 W,H，预测中心为 (cx_hat, cy_hat)，归一化中心误差为：",
            body,
        ),
        code(
            """
ex = (cx_hat - W / 2) / (W / 2)
ey = (cy_hat - H / 2) / (H / 2)
            """,
            mono,
        ),
        bullets(
            [
                "Yaw 控制：yaw_rate = clip(k_yaw * ex, -max_yaw_rate, max_yaw_rate)。",
                "垂直控制：vz = clip(k_z * ey, -max_vertical_speed, max_vertical_speed)。",
                "距离控制：使用目标框宽度，而不是强依赖深度图。框太小则前进，框太大则后退。",
                "云台辅助：云台 yaw 快速拉回水平中心，pitch 积分默认关闭以避免长时间漂移。",
            ],
            body,
        ),
        p("控制律细节", h2),
        code(
            """
deadzone:
  if abs(ex) < center_deadzone: ex = 0
  if abs(ey) < center_deadzone: ey = 0

yaw_rate = clip(k_yaw * ex, -max_yaw_rate_deg_s, max_yaw_rate_deg_s)
vz       = clip(k_z   * ey, -max_vertical_m_s,   max_vertical_m_s)

box_width_norm = predicted_w / image_width
size_error = desired_box_width_norm - box_width_norm
vx = clip(k_size * size_error, -max_reverse_m_s, max_forward_m_s)
            """,
            mono,
        ),
        p(
            "这里 vx 使用框宽而不是深度图作为主要距离信号，是因为无人机机架和螺旋桨细节容易让深度图局部不稳定。"
            "框宽是检测器直接输出的视觉量，和“画面里目标大小是否合适”这个追踪目标更一致。",
            body,
        ),
        p("丢失恢复层级", h2),
        table(
            [
                ["阶段", "触发条件", "动作"],
                ["短时丢检", "age <= max_prediction_gap_s", "继续用 Kalman 预测框控制"],
                ["预测超时", "age > max_prediction_gap_s", "预测器返回 None"],
                ["搜索扫描", "控制器没有可用目标", "云台或机体按最后一次目标方向扫描"],
            ],
            [3.2 * cm, 4.8 * cm, 7.5 * cm],
            font,
            bold,
        ),
        p("5. 运行时实现", h1),
        code(
            """
raw_detection = detector.detect(frame.rgb)
prediction = predictor.step(
    detection=raw_detection,
    timestamp_s=loop_start,
    image_width=image_width,
    image_height=image_height,
)
command = controller.command(
    prediction.detection,
    frame.depth,
    image_width,
    image_height,
    lost_time,
)
            """,
            mono,
        ),
        p(
            "PredictionResult.detection 的类型仍然是 Detection，因此控制器接口不需要改变。"
            "这一点很重要：预测器是插在检测器和控制器之间的增强层，而不是推翻原有控制结构。",
            body,
        ),
        p("工程文件映射", h2),
        table(
            [
                ["文件", "职责"],
                ["src/drone_tracker/detector.py", "封装 YOLOv8，输出 Detection"],
                ["src/drone_tracker/predictor.py", "实现 ImagePlaneKalmanPredictor 和 PredictionResult"],
                ["src/drone_tracker/controller.py", "根据预测框中心误差和框宽生成控制命令"],
                ["src/drone_tracker/metrics.py", "写出 tracking CSV 和 summary JSON"],
                ["scripts/run_tracking.py", "主闭环，串联 AirSim、YOLO、预测器、控制器和日志"],
                ["config/tracking_config.json", "启用预测的生产配置"],
                ["config/tracking_config_no_prediction.json", "关闭预测的 A/B 对照配置"],
            ],
            [6.0 * cm, 9.5 * cm],
            font,
            bold,
        ),
        p("主循环顺序", h2),
        code(
            """
1. move_target(...)                 # AirSim 中移动 Target
2. frame = get_scene_and_depth(...) # Tracker 相机取图
3. raw = detector.detect(frame.rgb) # YOLO 原始检测
4. pred = predictor.step(raw, t)    # Kalman 更新和未来预测
5. cmd = controller.command(pred.detection, ...)
6. set_camera_gimbal(...)           # 云台辅助居中
7. move_body_velocity(...)          # 发送 Tracker 控制命令
8. write TrackingSample             # 记录原始检测和预测字段
            """,
            mono,
        ),
        p("6. 日志指标", h1),
        p("CSV 在原有追踪字段基础上增加预测字段：", body),
        bullets(
            [
                "prediction_used: 当前控制是否使用预测结果。",
                "predicted_center_x / predicted_center_y: 预测框中心。",
                "raw_center_x / raw_center_y: YOLO 原始检测中心。",
                "prediction_age_s: 距离最近一次真实检测的时间。",
            ],
            body,
        ),
        p("summary 指标包括 visible_rate、lost_count、max_lost_duration_s、center_error_mean_px、center_error_p95_px 和 prediction_used_rate。", body),
        p("指标解释", h2),
        table(
            [
                ["指标", "解释", "理想结果"],
                ["visible_rate", "控制周期中目标可用的比例", "越接近 1 越好"],
                ["lost_count", "连续丢失段数量", "0"],
                ["max_lost_duration_s", "最长连续丢失时间", "0 或尽可能小"],
                ["center_error_p95_px", "95% 情况下的中心误差上界", "低于基线 67.71 px"],
                ["prediction_used_rate", "预测器参与控制的比例", "用于理解预测器实际作用"],
            ],
            [4.0 * cm, 7.5 * cm, 4.0 * cm],
            font,
            bold,
        ),
        p("7. 验证与验收", h1),
        code(
            """
python -m compileall src scripts tests
python tests\\test_predictor.py

python scripts\\run_tracking.py --config config\\tracking_config.json ^
  --weights D:/sim/runs/drone_tracker/detect/drone_centered_yolov8s_v2/weights/best.pt ^
  --seconds 90

python scripts\\run_tracking.py --config config\\tracking_config_no_prediction.json ^
  --weights D:/sim/runs/drone_tracker/detect/drone_centered_yolov8s_v2/weights/best.pt ^
  --seconds 90
            """,
            mono,
        ),
        table(
            [
                ["验收项", "目标"],
                ["visible_rate", ">= 0.98"],
                ["lost_count", "0"],
                ["center_error_p95_px", "<= 67.71 px"],
            ],
            [6.0 * cm, 9.5 * cm],
            font,
            bold,
        ),
        p(
            "A/B 对照很重要：只看启用预测的一次结果，无法判断收益来自预测器还是目标轨迹更简单。"
            "因此需要用同一权重、同一目标策略、同一运行时长分别跑 prediction.enabled=true 和 false。",
            body,
        ),
        p("8. 调参建议", h1),
        bullets(
            [
                "追踪慢半拍：将 horizon_s 从 0.2 增加到 0.25。",
                "突然转向过冲：将 horizon_s 降到 0.15。",
                "检测框抖动明显：增大 measurement_noise。",
                "预测跟不上快速运动：增大 process_noise。",
                "丢检后预测漂移：降低 max_prediction_gap_s。",
            ],
            body,
        ),
        p("9. 局限与后续方向", h1),
        p(
            "当前预测器只在图像平面工作，适合短时延迟补偿和短时丢检恢复，不适合长时间无观测预测。"
            "后续可以加入相机内参和框宽估距，形成 3D EKF；也可以用 AirSim 日志训练 GRU/LSTM，处理更复杂的机动模式。",
            body,
        ),
        p("典型问题定位", h2),
        table(
            [
                ["现象", "可能原因", "优先处理"],
                ["目标总是偏左或偏右", "yaw 控制符号或云台方向不匹配", "检查 ex 与 yaw_rate 的符号"],
                ["目标上下漂移", "pitch 积分漂移或 k_z 偏小", "关闭 pitch 积分，增大 k_z"],
                ["目标忽大忽小", "k_size 过大或目标框抖动", "降低 k_size 或增大 measurement_noise"],
                ["短时丢检后乱追", "max_prediction_gap_s 过长", "降低到 0.3 到 0.5"],
                ["转弯时过冲", "horizon_s 过大", "降低到 0.15"],
            ],
            [4.0 * cm, 6.0 * cm, 5.5 * cm],
            font,
            bold,
        ),
        p("10. 结论", h1),
        p(
            "自动追踪的核心是检测、预测、控制和恢复机制组成的闭环。YOLOv8 负责锁定目标无人机，"
            "Kalman 预测器根据历史检测框估计未来短时位置，视觉伺服控制器负责把目标拉回画面中心。"
            "这套结构实时、可解释、可调试，适合当前 AirSim 双无人机追踪任务。",
            body,
        ),
    ]

    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        rightMargin=2.0 * cm,
        leftMargin=2.0 * cm,
        topMargin=2.0 * cm,
        bottomMargin=1.8 * cm,
        title="AirSim YOLOv8 自动追踪算法原理与实现",
        author="airsim_yolov8_drone_tracker",
    )
    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    print(OUT)


if __name__ == "__main__":
    build()
