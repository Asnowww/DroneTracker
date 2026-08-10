"""One-command automated AirSim regression: runs the full variant x pattern matrix.

This is the RPC-driven substitute for clicking around the simulator by hand. It is
reproducible (scripted target trajectories, fixed episode length), it produces an
A/B table instead of a subjective impression, and it gates on explicit thresholds.

    # simulator already running:
    python scripts/auto_test.py --plan config/auto_test_plan.json

    # full pipeline from an empty project:
    python scripts/auto_test.py --plan config/auto_test_plan.json --collect --train
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from drone_tracker.config import deep_update, load_json, save_json  # noqa: E402

METRIC_KEYS = [
    "visible_rate",
    "lost_count",
    "max_lost_duration_s",
    "center_error_mean_px",
    "center_error_p95_px",
    "prediction_used_rate",
]


def resolve(path_like: str) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else (ROOT / path)


def run_stage(name: str, command: list[str]) -> None:
    print(f"\n=== stage: {name} ===\n$ {' '.join(command)}\n")
    result = subprocess.run(command, cwd=str(ROOT))
    if result.returncode != 0:
        raise SystemExit(f"stage {name!r} failed with exit code {result.returncode}")


def fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def delta_str(value: float | None, baseline: float | None, lower_is_better: bool) -> str:
    if value is None or baseline is None or baseline == 0:
        return "—"
    change = (value - baseline) / abs(baseline) * 100.0
    good = (change < 0) if lower_is_better else (change > 0)
    mark = "✓" if good and abs(change) >= 1.0 else ("✗" if not good and abs(change) >= 1.0 else "·")
    return f"{change:+.1f}% {mark}"


def build_report(results: list[dict], plan: dict) -> str:
    baseline_name = plan.get("baseline_variant", "no_prediction")
    acceptance = plan.get("acceptance", {})
    lines: list[str] = ["# AirSim 自动追踪回归报告", ""]
    lines.append(f"- weights: `{plan.get('weights')}`")
    lines.append(f"- episode: {plan.get('seconds')} s @ {plan.get('control_hz', 20)} Hz")
    lines.append(f"- baseline variant: `{baseline_name}`")
    lines.append("")

    by_pattern: dict[str, list[dict]] = {}
    for row in results:
        by_pattern.setdefault(row["pattern"], []).append(row)

    for pattern, rows in by_pattern.items():
        lines.append(f"## target pattern: `{pattern}`")
        lines.append("")
        lines.append(
            "| variant | visible_rate | lost_count | center_err_mean_px | Δmean | center_err_p95_px | Δp95 | pred_used | status |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")
        baseline = next((r for r in rows if r["variant"] == baseline_name and r.get("ok")), None)
        for row in rows:
            if not row.get("ok"):
                lines.append(f"| `{row['variant']}` | ERROR | | | | | | | {row.get('error', '')[:60]} |")
                continue
            summary = row["summary"]
            b_mean = baseline["summary"].get("center_error_mean_px") if baseline else None
            b_p95 = baseline["summary"].get("center_error_p95_px") if baseline else None
            lines.append(
                "| `{v}` | {vis} | {lost} | {mean} | {dmean} | {p95} | {dp95} | {pred} | {status} |".format(
                    v=row["variant"],
                    vis=fmt(summary.get("visible_rate")),
                    lost=fmt(summary.get("lost_count")),
                    mean=fmt(summary.get("center_error_mean_px"), 2),
                    dmean=delta_str(summary.get("center_error_mean_px"), b_mean, True),
                    p95=fmt(summary.get("center_error_p95_px"), 2),
                    dp95=delta_str(summary.get("center_error_p95_px"), b_p95, True),
                    pred=fmt(summary.get("prediction_used_rate"), 2),
                    status="PASS" if row["accepted"] else "FAIL",
                )
            )
        lines.append("")

    lines.append("## 验收阈值")
    lines.append("")
    for key, value in acceptance.items():
        lines.append(f"- `{key}` = {value}")
    lines.append("")
    passed = sum(1 for r in results if r.get("accepted"))
    lines.append(f"**{passed}/{len(results)} 组通过验收。**")
    lines.append("")
    return "\n".join(lines)


def check_acceptance(summary: dict, acceptance: dict) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    visible = summary.get("visible_rate")
    if visible is not None and visible < float(acceptance.get("visible_rate_min", 0.0)):
        reasons.append(f"visible_rate {visible:.3f} < {acceptance['visible_rate_min']}")
    lost = summary.get("lost_count")
    if lost is not None and lost > int(acceptance.get("lost_count_max", 10**9)):
        reasons.append(f"lost_count {lost} > {acceptance['lost_count_max']}")
    p95 = summary.get("center_error_p95_px")
    limit = acceptance.get("center_error_p95_px_max")
    if p95 is not None and limit is not None and p95 > float(limit):
        reasons.append(f"center_error_p95_px {p95:.2f} > {limit}")
    return (not reasons), reasons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=str(ROOT / "config" / "auto_test_plan.json"))
    parser.add_argument("--weights", default=None, help="override plan weights")
    parser.add_argument("--seconds", type=float, default=None)
    parser.add_argument("--collect", action="store_true", help="collect a dataset first")
    parser.add_argument("--train", action="store_true", help="train the detector first")
    parser.add_argument("--samples", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args()

    plan = load_json(args.plan)
    weights = args.weights or plan["weights"]
    seconds = float(args.seconds if args.seconds is not None else plan.get("seconds", 90))
    acceptance = plan.get("acceptance", {})
    report_dir = resolve(plan.get("report_dir", "runs/auto_test"))
    stamp = time.strftime("%Y%m%d-%H%M%S")

    python = sys.executable

    if args.collect:
        run_stage(
            "collect_dataset",
            [python, "scripts/collect_dataset.py", "--samples", str(args.samples)],
        )
    if args.train:
        dataset_cfg = load_json(ROOT / "config" / "dataset_config.json")
        data_yaml = Path(dataset_cfg["output_dir"]) / "data.yaml"
        run_stage(
            "train_yolo",
            [python, "scripts/train_yolo.py", "--data", str(data_yaml), "--epochs", str(args.epochs)],
        )

    if not args.skip_preflight:
        run_stage(
            "preflight",
            [python, "scripts/check_airsim_ready.py", "--weights", str(resolve(weights))],
        )

    from run_tracking import run_episode  # noqa: PLC0415 - needs sys.path set above

    base_cfg = load_json(resolve(plan["base_config"]))
    results: list[dict] = []
    patterns = plan.get("patterns", [base_cfg["target_policy"].get("pattern", "front_sweep")])

    total = len(patterns) * len(plan["variants"])
    index = 0
    for pattern in patterns:
        for variant in plan["variants"]:
            index += 1
            name = variant["name"]
            print(f"\n=== run {index}/{total}: pattern={pattern} variant={name} ===")
            cfg = deep_update(base_cfg, variant.get("overrides", {}))
            cfg = deep_update(cfg, {"target_policy": {"pattern": pattern}})
            cfg["run_dir"] = str(report_dir / stamp / f"{pattern}__{name}")
            # Stress patterns may declare their own (usually relaxed) gates.
            pattern_acceptance = deep_update(
                acceptance, plan.get("acceptance_overrides", {}).get(pattern, {})
            )

            row: dict = {"pattern": pattern, "variant": name}
            try:
                summary = run_episode(cfg, str(resolve(weights)), seconds)
                accepted, reasons = check_acceptance(summary, pattern_acceptance)
                row.update(ok=True, summary=summary, accepted=accepted, reasons=reasons)
                print(f"--> accepted={accepted} reasons={reasons}")
            except Exception as exc:  # noqa: BLE001 - one bad run must not kill the matrix
                traceback.print_exc()
                row.update(ok=False, accepted=False, error=f"{type(exc).__name__}: {exc}")
            results.append(row)

            settle = float(plan.get("settle_s", 5.0))
            if index < total and settle > 0:
                time.sleep(settle)

    report_path = report_dir / stamp / "report.md"
    json_path = report_dir / stamp / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_report(results, {**plan, "weights": weights, "seconds": seconds}), encoding="utf-8")
    save_json(json_path, {"plan": plan, "weights": str(weights), "seconds": seconds, "results": results})

    print("\n" + report_path.read_text(encoding="utf-8"))
    print(f"report_md={report_path}")
    print(f"report_json={json_path}")

    failed = [r for r in results if not r.get("accepted")]
    if failed:
        print(f"\nFAILED {len(failed)}/{len(results)} runs")
        sys.exit(1)
    print("\nauto_test_passed=true")


if __name__ == "__main__":
    main()
