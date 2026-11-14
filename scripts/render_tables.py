#!/usr/bin/env python3
"""Render saved CW1/SW1 summary arithmetic as readable Markdown and CSV.

This reads completed SUMMARY.json only. It does not fit or evaluate models.
"""
import argparse
import csv
import json
from pathlib import Path


def scalar(value):
    if isinstance(value, dict):
        return value.get("mean")
    return value


def display(value):
    return "NA" if value is None else f"{value:.9f}"


def render(summary_path, destination):
    summary_path, destination = Path(summary_path), Path(destination)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    destination.mkdir(parents=True, exist_ok=True)
    rows, horizons, contrasts, subgroup_rows, disagreement_rows = [], [], [], [], []
    lines = ["# Recomputed experiment tables", "",
             "Brier and log loss are lower-is-better. AUC is higher-is-better. Means weight the sixteen paired blocks equally.", "",
             "Intervals are pointwise paired-block bootstrap intervals, not simultaneous discoveries. The supplied four-step symmetry repeats h1–h4 at h5–h8; this is not learned extrapolation.", ""]
    for condition, result in summary["conditions"].items():
        lines += ["## " + condition, "", "| Predictor | h5–h8 Brier | Log loss | AUC |",
                  "|---|---:|---:|---:|"]
        for method, values in result["strata"]["all"]["methods"].items():
            row = {"condition": condition, "method": method,
                   **{metric: scalar(values["h5_h8"][metric]) for metric in ("brier", "logloss", "auc")}}
            rows.append(row)
            lines.append("| " + method + " | " + " | ".join(display(row[m]) for m in ("brier", "logloss", "auc")) + " |")
            for h, metrics in values["per_horizon"].items():
                horizons.append({"condition": condition, "method": method, "horizon": int(h),
                                 **{m: scalar(metrics[m]) for m in ("brier", "logloss", "auc")}})
        lines += ["", "### Paired Brier contrasts", "", "| Contrast | Mean difference | 95% interval | Negative blocks |",
                  "|---|---:|---|---:|"]
        for name, value in result["contrasts_h5_h8"].items():
            interval = value["interval_percentile95"]
            if not isinstance(interval, list) or len(interval) != 2:
                raise ValueError("Expected the original paired-bootstrap percentile interval")
            row = {"condition": condition, "contrast": name, "mean": value["mean"],
                   "lower": interval[0], "upper": interval[1], "negative_blocks": value["negative_blocks"]}
            contrasts.append(row)
            lines.append(f"| {name} | {display(row['mean'])} | [{display(row['lower'])}, {display(row['upper'])}] | {row['negative_blocks']} |")
        lines.append("")
        for group, group_result in result["strata"].items():
            for method, values in group_result["methods"].items():
                subgroup_rows.append({"condition": condition, "group": group, "method": method,
                                      "pairs": sum(group_result.get("pair_counts_by_block", [])),
                                      **{m: scalar(values["h5_h8"][m]) for m in ("brier", "logloss", "auc")}})
        disagreement = result.get("D_strata_h5_h8", {})
        if disagreement:
            lines += ["### Forecast disagreement strata", "", "These are descriptive query-level strata; fixed thresholds and full block aggregation are retained.", "",
                      "| Disagreement stratum | Method | Query count | Brier |", "|---|---|---:|---:|"]
            for group, group_result in disagreement.items():
                for method, values in group_result["methods"].items():
                    row = {"condition": condition, "group": group, "method": method,
                           "queries": sum(group_result["rows_by_block"]), "brier": scalar(values)}
                    disagreement_rows.append(row)
                    lines.append(f"| {group} | {method} | {row['queries']} | {display(row['brier'])} |")
            lines.append("")
    outputs = {"TABLES.md": "\n".join(lines) + "\n"}
    for filename, data in (("primary_metrics.csv", rows), ("per_horizon.csv", horizons),
                           ("paired_contrasts.csv", contrasts), ("subgroup_metrics.csv", subgroup_rows),
                           ("disagreement_metrics.csv", disagreement_rows)):
        if not data:
            continue
        import io
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=list(data[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(data)
        outputs[filename] = buffer.getvalue()
    for name, content in outputs.items():
        path = destination / name
        if path.exists():
            if path.read_text(encoding="utf-8") != content:
                raise RuntimeError("Existing table differs; use a new output folder: " + str(path))
        else:
            with path.open("x", encoding="utf-8", newline="") as handle:
                handle.write(content)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(render(args.summary, args.output))
