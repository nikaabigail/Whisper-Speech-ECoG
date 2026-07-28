#!/usr/bin/env python3
"""Build reproducible publication figures and tables for the SWPD PCA50 result."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from whisper_ecog_ext.integrity import sha256_file  # noqa: E402


SYSTEMS = ("mel80", "L3", "L4", "L5")
DISPLAY = {"mel80": "MEL80", "L3": "Whisper L3", "L4": "Whisper L4", "L5": "Whisper L5"}
COLORS = {"mel80": "#6B7280", "L3": "#0072B2", "L4": "#009E73", "L5": "#D55E00"}
EXPECTED_FINAL_SHA256 = "5c6fa8cbcedaf81867e11f77aafd502779190e5fb3abc4aa6ab333bb6424f3f6"

GROUPED_DISPLAY = {
    "mel80": "MEL80\n(authors' target)",
    "L3": "Whisper L3\n(ours)",
    "L4": "Whisper L4\n(ours)",
    "L5": "Whisper L5\n(ours)",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "figure.dpi": 130,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "svg.fonttype": "none",
            "svg.hashsalt": "swpd_matched_pca50_publication_assets_v1",
        }
    )


def _save(fig: plt.Figure, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.png", facecolor="white")
    fig.savefig(output / f"{stem}.svg", facecolor="white")
    plt.close(fig)


def _rows(primary: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = list(primary["subject_rows"])
    if len(rows) != 8:
        raise RuntimeError(f"Expected 8 primary patient rows, found {len(rows)}")
    return rows


def _system_values(rows: Sequence[Mapping[str, Any]], system: str) -> np.ndarray:
    return np.asarray([row[system]["fisher_r"] for row in rows], dtype=np.float64)


def _mark_method_groups(ax: plt.Axes, *, top: float) -> None:
    """Make the reproduced authors' target and our replacement visually explicit."""
    ax.axvspan(-0.5, 0.5, color=COLORS["mel80"], alpha=0.075, zorder=0)
    ax.axvspan(0.5, 3.5, color=COLORS["L3"], alpha=0.045, zorder=0)
    ax.axvline(0.5, color="#9CA3AF", linewidth=1.0, linestyle="--", zorder=1)
    ax.text(
        0,
        top,
        "AUTHORS' MEL TARGET\nreproduced control",
        ha="center",
        va="top",
        fontsize=7.8,
        color="#374151",
        weight="bold",
    )
    ax.text(
        2,
        top,
        "OUR WHISPER TARGETS\nproposed replacement",
        ha="center",
        va="top",
        fontsize=7.8,
        color="#075985",
        weight="bold",
    )


def figure_system_performance(primary: Mapping[str, Any], output: Path) -> None:
    rows = _rows(primary)
    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    rng = np.random.default_rng(20260728)
    for index, system in enumerate(SYSTEMS):
        values = _system_values(rows, system)
        jitter = rng.uniform(-0.10, 0.10, len(values))
        ax.scatter(
            index + jitter,
            values,
            s=25,
            facecolors="white",
            edgecolors=COLORS[system],
            linewidths=1.1,
            alpha=0.95,
            zorder=3,
        )
        stats = primary["systems"][system]["fisher_r"]
        mean = float(stats["mean"])
        low, high = map(float, stats["ci95_t"])
        ax.errorbar(
            index,
            mean,
            yerr=[[mean - low], [high - mean]],
            fmt="D",
            markersize=6,
            color=COLORS[system],
            markeredgecolor="white",
            markeredgewidth=0.7,
            capsize=5,
            elinewidth=2.0,
            zorder=4,
        )
        ax.text(index, high + 0.0035, f"{mean:.3f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(range(len(SYSTEMS)), [GROUPED_DISPLAY[item] for item in SYSTEMS])
    ax.set_ylabel("Held-out component correlation (r)")
    ax.set_title("Authors' MEL target versus our Whisper replacement")
    ax.set_ylim(0.0, 0.105)
    _mark_method_groups(ax, top=0.102)
    ax.grid(axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.75)
    ax.text(
        0.99,
        0.02,
        "Points: patients (n=8)   Diamonds: mean   Bars: 95% t-CI",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.3,
        color="#374151",
    )
    _save(fig, output, "figure_01_system_performance")


def figure_paired_patients(primary: Mapping[str, Any], output: Path) -> None:
    rows = _rows(primary)
    x = np.arange(len(SYSTEMS))
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    right_labels: list[tuple[float, str]] = []
    for row in rows:
        values = np.asarray([row[system]["fisher_r"] for system in SYSTEMS])
        ax.plot(x, values, color="#9CA3AF", linewidth=0.9, alpha=0.75, zorder=1)
        ax.scatter(x, values, color=[COLORS[item] for item in SYSTEMS], s=18, zorder=2)
        right_labels.append((float(values[-1]), row["subject"].replace("sub-", "P")))

    previous_y = -np.inf
    for value, label in sorted(right_labels):
        label_y = max(value, previous_y + 0.0027)
        previous_y = label_y
        ax.plot([x[-1] + 0.01, x[-1] + 0.06], [value, label_y], color="#9CA3AF", linewidth=0.6)
        ax.text(
            x[-1] + 0.075,
            label_y,
            label,
            va="center",
            fontsize=7.5,
            color="#4B5563",
        )
    means = np.asarray(
        [primary["systems"][system]["fisher_r"]["mean"] for system in SYSTEMS]
    )
    ax.plot(x, means, color="#111827", linewidth=2.4, zorder=3)
    ax.scatter(x, means, marker="D", color="#111827", edgecolor="white", s=45, zorder=4)
    ax.set_xticks(x, [GROUPED_DISPLAY[item] for item in SYSTEMS])
    ax.set_xlim(-0.18, 3.43)
    ax.set_ylim(0.0, 0.105)
    ax.set_ylabel("Held-out component correlation (r)")
    ax.set_title("Every patient: authors' MEL target versus our Whisper targets")
    _mark_method_groups(ax, top=0.102)
    ax.grid(axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.75)
    ax.text(
        0.99,
        0.02,
        "Each line is one confirmatory patient; black diamonds show means",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.3,
        color="#374151",
    )
    _save(fig, output, "figure_02_paired_patients")


def figure_deltas(primary: Mapping[str, Any], output: Path) -> None:
    rows = _rows(primary)
    layers = ("L3", "L4", "L5")
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    rng = np.random.default_rng(345)
    for index, layer in enumerate(layers):
        deltas = np.asarray(
            [row[layer]["fisher_r"] - row["mel80"]["fisher_r"] for row in rows]
        )
        jitter = rng.uniform(-0.09, 0.09, len(deltas))
        ax.scatter(
            index + jitter,
            deltas,
            s=30,
            color=COLORS[layer],
            alpha=0.80,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        contrast = primary["contrasts"][f"{layer}_minus_mel80"]
        mean = float(contrast["mean"])
        low, high = map(float, contrast["ci95_t"])
        ax.errorbar(
            index,
            mean,
            yerr=[[mean - low], [high - mean]],
            fmt="D",
            markersize=7,
            color="#111827",
            markerfacecolor=COLORS[layer],
            markeredgecolor="white",
            capsize=5,
            linewidth=2,
            zorder=4,
        )
        p_holm = float(contrast["two_sided_paired_t_p_holm"])
        ax.text(
            index,
            0.052,
            f"8/8 wins\nHolm p={p_holm:.2g}",
            ha="center",
            va="top",
            fontsize=8.3,
        )
    ax.axhline(0.0, color="#111827", linewidth=1.0)
    ax.set_xticks(range(3), [f"Whisper {item} − MEL80" for item in layers])
    ax.set_ylabel("Paired correlation difference (Δr)")
    ax.set_title("Our Whisper targets minus the reproduced authors' MEL target")
    ax.set_ylim(-0.005, 0.056)
    ax.grid(axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.75)
    _save(fig, output, "figure_03_whisper_minus_mel")


def _box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    face: str,
    edge: str = "#374151",
    fontsize: float = 9.0,
) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        facecolor=face,
        edgecolor=edge,
        linewidth=1.0,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=fontsize)


def _arrow(ax: plt.Axes, start: tuple[float, float], stop: tuple[float, float]) -> None:
    ax.annotate(
        "",
        xy=stop,
        xytext=start,
        arrowprops={"arrowstyle": "-|>", "lw": 1.1, "color": "#4B5563"},
    )


def figure_architecture(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.0, 6.0), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _box(ax, (0.03, 0.68), 0.15, 0.16, "ECoG\n1024 Hz", face="#E5E7EB")
    _box(ax, (0.23, 0.68), 0.19, 0.16, "70–170 Hz envelope\n50 ms window / 20 ms step", face="#DBEAFE", edge=COLORS["L3"])
    _box(ax, (0.47, 0.68), 0.17, 0.16, "Train-only\nScaler + PCA50", face="#DBEAFE", edge=COLORS["L3"])
    _box(ax, (0.03, 0.23), 0.15, 0.16, "Recorded audio\n48 kHz", face="#E5E7EB")
    _box(ax, (0.23, 0.35), 0.19, 0.13, "MEL80\n50 Hz", face="#F3F4F6", edge=COLORS["mel80"])
    _box(ax, (0.23, 0.12), 0.19, 0.17, "Whisper-base\nencoder L3 / L4 / L5\n50 Hz", face="#ECFDF5", edge=COLORS["L4"])
    _box(ax, (0.47, 0.23), 0.17, 0.16, "Separate train-only\nScaler + PCA50\nfor each target", face="#FEF3C7", edge="#B45309")
    _box(ax, (0.69, 0.48), 0.13, 0.18, "Identical OLS\n50 → 50\nper target", face="#FCE7F3", edge="#9D174D")
    _box(ax, (0.86, 0.48), 0.11, 0.18, "Held-out\nblock metric\ncomponent r", face="#F3E8FF", edge="#6B21A8")
    _box(ax, (0.69, 0.16), 0.28, 0.16, "Five temporal folds per patient\n3 train + 1 validation + 1 test\nPatient-level paired statistics", face="#F9FAFB")
    _arrow(ax, (0.18, 0.76), (0.23, 0.76))
    _arrow(ax, (0.42, 0.76), (0.47, 0.76))
    _arrow(ax, (0.18, 0.31), (0.23, 0.405))
    _arrow(ax, (0.18, 0.31), (0.23, 0.205))
    _arrow(ax, (0.42, 0.415), (0.47, 0.33))
    _arrow(ax, (0.42, 0.205), (0.47, 0.31))
    _arrow(ax, (0.64, 0.76), (0.69, 0.60))
    _arrow(ax, (0.64, 0.31), (0.69, 0.54))
    _arrow(ax, (0.82, 0.57), (0.86, 0.57))
    _arrow(ax, (0.915, 0.48), (0.84, 0.32))
    ax.text(0.03, 0.93, "Matched SWPD external-validation architecture", fontsize=14, weight="bold", ha="left")
    ax.text(
        0.03,
        0.89,
        "Neural input, splits, decoder and metric are identical; only the acoustic target representation changes.",
        fontsize=9.5,
        color="#374151",
        ha="left",
    )
    ax.text(
        0.23,
        0.055,
        "L3/L4/L5 PCA coordinates are separate; direct averaging is therefore not a defined ensemble.",
        fontsize=8.7,
        color="#7C2D12",
        ha="left",
    )
    _save(fig, output, "figure_04_architecture")


def figure_qc_flow(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 3.5), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _box(ax, (0.03, 0.36), 0.17, 0.27, "10 planned\nSWPD participants", face="#E5E7EB", fontsize=10)
    _box(ax, (0.29, 0.60), 0.20, 0.23, "sub-01\nDevelopment only", face="#DBEAFE", edge=COLORS["L3"], fontsize=10)
    _box(ax, (0.29, 0.17), 0.20, 0.25, "sub-10 excluded\n95/100 valid trials\nrecording truncated", face="#FEE2E2", edge="#B91C1C", fontsize=9.5)
    _box(ax, (0.60, 0.36), 0.17, 0.27, "8 confirmatory\nsub-02…sub-09", face="#DCFCE7", edge="#15803D", fontsize=10)
    _box(ax, (0.83, 0.36), 0.14, 0.27, "Primary\npatient-level\ninference", face="#F3E8FF", edge="#6B21A8", fontsize=9.5)
    _arrow(ax, (0.20, 0.50), (0.29, 0.71))
    _arrow(ax, (0.20, 0.48), (0.29, 0.29))
    _arrow(ax, (0.20, 0.50), (0.60, 0.50))
    _arrow(ax, (0.77, 0.50), (0.83, 0.50))
    ax.text(0.50, 0.91, "Transparent cohort accounting", ha="center", fontsize=13, weight="bold")
    ax.text(
        0.50,
        0.05,
        "No imputation and no participant-specific 95-trial split; exclusion is based only on source-file QC.",
        ha="center",
        fontsize=8.8,
        color="#374151",
    )
    _save(fig, output, "figure_05_qc_cohort_flow")


def figure_summary_panel(primary: Mapping[str, Any], output: Path) -> None:
    rows = _rows(primary)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), constrained_layout=True)
    ax = axes[0]
    for index, system in enumerate(SYSTEMS):
        stats = primary["systems"][system]["fisher_r"]
        mean = float(stats["mean"])
        low, high = map(float, stats["ci95_t"])
        ax.errorbar(
            index,
            mean,
            yerr=[[mean - low], [high - mean]],
            fmt="D",
            markersize=7,
            color=COLORS[system],
            capsize=5,
            linewidth=2,
        )
        ax.text(index, high + 0.003, f"{mean:.3f}", ha="center", fontsize=8.5)
    ax.set_xticks(range(4), [GROUPED_DISPLAY[item] for item in SYSTEMS])
    ax.set_ylim(0, 0.105)
    ax.set_ylabel("Held-out component correlation (r)")
    ax.set_title("A  Authors' MEL target vs our Whisper targets", loc="left")
    _mark_method_groups(ax, top=0.102)
    ax.grid(axis="y", color="#D1D5DB", linewidth=0.6)

    ax = axes[1]
    for row in rows:
        values = np.asarray([row[item]["fisher_r"] for item in SYSTEMS])
        ax.plot(range(4), values, color="#9CA3AF", linewidth=0.8, alpha=0.7)
        ax.scatter(range(4), values, color=[COLORS[item] for item in SYSTEMS], s=18)
    means = [primary["systems"][item]["fisher_r"]["mean"] for item in SYSTEMS]
    ax.plot(range(4), means, color="#111827", linewidth=2.2)
    ax.scatter(range(4), means, marker="D", color="#111827", edgecolor="white", s=40)
    ax.set_xticks(range(4), [GROUPED_DISPLAY[item] for item in SYSTEMS])
    ax.set_ylim(0, 0.105)
    ax.set_title("B  Paired confirmatory patients (n=8)", loc="left")
    _mark_method_groups(ax, top=0.102)
    ax.grid(axis="y", color="#D1D5DB", linewidth=0.6)
    _save(fig, output, "figure_00_main_summary")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_tables(primary: Mapping[str, Any], output: Path) -> None:
    system_rows = []
    for system in SYSTEMS:
        r_stats = primary["systems"][system]["fisher_r"]
        mse_stats = primary["systems"][system]["standardized_mse"]
        system_rows.append(
            {
                "system": DISPLAY[system],
                "n": r_stats["n"],
                "fisher_r_mean": r_stats["mean"],
                "fisher_r_sd": r_stats["sd"],
                "fisher_r_ci95_low": r_stats["ci95_t"][0],
                "fisher_r_ci95_high": r_stats["ci95_t"][1],
                "standardized_mse_mean": mse_stats["mean"],
                "standardized_mse_sd": mse_stats["sd"],
            }
        )
    _write_csv(output / "table_01_system_performance.csv", system_rows)

    contrast_rows = []
    for layer in ("L3", "L4", "L5"):
        stats = primary["contrasts"][f"{layer}_minus_mel80"]
        contrast_rows.append(
            {
                "contrast": f"Whisper {layer} - MEL80",
                "n": stats["n"],
                "delta_r_mean": stats["mean"],
                "delta_r_sd": stats["sd"],
                "delta_r_ci95_low": stats["ci95_t"][0],
                "delta_r_ci95_high": stats["ci95_t"][1],
                "wins": stats["wins"],
                "paired_t_p_raw": stats["two_sided_paired_t_p_raw"],
                "paired_t_p_holm": stats["two_sided_paired_t_p_holm"],
                "exact_sign_p_two_sided_sensitivity": 0.0078125,
            }
        )
    _write_csv(output / "table_02_whisper_vs_mel_contrasts.csv", contrast_rows)

    patient_rows = []
    for source in _rows(primary):
        row = {"subject": source["subject"]}
        for system in SYSTEMS:
            row[f"{system}_fisher_r"] = source[system]["fisher_r"]
            row[f"{system}_standardized_mse"] = source[system]["standardized_mse"]
        patient_rows.append(row)
    _write_csv(output / "table_03_patient_level_metrics.csv", patient_rows)

    markdown = [
        "# SWPD matched PCA50 publication tables",
        "",
        "## System performance",
        "",
        "| System | r, mean ± SD | 95% t-CI | Standardized MSE, mean ± SD |",
        "|---|---:|---:|---:|",
    ]
    for row in system_rows:
        markdown.append(
            f"| {row['system']} | {row['fisher_r_mean']:.5f} ± {row['fisher_r_sd']:.5f} | "
            f"[{row['fisher_r_ci95_low']:.5f}, {row['fisher_r_ci95_high']:.5f}] | "
            f"{row['standardized_mse_mean']:.5f} ± {row['standardized_mse_sd']:.5f} |"
        )
    markdown.extend(
        [
            "",
            "## Predeclared paired contrasts",
            "",
            "| Contrast | Δr, mean ± SD | 95% t-CI | Wins | Raw p | Holm p |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in contrast_rows:
        markdown.append(
            f"| {row['contrast']} | {row['delta_r_mean']:.5f} ± {row['delta_r_sd']:.5f} | "
            f"[{row['delta_r_ci95_low']:.5f}, {row['delta_r_ci95_high']:.5f}] | "
            f"{row['wins']}/8 | {row['paired_t_p_raw']:.3g} | {row['paired_t_p_holm']:.3g} |"
        )
    markdown.extend(
        [
            "",
            "Primary cohort: sub-02 through sub-09 (n=8). sub-01 is development-only; sub-10 is excluded for an incomplete source recording.",
            "",
        ]
    )
    (output / "publication_tables.md").write_text("\n".join(markdown), encoding="utf-8", newline="\n")


def write_captions(output: Path) -> None:
    text = """# Figure captions

**Figure 1. Matched acoustic-representation decoding.** Grey denotes the authors' MEL80 acoustic target reproduced by us as the control inside the matched pipeline; colored marks denote our Whisper L3/L4/L5 replacements. These are all recomputed results on SWPD, not a direct copy of a number from the source paper. Each open circle is one confirmatory patient (sub-02 through sub-09). Diamonds show the patient mean and error bars show two-sided 95% t confidence intervals. Neural inputs, temporal splits, train-only PCA50 transforms, OLS decoder, and metric were identical across MEL80 and Whisper layers.

**Figure 2. Patient-level paired comparison.** The grey column is the reproduced authors' MEL80 target and the colored columns are our Whisper targets. Lines connect the four target representations within each patient. The black line and diamonds show patient means. The within-patient design isolates the target representation while holding the ECoG data and decoder protocol fixed.

**Figure 3. Our Whisper-minus-authors' MEL80 paired effects.** Points are patient-level differences between our Whisper target and the reproduced MEL80 control. Diamonds and bars show mean differences and 95% t confidence intervals. Reported p-values are two-sided paired t-tests with Holm correction across the three predeclared layer-versus-MEL contrasts.

**Figure 4. Matched SWPD architecture.** ECoG high-gamma features and each acoustic target are reduced using fold-train-only standardized PCA50 transforms. A shared neural reducer and identical OLS decoder are used for all targets. Layer-specific PCA coordinates are not directly averaged, so this experiment does not define an L3+L4+L5 ensemble.

**Figure 5. Cohort accounting.** Ten participants were planned. sub-01 was used only for development. sub-10 was excluded because its official recording ends after 95 valid word trials and the final five event rows have zero duration at the final recorded sample. No imputation or participant-specific split was used. Primary inference therefore includes eight confirmatory patients.
"""
    (output / "figure_captions.md").write_text(text, encoding="utf-8", newline="\n")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-unpinned-source", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    source = args.final_summary.resolve()
    observed_sha256 = sha256_file(source)
    if not args.allow_unpinned_source and observed_sha256 != EXPECTED_FINAL_SHA256:
        raise RuntimeError(
            f"Final summary checksum changed: {observed_sha256} != {EXPECTED_FINAL_SHA256}"
        )
    payload = json.loads(source.read_text(encoding="utf-8"))
    primary = payload["primary_confirmatory_after_qc"]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _style()
    figure_summary_panel(primary, output)
    figure_system_performance(primary, output)
    figure_paired_patients(primary, output)
    figure_deltas(primary, output)
    figure_architecture(output)
    figure_qc_flow(output)
    write_tables(primary, output)
    write_captions(output)
    manifest = {
        "schema_version": 1,
        "kind": "swpd_matched_pca50_publication_assets",
        "source_result_id": "swpd_matched_pca50_all_v2_qc_final",
        "source_summary_sha256": observed_sha256,
        "primary_subjects": primary["subjects"],
        "files": sorted(path.name for path in output.iterdir() if path.is_file()),
    }
    (output / "asset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"[assets] {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
