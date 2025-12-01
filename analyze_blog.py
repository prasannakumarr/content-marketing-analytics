"""
Content marketing analysis script for blog performance.

Reads blog.csv, cleans percentage fields, and generates marketing-focused
plots saved to the repository root. Also prints short POV notes for each
visual to guide interpretation. Run with:
    python3 analyze_blog.py
"""

from pathlib import Path
from typing import Dict, Tuple

import matplotlib
from matplotlib.patches import Rectangle

# Use a non-interactive backend for sandboxed/CLI environments; figures are saved to disk.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "blog.csv"


def load_data() -> pd.DataFrame:
    """Load and normalize the blog dataset."""
    df = pd.read_csv(CSV_PATH)
    df["conversion_rate"] = df["Conversion rate"].str.rstrip("%").astype(float)
    df["bounce_rate"] = df["Bounce Rate"].str.rstrip("%").astype(float)
    df["engagement_seconds"] = df["Average engagement time per session"]
    df["weighted_conversion_rate"] = (df["Signups"] / df["Users"]) * 100
    return df


def marketing_notes() -> Dict[str, str]:
    """Brief marketing POV for each requested plot."""
    return {
        "author_traffic_signup_conversion": (
            "Shows which writers pull in traffic and signups so we can prioritize"
            " amplification for high-performing authors and support lagging ones."
        ),
        "author_traffic_signup_conversion_log": (
            "Same view on a log scale to spot mid-pack authors when traffic is"
            " skewed by a few large posts."
        ),
        "most_efficient_author": (
            "Ranks authors by conversion efficiency, highlighting who turns readers"
            " into signups best and whose playbooks we should replicate."
        ),
        "traffic_vs_signup_corr": (
            "Reveals how strongly traffic maps to signups—helps decide whether to"
            " focus on acquisition (traffic) or conversion improvements."
        ),
        "engagement_vs_bounce": (
            "Plots stickiness (engagement) against bounce to see whose content keeps"
            " readers on-page and where experience or targeting needs work."
        ),
        "engagement_signup_conversion": (
            "Connects engagement to signups/conversion, showing if deeper"
            " consumption actually drives downstream actions."
        ),
        "posts_needing_conversion_help": (
            "Flags posts with traffic but low conversion so we can fix CTAs,"
            " offers, or intent alignment without needing more visitors."
        ),
        "quadrant_users_vs_conversion": (
            "Segment authors into four quadrants by traffic and conversion rate to"
            " see who needs acquisition help vs conversion fixes vs growth fuel."
        ),
        "quadrant_posts_users_vs_conversion": (
            "Same quadrant view but at the post level to spot which articles combine"
            " traffic and conversion strength versus those needing either acquisition"
            " or conversion work."
        ),
        "correlation_conversion_engagement": (
            "Quantifies whether higher engagement time aligns with better conversion,"
            " clarifying if depth of consumption predicts signups."
        ),
        "engagement_conversion_bounce_interaction": (
            "Shows how engagement and conversion move together while bounce rate is"
            " encoded as color, to reveal if deeper sessions both convert and reduce exits."
        ),
        "author_heatmap_conversion_engagement": (
            "Highlights which authors pair strong conversion efficiency with deeper"
            " engagement, spotlighting best-practice content creators."
        ),
        "quality_volume_outcome": (
            "Blends engagement (up) and bounce (down) into a quality score and plots"
            " it against conversion, with traffic as volume context to see which"
            " posts have quality that actually converts."
        ),
        "bounce_traffic_conversion": (
            "Tests whether high-traffic posts with lower bounce also deliver better"
            " conversion, highlighting if popular content is converting or just browsing."
        ),
    }


def plot_author_traffic_signup_conversion(author_stats: pd.DataFrame) -> Path:
    """Plot author traffic, signups, and conversion rate."""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    sorted_stats = author_stats.sort_values("Users", ascending=False)
    bar_width = 0.35
    positions = range(len(sorted_stats))

    ax1.bar(
        [p - bar_width / 2 for p in positions],
        sorted_stats["Users"],
        width=bar_width,
        label="Users",
        color="#4C78A8",
    )
    ax1.bar(
        [p + bar_width / 2 for p in positions],
        sorted_stats["Signups"],
        width=bar_width,
        label="Signups",
        color="#F28E2B",
    )
    ax1.set_ylabel("Count")
    ax1.set_xticks(list(positions))
    ax1.set_xticklabels(sorted_stats.index, rotation=20, ha="right")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(
        positions,
        sorted_stats["weighted_conversion_rate"],
        color="#59A14F",
        marker="o",
        label="Conversion rate (%)",
    )
    ax2.set_ylabel("Conversion rate (%)")
    ax2.legend(loc="upper right")
    ax1.set_title("Author performance: traffic, signups, conversion rate")

    output = ROOT / "author_traffic_signup_conversion.png"
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    return output


def plot_author_traffic_signup_conversion_log(author_stats: pd.DataFrame) -> Path:
    """Plot author traffic, signups, and conversion rate with log-scaled counts."""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    sorted_stats = author_stats.sort_values("Users", ascending=False)
    bar_width = 0.35
    positions = range(len(sorted_stats))

    ax1.bar(
        [p - bar_width / 2 for p in positions],
        sorted_stats["Users"],
        width=bar_width,
        label="Users",
        color="#4C78A8",
    )
    ax1.bar(
        [p + bar_width / 2 for p in positions],
        sorted_stats["Signups"],
        width=bar_width,
        label="Signups",
        color="#F28E2B",
    )
    ax1.set_ylabel("Count (log scale)")
    ax1.set_yscale("log")
    ax1.set_xticks(list(positions))
    ax1.set_xticklabels(sorted_stats.index, rotation=20, ha="right")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(
        positions,
        sorted_stats["weighted_conversion_rate"],
        color="#59A14F",
        marker="o",
        label="Conversion rate (%)",
    )
    ax2.set_ylabel("Conversion rate (%)")
    ax2.legend(loc="upper right")
    ax1.set_title("Author performance: traffic, signups, conversion rate (log scale)")

    output = ROOT / "author_traffic_signup_conversion_log.png"
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    return output


def plot_most_efficient_author(author_stats: pd.DataFrame) -> Path:
    """Plot conversion efficiency per author, highlighting the leader and showing engagement."""
    ordered = author_stats.sort_values("weighted_conversion_rate", ascending=False)
    top_author = ordered.index[0]
    colors = ["#E15759" if author == top_author else "#9C9E9F" for author in ordered.index]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    positions = range(len(ordered))
    bars = ax1.bar(positions, ordered["weighted_conversion_rate"], color=colors, width=0.6)
    ax1.set_ylabel("Conversion rate (%)")
    ax1.set_title("Most efficient author (signups per user) with engagement context")
    ax1.set_xticks(list(positions))
    ax1.set_xticklabels(ordered.index, rotation=20, ha="right")
    ax1.bar_label(bars, fmt="%.2f")
    median_conv = ordered["weighted_conversion_rate"].median()
    ax1.axhline(
        median_conv,
        color="#7F7F7F",
        linestyle="--",
        linewidth=1,
        label=f"Median conversion ({median_conv:.2f}%)",
    )

    # Overlay engagement on a twin axis for additional context.
    ax2 = ax1.twinx()
    norm = plt.Normalize(vmin=ordered["engagement_seconds"].min(), vmax=ordered["engagement_seconds"].max())
    engagement_scatter = ax2.scatter(
        positions,
        ordered["engagement_seconds"],
        c=ordered["engagement_seconds"],
        cmap="Blues",
        norm=norm,
        marker="o",
        s=80,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.9,
        label="Avg engagement (sec)",
    )
    ax2.set_ylabel("Avg engagement time per session (seconds)")
    median_eng = ordered["engagement_seconds"].median()
    ax2.axhline(
        median_eng,
        color="#59A14F",
        linestyle="--",
        linewidth=1,
        label=f"Median engagement ({median_eng:.1f}s)",
    )
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    cbar = fig.colorbar(engagement_scatter, ax=ax2, label="Avg engagement (sec)")

    output = ROOT / "most_efficient_author.png"
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    return output


def plot_traffic_vs_signup(df: pd.DataFrame) -> Path:
    """Plot correlation between traffic and signups."""
    filtered = df[df["Users"] <= 7000]
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.regplot(
        data=filtered,
        x="Users",
        y="Signups",
        scatter_kws={"alpha": 0.7},
        line_kws={"color": "#E15759"},
        ax=ax,
    )
    ax.set_title("Traffic vs Signups (filtered, Users <= 7000)")

    output = ROOT / "traffic_vs_signup_corr.png"
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    return output


def plot_engagement_vs_bounce(author_stats: pd.DataFrame) -> Path:
    """Plot engagement versus bounce by author."""
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        author_stats["engagement_seconds"],
        author_stats["bounce_rate"],
        s=author_stats["Users"] / 5,
        c="#4C78A8",
        alpha=0.75,
    )
    for author, row in author_stats.iterrows():
        ax.annotate(author, (row["engagement_seconds"], row["bounce_rate"]), fontsize=8)

    ax.set_xlabel("Avg engagement time per session (seconds)")
    ax.set_ylabel("Bounce rate (%)")
    ax.set_title("Author stickiness: engagement vs bounce")

    output = ROOT / "engagement_vs_bounce.png"
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    return output


def plot_engagement_signup_conversion(df: pd.DataFrame) -> Path:
    """Show relationship between engagement and downstream signup/conversion."""
    filtered = df[(df["Signups"] <= 140) & (df["engagement_seconds"] <= 200)]
    size_scale = 300 / filtered["Users"].max() if not filtered.empty else 1
    norm = plt.Normalize(vmin=filtered["Users"].min(), vmax=filtered["Users"].max()) if not filtered.empty else None
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    axes[0].scatter(
        filtered["engagement_seconds"],
        filtered["Signups"],
        alpha=0.65,
        s=filtered["Users"] * size_scale,
        c=filtered["Users"],
        cmap="viridis",
        norm=norm,
    )
    sns.regplot(
        data=filtered,
        x="engagement_seconds",
        y="Signups",
        scatter=False,
        line_kws={"color": "#E15759"},
        ax=axes[0],
    )
    axes[0].set_title("Engagement vs Signups")
    axes[0].set_xlabel("Avg engagement time per session (seconds)")
    if norm is not None:
        fig.colorbar(
            plt.cm.ScalarMappable(cmap="viridis", norm=norm),
            ax=axes[0],
            label="Users (traffic)",
        )

    sns.regplot(
        data=filtered,
        x="engagement_seconds",
        y="weighted_conversion_rate",
        scatter_kws={"alpha": 0.7},
        line_kws={"color": "#59A14F"},
        ax=axes[1],
    )
    axes[1].set_title("Engagement vs Conversion rate")
    axes[1].set_xlabel("Avg engagement time per session (seconds)")
    axes[1].set_ylabel("Conversion rate (%)")

    output = ROOT / "engagement_signup_conversion.png"
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    return output


def plot_posts_needing_conversion_help(df: pd.DataFrame) -> Path:
    """Highlight posts with traffic but weak conversion."""
    filtered = df[df["Users"] <= 6000]
    median_conv = filtered["weighted_conversion_rate"].median()
    median_users = filtered["Users"].median()
    fig, ax = plt.subplots(figsize=(9, 6))

    scatter = ax.scatter(
        filtered["Users"],
        filtered["Signups"],
        c=filtered["weighted_conversion_rate"],
        cmap="viridis",
        alpha=0.8,
    )

    ax.axvline(median_users, color="#9C9E9F", linestyle="--", linewidth=1, label="Median users")
    ax.axhline(filtered["Signups"].median(), color="#E15759", linestyle="--", linewidth=1, label="Median signups")
    ax.set_xlabel("Users")
    ax.set_ylabel("Signups")
    ax.set_title("Posts needing conversion uplift (color = conversion rate)")
    cbar = plt.colorbar(scatter, ax=ax, label="Conversion rate (%)")
    ax.legend()

    output = ROOT / "posts_needing_conversion_help.png"
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    return output


def plot_quadrant_users_vs_conversion(author_stats: pd.DataFrame) -> Path:
    """4-quadrant analysis: Users vs conversion rate performance by author."""
    median_users = author_stats["Users"].median()
    median_conv = author_stats["weighted_conversion_rate"].median()

    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(
        author_stats["Users"],
        author_stats["weighted_conversion_rate"],
        c=author_stats["weighted_conversion_rate"],
        cmap="RdYlGn",
        s=120,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.6,
    )
    for author, row in author_stats.iterrows():
        ax.annotate(author, (row["Users"], row["weighted_conversion_rate"]), fontsize=8, xytext=(4, 4), textcoords="offset points")

    ax.axvline(median_users, color="#9C9E9F", linestyle="--", linewidth=1, label=f"Median users: {median_users:.0f}")
    ax.axhline(median_conv, color="#636363", linestyle="--", linewidth=1, label=f"Median conversion: {median_conv:.2f}%")
    ax.set_xlabel("Users")
    ax.set_ylabel("Conversion rate (%)")
    ax.set_title("4-Quadrant Analysis: Users vs Conversion rate performance")
    fig.colorbar(scatter, ax=ax, label="Conversion rate (%)")
    ax.legend(loc="lower right")

    output = ROOT / "quadrant_users_vs_conversion.png"
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    return output


def plot_quadrant_posts_users_vs_conversion(df: pd.DataFrame) -> Path:
    """4-quadrant analysis: Users vs conversion rate performance by post."""
    filtered = df[df["Users"] <= 6000]
    median_users = filtered["Users"].median()
    median_conv = filtered["weighted_conversion_rate"].median()
    min_users, max_users = filtered["Users"].min(), filtered["Users"].max()
    min_conv, max_conv = filtered["weighted_conversion_rate"].min(), filtered["weighted_conversion_rate"].max()

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(
        filtered["Users"],
        filtered["weighted_conversion_rate"],
        c=filtered["weighted_conversion_rate"],
        cmap="RdYlGn",
        s=80,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )
    # Highlight high-traffic / low-conversion quadrant as a conversion-opportunity zone.
    if max_users > median_users and median_conv > min_conv:
        ax.add_patch(
            Rectangle(
                (median_users, min_conv),
                max_users - median_users,
                median_conv - min_conv,
                facecolor="#FADBD8",
                alpha=0.3,
                edgecolor="none",
                zorder=0,
            )
        )
        ax.text(
            median_users + (max_users - median_users) * 0.05,
            min_conv + (median_conv - min_conv) * 0.5,
            "Opportunity:\nhigh traffic,\nlow conversion",
            fontsize=9,
            color="#A33",
            alpha=0.9,
        )

    ax.axvline(median_users, color="#9C9E9F", linestyle="--", linewidth=1, label=f"Median users: {median_users:.0f}")
    ax.axhline(median_conv, color="#636363", linestyle="--", linewidth=1, label=f"Median conversion: {median_conv:.2f}%")
    ax.set_xlabel("Users")
    ax.set_ylabel("Conversion rate (%)")
    ax.set_title("4-Quadrant Analysis (Posts): Users vs Conversion rate performance")
    fig.colorbar(scatter, ax=ax, label="Conversion rate (%)")
    ax.legend(loc="lower right")

    output = ROOT / "quadrant_posts_users_vs_conversion.png"
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    return output


def plot_correlation_conversion_engagement(df: pd.DataFrame) -> Path:
    """Plot correlation between conversion rate and engagement with annotation."""
    filtered = df[df["engagement_seconds"] <= 170]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.regplot(
        data=filtered,
        x="engagement_seconds",
        y="weighted_conversion_rate",
        scatter_kws={"alpha": 0.7},
        line_kws={"color": "#E15759"},
        ax=ax,
    )
    r = filtered["engagement_seconds"].corr(filtered["weighted_conversion_rate"])
    interpretation = (
        "no clear correlation"
        if abs(r) < 0.1
        else "weak positive correlation"
        if 0.1 <= r < 0.3
        else "moderate positive correlation"
        if 0.3 <= r < 0.5
        else "strong positive correlation"
        if r >= 0.5
        else "negative correlation"
    )
    ax.text(
        0.05,
        0.95,
        f"Pearson r = {r:.2f}\nInterpretation: {interpretation}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    ax.set_xlabel("Avg engagement time per session (seconds)")
    ax.set_ylabel("Conversion rate (%)")
    ax.set_title("Correlation: Engagement vs Conversion rate")

    output = ROOT / "correlation_conversion_engagement.png"
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    return output


def plot_engagement_conversion_bounce(df: pd.DataFrame) -> Path:
    """Plot engagement vs conversion with bounce rate as color."""
    filtered = df[df["engagement_seconds"] <= 200]
    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(
        filtered["engagement_seconds"],
        filtered["weighted_conversion_rate"],
        c=filtered["bounce_rate"],
        cmap="RdYlGn_r",
        alpha=0.8,
        s=60,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_xlabel("Avg engagement time per session (seconds)")
    ax.set_ylabel("Conversion rate (%)")
    ax.set_title("Engagement vs Conversion with Bounce rate (color)")

    # Correlation metrics
    r_conv = filtered["engagement_seconds"].corr(filtered["weighted_conversion_rate"])
    r_bounce = filtered["engagement_seconds"].corr(filtered["bounce_rate"])
    interp_conv = (
        "no clear correlation"
        if abs(r_conv) < 0.1
        else "weak positive"
        if 0.1 <= r_conv < 0.3
        else "moderate positive"
        if 0.3 <= r_conv < 0.5
        else "strong positive"
        if r_conv >= 0.5
        else "negative"
    )
    interp_bounce = (
        "no clear correlation"
        if abs(r_bounce) < 0.1
        else "weak positive"
        if 0.1 <= r_bounce < 0.3
        else "moderate positive"
        if 0.3 <= r_bounce < 0.5
        else "strong positive"
        if r_bounce >= 0.5
        else "negative"
    )
    ax.text(
        0.02,
        0.98,
        f"r(engagement, conversion) = {r_conv:.2f} ({interp_conv})\n"
        f"r(engagement, bounce) = {r_bounce:.2f} ({interp_bounce})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )
    cbar = plt.colorbar(scatter, ax=ax, label="Bounce rate (%)")

    output = ROOT / "engagement_conversion_bounce_interaction.png"
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    return output


def save_correlation_table(df: pd.DataFrame) -> Tuple[pd.DataFrame, Path]:
    """Save correlation table between engagement and conversion."""
    filtered = df[df["engagement_seconds"] <= 170]
    r = filtered["engagement_seconds"].corr(filtered["weighted_conversion_rate"])
    table = pd.DataFrame(
        {
            "metric_a": ["engagement_seconds"],
            "metric_b": ["weighted_conversion_rate"],
            "pearson_r": [r],
        }
    )
    output = ROOT / "correlation_conversion_engagement.csv"
    table.to_csv(output, index=False)
    return table, output


def plot_author_heatmap_conversion_engagement(author_stats: pd.DataFrame) -> Path:
    """Heatmap of conversion rate and engagement by author."""
    metrics = author_stats[["weighted_conversion_rate", "engagement_seconds"]]
    # Normalize for color scale; annotate with actual values.
    norm = metrics.copy()
    for col in metrics.columns:
        col_min, col_max = metrics[col].min(), metrics[col].max()
        if col_max - col_min == 0:
            norm[col] = 0
        else:
            norm[col] = (metrics[col] - col_min) / (col_max - col_min)

    fig, ax = plt.subplots(figsize=(7, max(4, 0.4 * len(metrics) + 2)))
    sns.heatmap(
        norm,
        annot=metrics.round(2),
        fmt=".2f",
        cmap="YlGnBu",
        cbar_kws={"label": "Normalized scale (0-1)"},
        ax=ax,
    )
    ax.set_xlabel("Metric")
    ax.set_ylabel("Author")
    ax.set_title("Author heatmap: conversion rate (%) and engagement (sec)")
    output = ROOT / "author_heatmap_conversion_engagement.png"
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    return output


def plot_quality_volume_outcome(df: pd.DataFrame) -> Path:
    """Three-parameter correlation: quality score (engagement up, bounce down) vs conversion, sized by traffic."""
    # Compute z-scores
    eng_z = (df["engagement_seconds"] - df["engagement_seconds"].mean()) / df["engagement_seconds"].std(ddof=0)
    bounce_z = (df["bounce_rate"] - df["bounce_rate"].mean()) / df["bounce_rate"].std(ddof=0)
    df_quality = df.copy()
    df_quality["quality_score"] = eng_z - bounce_z  # higher engagement, lower bounce

    size_scale = 400 / df_quality["Users"].max() if not df_quality.empty else 1
    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(
        df_quality["quality_score"],
        df_quality["weighted_conversion_rate"],
        s=df_quality["Users"] * size_scale,
        c=df_quality["Users"],
        cmap="Blues",
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )
    sns.regplot(
        data=df_quality,
        x="quality_score",
        y="weighted_conversion_rate",
        scatter=False,
        line_kws={"color": "#E15759"},
        ax=ax,
    )
    r = df_quality["quality_score"].corr(df_quality["weighted_conversion_rate"])
    interpretation = (
        "no clear correlation"
        if abs(r) < 0.1
        else "weak positive"
        if 0.1 <= r < 0.3
        else "moderate positive"
        if 0.3 <= r < 0.5
        else "strong positive"
        if r >= 0.5
        else "negative"
    )
    ax.text(
        0.02,
        0.98,
        f"r(quality, conversion) = {r:.2f} ({interpretation})\nSize/Color = Users (volume)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )
    ax.set_xlabel("Quality score (engagement z-score minus bounce z-score)")
    ax.set_ylabel("Conversion rate (%)")
    ax.set_title("Quality-Volume-Outcome: content quality vs conversion (bubble = traffic)")
    fig.colorbar(scatter, ax=ax, label="Users (traffic)")

    output = ROOT / "quality_volume_outcome.png"
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    return output


def plot_bounce_traffic_conversion(df: pd.DataFrame) -> Path:
    """Bounce + traffic vs conversion: are popular posts also converting?"""
    filtered = df[df["Users"] <= 5000]
    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(
        filtered["Users"],
        filtered["weighted_conversion_rate"],
        c=filtered["bounce_rate"],
        cmap="RdYlGn_r",
        alpha=0.8,
        s=70,
        edgecolor="black",
        linewidth=0.5,
    )
    sns.regplot(
        data=filtered,
        x="Users",
        y="weighted_conversion_rate",
        scatter=False,
        line_kws={"color": "#E15759"},
        ax=ax,
    )
    r_tc = filtered["Users"].corr(filtered["weighted_conversion_rate"])
    r_bounce_conv = filtered["bounce_rate"].corr(filtered["weighted_conversion_rate"])
    interp_tc = (
        "no clear correlation"
        if abs(r_tc) < 0.1
        else "weak positive"
        if 0.1 <= r_tc < 0.3
        else "moderate positive"
        if 0.3 <= r_tc < 0.5
        else "strong positive"
        if r_tc >= 0.5
        else "negative"
    )
    interp_bc = (
        "no clear correlation"
        if abs(r_bounce_conv) < 0.1
        else "weak positive"
        if 0.1 <= r_bounce_conv < 0.3
        else "moderate positive"
        if 0.3 <= r_bounce_conv < 0.5
        else "strong positive"
        if r_bounce_conv >= 0.5
        else "negative"
    )
    ax.text(
        0.02,
        0.98,
        f"r(traffic, conversion) = {r_tc:.2f} ({interp_tc})\n"
        f"r(bounce, conversion) = {r_bounce_conv:.2f} ({interp_bc})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )
    ax.set_xlabel("Users (traffic)")
    ax.set_ylabel("Conversion rate (%)")
    ax.set_title("Bounce + Traffic → Conversion rate")
    fig.colorbar(scatter, ax=ax, label="Bounce rate (%)")

    output = ROOT / "bounce_traffic_conversion.png"
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    return output


def build_author_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate stats per author."""
    grouped = df.groupby("Author").agg(
        Users=("Users", "sum"),
        Signups=("Signups", "sum"),
        weighted_conversion_rate=("weighted_conversion_rate", "mean"),
        engagement_seconds=("engagement_seconds", "mean"),
        bounce_rate=("bounce_rate", "mean"),
    )
    return grouped


def main() -> None:
    sns.set_theme(style="whitegrid")
    df = load_data()
    author_stats = build_author_stats(df)

    generated: Dict[str, Tuple[str, Path]] = {}
    generated["author_traffic_signup_conversion"] = plot_author_traffic_signup_conversion(author_stats)
    generated["author_traffic_signup_conversion_log"] = plot_author_traffic_signup_conversion_log(author_stats)
    generated["most_efficient_author"] = plot_most_efficient_author(author_stats)
    generated["traffic_vs_signup_corr"] = plot_traffic_vs_signup(df)
    generated["engagement_vs_bounce"] = plot_engagement_vs_bounce(author_stats)
    generated["engagement_signup_conversion"] = plot_engagement_signup_conversion(df)
    generated["posts_needing_conversion_help"] = plot_posts_needing_conversion_help(df)
    generated["quadrant_users_vs_conversion"] = plot_quadrant_users_vs_conversion(author_stats)
    generated["quadrant_posts_users_vs_conversion"] = plot_quadrant_posts_users_vs_conversion(df)
    generated["correlation_conversion_engagement"] = plot_correlation_conversion_engagement(df)
    generated["engagement_conversion_bounce_interaction"] = plot_engagement_conversion_bounce(df)
    generated["author_heatmap_conversion_engagement"] = plot_author_heatmap_conversion_engagement(author_stats)
    generated["quality_volume_outcome"] = plot_quality_volume_outcome(df)
    generated["bounce_traffic_conversion"] = plot_bounce_traffic_conversion(df)
    correlation_table, correlation_table_path = save_correlation_table(df)

    notes = marketing_notes()
    print("\nMarketing POV (why each plot matters):")
    for key, path in generated.items():
        print(f"- {key}: {notes[key]} (saved to {path.name})")
    print(f"- correlation_conversion_engagement_table: saved to {correlation_table_path.name}")
    # Print correlation table as markdown in terminal.
    print("\nCorrelation table (filtered engagement_seconds <= 170):")
    headers = correlation_table.columns.tolist()
    rows = correlation_table.values.tolist()
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    value_line = "| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in rows[0]) + " |"
    print("\n".join([header_line, separator, value_line]))

    try:
        plt.show()
    except Exception:
        # In headless environments, show may fail; images are already saved.
        pass


if __name__ == "__main__":
    main()
