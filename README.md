# Blog Performance Analysis

Python tooling to explore blog performance by author and post, generate marketing-friendly visuals, and quantify key relationships (traffic, signups, conversion, engagement, bounce). All plots are saved to the project root.

## Contents
- `analyze_blog.py` — main analysis/plotting script.
- `blog.csv` — input dataset (authors, users, signups, conversion rate, engagement time, bounce).
- Generated outputs (PNG/CSV) — produced when you run the script.
- `analysis.md` — human-readable summary of findings per plot.

## Prerequisites
- Python 3.9+ recommended.
- pip-installed dependencies: `pandas`, `matplotlib`, `seaborn`.

## Setup
```bash
python3 -m pip install -r requirements.txt
```

## Usage
From the repo root:
```bash
python3 analyze_blog.py
```

What happens:
- Reads `blog.csv`, cleans percentage fields, and computes helper metrics (weighted conversion, engagement seconds, bounce rate as numeric).
- Generates all plots defined in `analyze_blog.py`, saving them to the root directory (e.g., `author_traffic_signup_conversion.png`, `quality_volume_outcome.png`, etc.).
- Prints brief marketing POV notes and a Markdown correlation table to the terminal.

## Key Outputs (PNG/CSV)
- `author_traffic_signup_conversion.png` — author traffic, signups, conversion.
- `author_traffic_signup_conversion_log.png` — same on log scale.
- `most_efficient_author.png` — conversion efficiency with engagement context.
- `traffic_vs_signup_corr.png` — traffic vs signups correlation (filtered).
- `engagement_vs_bounce.png` — stickiness by author.
- `engagement_signup_conversion.png` — engagement vs signups/conversion (filtered).
- `posts_needing_conversion_help.png` — high-traffic/low-conversion posts to fix.
- `quadrant_users_vs_conversion.png` — author quadrants (traffic vs conversion).
- `quadrant_posts_users_vs_conversion.png` — post quadrants (traffic vs conversion, filtered).
- `correlation_conversion_engagement.png` — engagement vs conversion correlation (filtered).
- `engagement_conversion_bounce_interaction.png` — engagement vs conversion colored by bounce.
- `author_heatmap_conversion_engagement.png` — heatmap of author conversion/engagement.
- `quality_volume_outcome.png` — combined quality score vs conversion (bubble = traffic).
- `bounce_traffic_conversion.png` — traffic vs conversion colored by bounce (filtered).
- `correlation_conversion_engagement.csv` — Pearson correlation table (filtered).
- See `analysis.md` for narrative summaries and recommendations per plot.

## Notes
- Headless environments: the script forces a non-interactive matplotlib backend and will skip `plt.show()` errors; images are still saved.
- If you see font/cache warnings, set `MPLCONFIGDIR` to a writable path (e.g., `export MPLCONFIGDIR=/tmp/mplconfig`).
- Modify filtering thresholds or add new plots directly in `analyze_blog.py`.

## Contributing
- Open PRs with new analyses, improved visual design, or additional tests. Add plot descriptions to `analysis.md` when you introduce new outputs.
