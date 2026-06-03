#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import h5py, numpy as np, pandas as pd, pyBigWig, logomaker, matplotlib.pyplot as plt

DEFAULT_WINDOW_SPAN = 2114
DEFAULT_OUTPUT_SPAN = 1000

ap = argparse.ArgumentParser(description="Render simplified tracker plots only.")
ap.add_argument("--region-config", required=True)
ap.add_argument("--track-config", required=True)
ap.add_argument("--output-dir", default="/Users/vinay/Documents/GitHub/shapley_masking/07_simplified_tracker/output")
ap.add_argument("--window-span", type=int, default=None)
ap.add_argument("--output-span", type=int, default=None)
args = ap.parse_args()

payload = json.loads(Path(args.region_config).read_text())
if "regions" not in payload or not payload["regions"]:
    raise ValueError("Region config contains no regions")

spans = {int(r["end"]) - int(r["start"]) for r in payload["regions"] if "end" in r and "start" in r}
if not spans:
    inferred = None
elif len(spans) > 1:
    raise ValueError(f"Region config contains multiple window spans: {sorted(spans)}")
else:
    inferred = spans.pop()
if args.window_span is None:
    window_span = inferred if inferred is not None else DEFAULT_WINDOW_SPAN
else:
    if inferred is not None and inferred != args.window_span:
        raise ValueError("Region config window span does not match --window-span")
    window_span = args.window_span
output_span = args.output_span or min(window_span, DEFAULT_OUTPUT_SPAN)

def sanitize(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s).lower()

track_payload = json.loads(Path(args.track_config).read_text())
tracks = []
for e in track_payload.get("tracks", []):
    t = e.get("file_type")
    if t == "bw":
        p = Path(e.get("file_location", ""))
        if str(p).startswith("01_download_data/raw_data/"):
            tracks.append({"type": "bw", "label": e.get("track_name", p.name), "path": p})
    elif t == "bpnet_lite_contribution_h5":
        h5 = Path(e.get("deeplift_h5", ""))
        top = Path(e.get("top_tsv", ""))
        task, mode = e.get("task"), e.get("mode")
        if not (str(h5).startswith("01_download_data/raw_data/") and h5.exists() and top.exists() and task and mode):
            continue
        ds = f"hyp_imp/{task}/profile/wn" if mode == "profile" else f"hyp_imp/{task}/counts/pre-act"
        df = pd.read_csv(top, sep="\t")
        idx_map = {(r.chrom, int(r.start), int(r.end)): int(r.peak_index) for r in df.itertuples(index=False)}
        tracks.append({"type": "h5", "label": e.get("track_name", f"{task} {mode}"), "h5": h5, "ds": ds, "index": idx_map})
if not tracks:
    raise ValueError("Track config has no supported tracks under 01_download_data/raw_data")

out_dir = Path(args.output_dir) / "indv_plots"
for r in payload["regions"]:
    chrom = r["chr"]; start = int(r["start"])
    end = int(r.get("end") or (start + window_span))
    name = r["region_name"]
    fname = f"{chrom}_{start}_{end}_{sanitize(name)}.png"
    x = np.arange(window_span)
    fig, axes = plt.subplots(len(tracks), 1, sharex=True, figsize=(22, max(6, 1.2 * len(tracks))))
    if len(tracks) == 1:
        axes = [axes]
    offset = max(0, (window_span - output_span) // 2)
    for i, tr in enumerate(tracks):
        ax = axes[i]
        if tr["type"] == "bw":
            if not tr["path"].exists():
                ax.text(0.5, 0.5, f"Missing: {tr['path']}", ha="center", va="center", fontsize=8); ax.set_axis_off(); continue
            with pyBigWig.open(str(tr["path"])) as bw:
                v = bw.values(chrom, start, end, numpy=True)
            a = np.nan_to_num(np.asarray(v, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            ax.plot(x, a, linewidth=0.8); ax.fill_between(x, 0, a, alpha=0.2)
        else:
            key = (chrom, start + offset, start + offset + output_span)
            idx = tr["index"].get(key)
            if idx is None:
                ax.text(0.5, 0.5, "Missing attribution for region", ha="center", va="center", fontsize=8); ax.set_axis_off(); continue
            with h5py.File(tr["h5"], "r") as h:
                hyp = h[tr["ds"]][idx].astype(np.float32)
                ohe = h["inputs/seq"][idx].astype(np.float32)
            contrib = (hyp * ohe).T
            full = np.zeros((4, window_span), dtype=np.float32)
            full[:, offset:offset + min(output_span, contrib.shape[1])] = contrib[:, :output_span]
            logomaker.Logo(pd.DataFrame(full.T, columns=["A", "C", "G", "T"]), ax=ax, color_scheme="classic", vpad=0.1)
            ax.axhline(0, color="black", linewidth=0.6)
        ax.set_ylabel(tr["label"], rotation=0, ha="right", va="center", labelpad=50, fontsize=8)
        ax.grid(alpha=0.2, linestyle="--", linewidth=0.5)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        if output_span < window_span:
            ax.axvline(offset, color="gray", linestyle="--", linewidth=0.6, alpha=0.6)
            ax.axvline(offset + output_span, color="gray", linestyle="--", linewidth=0.6, alpha=0.6)
    ticks = np.linspace(0, window_span - 1, 6, dtype=int)
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels([f"{start + int(pos):,}" for pos in ticks], rotation=45, ha="right", fontsize=8)
    axes[-1].set_xlabel(f"{chrom} coordinate")
    fig.suptitle(f"{name} ({chrom}:{start:,}-{end:,})", fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    out_path = out_dir / fname
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
