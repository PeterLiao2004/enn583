"""Plot feature correspondences written by ``match_features``.

Run from anywhere with, for example:

    python src/matches_plotting.py --frame-i 115 --frame-j 116
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SUPPORT_DIR = REPO_ROOT / "support"
if str(SUPPORT_DIR) not in sys.path:
    sys.path.insert(0, str(SUPPORT_DIR))

import kitti_utils as kitti


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", default="2011_09_26_drive_0035")
    parser.add_argument("--frame-i", type=int, default=115)
    parser.add_argument("--frame-j", type=int, default=116)
    parser.add_argument(
        "--matches", type=Path, default=REPO_ROOT / "results_matches.csv",
        help="CSV created by match_features (default: repository root)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output PNG (default: repository root/feature_matches_I_J.png)",
    )
    parser.add_argument(
        "--step", type=int, default=20,
        help="Plot every Nth match to avoid clutter (default: 20)",
    )
    parser.add_argument("--show", action="store_true", help="Open an interactive plot window")
    return parser.parse_args()


def load_matches(path: Path) -> list[dict[str, float]]:
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.is_file():
        raise FileNotFoundError(
            f"Match file not found: {path}\n"
            "Run match_features on the selected frame pair first."
        )

    required = {"u_i", "v_i", "u_j", "v_j"}
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
        return [{name: float(row[name]) for name in required} for row in reader]


def main() -> None:
    args = parse_args()
    if args.step < 1:
        raise ValueError("--step must be at least 1")

    dataset = kitti.load_kitti_dataset(name=args.sequence)
    for frame in (args.frame_i, args.frame_j):
        if not 0 <= frame < len(dataset):
            raise IndexError(f"Frame {frame} is outside the valid range 0..{len(dataset) - 1}")

    img_i = dataset.stereo(args.frame_i)[0]
    img_j = dataset.stereo(args.frame_j)[0]
    matches = load_matches(args.matches)
    print(f"Number of matches: {len(matches)}")

    if img_i.ndim == 2:
        img_i = cv.cvtColor(img_i, cv.COLOR_GRAY2RGB)
    if img_j.ndim == 2:
        img_j = cv.cvtColor(img_j, cv.COLOR_GRAY2RGB)

    combined = np.hstack((img_i, img_j))
    offset = img_i.shape[1]

    plt.figure(figsize=(16, 6))
    plt.imshow(combined)
    for row in matches[::args.step]:
        u_i, v_i = row["u_i"], row["v_i"]
        u_j, v_j = row["u_j"] + offset, row["v_j"]
        plt.plot([u_i, u_j], [v_i, v_j], linewidth=0.7)
        plt.scatter([u_i, u_j], [v_i, v_j], s=8)

    plt.axis("off")
    plt.title(f"Feature Matches: Frames {args.frame_i} → {args.frame_j}")
    plt.tight_layout()

    output = args.output or REPO_ROOT / f"feature_matches_{args.frame_i}_{args.frame_j}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output}")

    if args.show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()
