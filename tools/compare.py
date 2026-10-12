"""Compares the default simulations of two versions of the code.

Each version is exported from git (or taken from the working tree) and runs its own code: the key quantities of
`neural_pointers.summary` are printed side by side and the overview figures are stored next to each other.

Usage:
    python tools/compare.py [REF_A] [REF_B] [--out DIR]

`REF_A` defaults to the tag `poster-bonnbrain2024`, `REF_B` to `WORKTREE` (the uncommitted working tree). Only
versions that contain `neural_pointers/summary.py` can be compared.
"""

import argparse
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKTREE = "WORKTREE"
SUMMARY_CMD = (
    "import json; from neural_pointers.summary import summarize_all; "
    "print('SUMMARY=' + json.dumps(summarize_all('configs')))"
)


def checkout(ref, tmp):
    """Returns a directory containing the code of `ref`."""
    if ref == WORKTREE:
        return ROOT
    archive = subprocess.run(
        ["git", "archive", ref], cwd=ROOT, capture_output=True, check=True
    ).stdout
    target = Path(tmp, ref.replace("/", "_"))
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        if hasattr(tarfile, "data_filter"):
            tar.extractall(target, filter="data")
        else:
            tar.extractall(target)
    return target


def run(code_dir, args):
    env = dict(os.environ, PYTHONPATH=str(code_dir), MPLBACKEND="Agg")
    out = subprocess.run(
        [sys.executable, *args],
        cwd=code_dir,
        env=env,
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        raise RuntimeError(out.stderr)
    return out.stdout


def flatten(d, prefix=""):
    items = {}
    for k, v in d.items():
        if isinstance(v, dict):
            items.update(flatten(v, prefix + k + "."))
        else:
            items[prefix + k] = v
    return items


def side_by_side(paths, labels, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.image import imread

    images = [imread(p) for p in paths]
    fig, axes = plt.subplots(len(images), 1, figsize=(14, 7.5 * len(images)), dpi=100)
    for ax, image, label in zip(axes, images, labels):
        ax.imshow(image)
        ax.set_title(label, fontsize=16, loc="left")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("ref_a", nargs="?", default="poster-bonnbrain2024")
    parser.add_argument("ref_b", nargs="?", default=WORKTREE)
    parser.add_argument("--out", default=str(ROOT / "outputs" / "compare"))
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries, figures = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for ref in (args.ref_a, args.ref_b):
            code_dir = checkout(ref, tmp)
            stdout = run(code_dir, ["-c", SUMMARY_CMD])
            line = [l for l in stdout.splitlines() if l.startswith("SUMMARY=")][-1]
            summaries.append(flatten(json.loads(line[len("SUMMARY=") :])))
            figure = out_dir / "overview_{}.png".format(ref.replace("/", "_"))
            run(code_dir, ["plot_overview.py", str(figure)])
            figures.append(figure)

    (out_dir / "summary.json").write_text(
        json.dumps({args.ref_a: summaries[0], args.ref_b: summaries[1]}, indent=1)
    )
    side_by_side(figures, [args.ref_a, args.ref_b], out_dir / "overview_comparison.png")

    a, b = summaries
    print("{:55s} {:>30s} {:>30s}".format("quantity", args.ref_a, args.ref_b))
    for key in sorted(set(a) | set(b)):
        va, vb = a.get(key), b.get(key)
        marker = "" if va == vb else "  <-- changed"
        print("{:55s} {:>30s} {:>30s}{}".format(key, str(va), str(vb), marker))
    print("\nFigures and summaries stored in", out_dir)


if __name__ == "__main__":
    main()
