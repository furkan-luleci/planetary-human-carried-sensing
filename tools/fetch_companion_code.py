"""Fetch the definitions this study uses from the companion code of the earlier simulation study.

Two modules of this repository, src/planetsense/evaluation/legacy_pipeline.py (spectral processing and
gait-harmonic removal) and src/planetsense/evaluation/legacy_oi.py (the Observability Index), are
character-for-character copies of definitions in the earlier study's public repository
(Babazade and Lüleci, human-carried-indirect-sensing). That code has its own licence, so the copies are
not distributed here: this script clones the repository and writes them.

    python tools/fetch_companion_code.py

Options: --repo (default below), --commit (default below), --dest (where to clone; default
.companion/ inside this repository, which git ignores).
"""
import argparse
import ast
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO = "https://github.com/furkan-luleci/human-carried-indirect-sensing"
COMMIT = "962af3e"

OI_NAMES = ["OBS_T1", "OBS_T2", "obs_class", "OBS_SIGMA", "OBS_WALK_WIDTHS", "display_smooth", "_curve_area",
            "OBS_TIE", "obs_curve_mode", "obs_oi", "obs_curve_device"]
OI_HEADER = '''"""Observability Index of the earlier simulation study, copied verbatim.

The functions and constants below are copied character for character from src/human_di.py of the
earlier study's repository (written by tools/fetch_companion_code.py; original line numbers are given
above each definition), so that the index can be computed on this project's signals without importing
the earlier study's package and its plotting dependencies. Nothing here was edited.

Use in this project: a diagnostic link to the earlier Earth study. The index and its class thresholds
(27.5 / 92.3) were defined and calibrated for Earth; they are not re-tuned for Mars or the Moon.
"""
import numpy as np

'''


def segments(path, names):
    src = open(path, encoding="utf-8").read()
    out = {}
    for node in ast.parse(src).body:
        nm = None
        if isinstance(node, ast.FunctionDef):
            nm = node.name
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            nm = node.targets[0].id
        if nm in names:
            out[nm] = (node.lineno, ast.get_source_segment(src, node))
    missing = [n for n in names if n not in out]
    if missing:
        raise SystemExit(f"not found in {path}: {missing}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--commit", default=COMMIT)
    ap.add_argument("--dest", default=os.path.join(ROOT, ".companion"))
    a = ap.parse_args()
    if not os.path.isdir(os.path.join(a.dest, ".git")):
        subprocess.run(["git", "clone", "--quiet", a.repo, a.dest], check=True)
    subprocess.run(["git", "-C", a.dest, "checkout", "--quiet", a.commit], check=True)

    hd = os.path.join(a.dest, "src", "human_di.py")
    seg = segments(hd, OI_NAMES)
    lines = [OI_HEADER.rstrip("\n"), ""]
    for nm, (ln, s) in sorted(seg.items(), key=lambda kv: kv[1][0]):
        lines += [f"# human_di.py line {ln}", s, ""]
    out = os.path.join(ROOT, "src", "planetsense", "evaluation", "legacy_oi.py")
    open(out, "w", encoding="utf-8").write("\n".join(lines).rstrip() + "\n")
    print("wrote", os.path.relpath(out, ROOT), len(seg), "definitions")

    # the spectral processing, written by the study's own vendoring script
    subprocess.run([sys.executable, os.path.join(ROOT, "experiments", "phase_I_baseline_match", "vendor_pipeline.py"),
                    "--base", a.dest], check=True)


if __name__ == "__main__":
    main()
