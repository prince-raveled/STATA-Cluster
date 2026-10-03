"""Does v3, under rule set v2, reproduce the v2 engine exactly? — plan §26.5.

The claim is environment-relative: the same v2 engine does not write the same bytes on
Windows and Linux, nor under every Python build (docs/V3_DEVIATIONS.md B1, B1a, B1b).
Hashes recorded on one machine therefore cannot be checked on another — CI's runners
change underneath them. So this runs both engines here, in one environment, on one set
of inputs, and compares their outputs byte for byte:

  1. check out the v2 engine (commit 8d1cd9f) into a temporary git worktree;
  2. copy this tree's demo datasets into it, so both engines read identical files;
  3. run the v2 engine there, in a subprocess with this interpreter;
  4. run this tree with ruleset="v2";
  5. compare the SHA-256 of all 18 outputs (three demos x six files).

    python tests/reference/compare_v2_engine.py
    python tests/reference/compare_v2_engine.py --v2-tree ../microverse-v2

Needs git and the v2 commit in the clone (CI checks out with fetch-depth 0), and the demo
datasets (python examples/make_examples.py).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests", "fixtures"))

import make_v2_golden  # noqa: E402

V2_COMMIT = "8d1cd9f"

#: Runs inside the v2 checkout. The v2 engine predates `ruleset`, so it is called as v2
#: called itself; the outputs are the ones make_v2_golden.outputs() builds.
CHILD = r"""
import gzip, hashlib, json, sys
sys.path.insert(0, ".")
from app.core.attribution import attribute
from app.core.report import (attribution_csv, long_results_csv_gz, robustness_csv,
                             specifications_csv)
from app.core.robustness import compute_robustness, verdict_sentence
from app.core.runner import run_multiverse
from app.services import load_demo

out = {}
for demo, mode in json.loads(sys.argv[1]):
    dataset = load_demo(demo)
    covariates = list(dataset.covariate_columns) if mode == "covariate" else ()
    run = run_multiverse(dataset, mode=mode, covariate_columns=covariates)
    summary = compute_robustness(run)
    attribution = attribute(run)
    files = {
        "robustness.csv": robustness_csv(summary),
        "specifications.csv": specifications_csv(run, summary),
        "results_long.csv": gzip.decompress(long_results_csv_gz(run)),
        "attribution.csv": attribution_csv(attribution),
        "verdict.txt": verdict_sentence(run, summary).encode("utf-8"),
        "grid.json": json.dumps({
            "n_enumerated": run.grid_report.n_enumerated,
            "n_valid": run.grid_report.n_valid,
            "n_pruned": run.grid_report.n_pruned,
            "pruned_reasons": run.grid_report.pruned_reasons,
            "tier_counts": summary.tier_counts,
        }, indent=2, sort_keys=True).encode("utf-8"),
    }
    for name, data in files.items():
        out[f"{demo}_{mode}/{name}"] = hashlib.sha256(data).hexdigest()
print(json.dumps(out))
"""


def _git(*args, cwd=ROOT) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True,
                          text=True).stdout.strip()


def _copy_examples(v2_tree: str) -> None:
    """Both engines read this tree's demo files, so the inputs are identical."""
    source = os.path.join(ROOT, "examples")
    target = os.path.join(v2_tree, "examples")
    os.makedirs(target, exist_ok=True)
    copied = 0
    for name in os.listdir(source):
        if name.endswith((".tsv", ".json")):
            shutil.copy2(os.path.join(source, name), os.path.join(target, name))
            copied += 1
    if not copied:
        raise SystemExit("No demo datasets in examples/. Run: python examples/make_examples.py")


def v2_hashes(v2_tree: str) -> dict:
    _copy_examples(v2_tree)
    env = dict(os.environ)
    # The v2 engine opens its own job database and storage folder; keep them inside the
    # throwaway checkout rather than wherever the caller's settings point.
    for key in [k for k in env if k.startswith("MICROVERSE_") or k == "DATABASE_URL"]:
        env.pop(key)
    result = subprocess.run(
        [sys.executable, "-c", CHILD, json.dumps(make_v2_golden.CASES)],
        cwd=v2_tree, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"The v2 engine failed:\n{result.stderr[-3000:]}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def v3_hashes() -> dict:
    out = {}
    for demo, mode in make_v2_golden.CASES:
        for name, data in make_v2_golden.outputs(demo, mode, ruleset="v2").items():
            out[f"{demo}_{mode}/{name}"] = hashlib.sha256(data).hexdigest()
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-tree", default="",
                        help="an existing checkout of the v2 engine; by default a "
                             f"temporary worktree at {V2_COMMIT} is made and removed")
    args = parser.parse_args(argv)

    made = ""
    v2_tree = args.v2_tree
    if not v2_tree:
        made = tempfile.mkdtemp(prefix="microverse-v2-")
        os.rmdir(made)                                  # git wants to create it itself
        _git("worktree", "add", "--detach", made, V2_COMMIT)
        v2_tree = made
    try:
        print(f"v2 engine: {_git('rev-parse', '--short', 'HEAD', cwd=v2_tree)}; "
              f"v3 tree: {_git('rev-parse', '--short', 'HEAD')}; Python "
              f"{sys.version.split()[0]} on {sys.platform}")
        expected = v2_hashes(v2_tree)
        actual = v3_hashes()
    finally:
        if made:
            _git("worktree", "remove", "--force", made)

    mismatched = []
    for key in sorted(expected):
        same = actual.get(key) == expected[key]
        print(f"  [{'PASS' if same else 'FAIL'}] {key}")
        if not same:
            mismatched.append(key)
    if set(actual) != set(expected):
        print(f"  [FAIL] outputs differ in kind: {sorted(set(actual) ^ set(expected))}")
        mismatched.append("output set")
    print("\nAll outputs byte-identical." if not mismatched
          else f"\nFAILED: {len(mismatched)} output(s) differ from the v2 engine.")
    return 1 if mismatched else 0


if __name__ == "__main__":
    sys.exit(main())
