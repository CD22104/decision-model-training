"""A stand-in for `ojh` in tests: `run` writes a finished harness-style run directory at once, `check` prints ok.

Environment:
  FAKE_OJH_OUT      where run directories go
  FAKE_OJH_CALLS    a file that gets one line per `run` call
  FAKE_OJH_QUALITY  JSON {scheme key: strength or "fail"}; the key is "<base>" plus "|<override>=<value>" for each
                    override, sorted; unknown keys have strength 1.0. Higher strength separates the classes better.
  FAKE_OJH_PENDING  "1": leave the run unfinished (no DONE)
"""
import argparse
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path

import yaml


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("exp")
    ap.add_argument("--base")
    ap.add_argument("--recipe")
    ap.add_argument("--gpus")
    ap.add_argument("--machine")
    ap.add_argument("--allow-shared", action="store_true")
    a = ap.parse_args()
    exp = yaml.safe_load(Path(a.exp).read_text())
    if a.cmd == "check":
        print(json.dumps({"ok": True, "cards_checked": False, "base": a.base}))
        return 0
    with open(os.environ["FAKE_OJH_CALLS"], "a") as f:
        f.write(f"{exp['name']}\n")
    seed = exp["seeds"][0]
    ovs = {k: v["value"] for k, v in (exp.get("overrides") or {}).items()}
    key = "|".join([a.base] + [f"{k}={ovs[k]}" for k in sorted(ovs)])
    quality = json.loads(os.environ.get("FAKE_OJH_QUALITY", "{}")).get(key, 1.0)
    root = Path(os.environ["FAKE_OJH_OUT"]) / exp["name"]
    n = len(list(root.glob("r*"))) if root.is_dir() else 0
    rd = root / f"r{n}"
    rd.mkdir(parents=True)
    recipe = Path(a.recipe)
    (rd / "run.lock.yaml").write_text(yaml.safe_dump({
        "base": {"name": a.base}, "experiment": exp["name"],
        "recipe": {"path": str(recipe), "sha256": hashlib.sha256(recipe.read_bytes()).hexdigest()},
        "overrides": exp.get("overrides") or {}}))
    gpus = [int(g) for g in a.gpus.split(",")]
    (rd / "gpus.json").write_text(json.dumps([{"uuid": f"GPU-{g}", "name": "NVIDIA H200"} for g in gpus]))
    out = {"run_dir": str(rd), "session": f"ojh-{exp['name']}", "log": str(rd / "runner.log"),
           "done": str(rd / "DONE"), "result": str(rd / "RESULT.md"), "base": a.base, "recipe": str(recipe),
           "gpus": gpus}
    print("prepare: ok", file=sys.stderr)
    print(json.dumps(out, indent=2))
    if os.environ.get("FAKE_OJH_PENDING") == "1":
        return 0
    if quality == "fail":
        (rd / "timing.json").write_text(json.dumps({"seeds": {str(seed): {"seconds": 3600}}}))
        (rd / "DONE").write_text(json.dumps({"exit_code": 1, "stage": "train", "error": "CUDA out of memory"}))
        return 0
    rng = random.Random(f"{key}|{seed}")
    lines = []
    for i in range(300):
        y = 1 if i % 10 == 0 else 0
        z = quality * (2 * y - 1) + rng.gauss(0, 1)
        p = 1 / (1 + math.exp(-z))
        lines.append(json.dumps({"seed": seed, "step": 800, "id": f"a{i}", "labels": ["yes", "no"],
                                 "probs": [p, 1 - p], "target": [float(y), float(1 - y)], "positive": p}))
    (rd / "predictions_select.jsonl").write_text("\n".join(lines) + "\n")
    (rd / "result.json").write_text(json.dumps({
        "wall_time": {"seconds": 7200, "gpu_hours": 2.0 * len(gpus)},
        "gpus": {"n": len(gpus), "cards": [{"uuid": f"GPU-{g}", "name": "NVIDIA H200"} for g in gpus]},
        "seeds": {str(seed): {"selected_step": 800, "checkpoint": "checkpoints/step-800",
                              "curve": [{"step": s, "nll": 0.1} for s in (400, 800, 1200)]}}}))
    (rd / "DONE").write_text(json.dumps({"exit_code": 0, "stage": "finalize"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
