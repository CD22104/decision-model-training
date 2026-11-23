"""The only module that talks to onejev-harness (ojh). Training goes through `ojh run`; results are read from the files
the harness writes in its run directory (DONE, result.json, predictions_select.jsonl, timing.json, gpus.json).
The harness itself is never modified."""
from __future__ import annotations

import copy
import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from .candidate import Candidate, file_sha256, trial_id
from .config import RunConfig
from .metrics import evaluate


class HarnessError(RuntimeError):
    pass


@dataclass
class TrialResult:
    run_dir: str
    seed: int
    status: str                      # done | failed
    metrics: dict = field(default_factory=dict)
    gpu_hours: float = 0.0           # as the harness measured it: training wall time x cards
    gpu_names: list = field(default_factory=list)
    selected_step: int | None = None
    last_step: int | None = None     # last evaluated step
    checkpoint: str | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrialResult":
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__ if k in d})


# ---- the environment ----------------------------------------------------------------------------------------

def load_exp(run: RunConfig) -> dict:
    exp = yaml.safe_load(run.exp.read_text()) or {}
    if exp.get("overrides"):
        raise HarnessError(f"{run.exp}: the environment exp.yaml must not have overrides; candidates bring their own")
    if not (exp.get("fields") or {}).get("label"):
        raise HarnessError(f"{run.exp}: fields.label is missing")
    return exp


def exp_sha256(run: RunConfig) -> str:
    return file_sha256(run.exp)


def positive_label(run: RunConfig) -> str:
    """The positive class: the last part of fields.label, e.g. 'yes' for target.yes."""
    return str(load_exp(run)["fields"]["label"]).rsplit(".", 1)[-1]


def _abs(base_dir: Path, v):
    s = str(v)
    if s.startswith(("/", "$", "~")):
        return s
    return str((base_dir / s).resolve())


def _absolutize(exp: dict, base_dir: Path) -> None:
    """The workspace exp.yaml lives elsewhere; make the paths the harness resolves relative to the exp file absolute."""
    data = exp.get("data") or {}
    for k in ("train", "split_file"):
        if data.get(k):
            data[k] = _abs(base_dir, data[k])
    if isinstance(data.get("adapter"), str) and data["adapter"].endswith(".py"):
        data["adapter"] = _abs(base_dir, data["adapter"])
    for s in (data.get("eval") or {}).values():
        for k in ("train_labels", "reference"):
            if isinstance(s, dict) and s.get(k):
                s[k] = _abs(base_dir, s[k])
    ev = exp.get("evaluator") or {}
    if ev.get("path"):
        ev["path"] = _abs(base_dir, ev["path"])


def write_workspace(run: RunConfig, c: Candidate, seed: int) -> Path:
    """<home>/work/<trial id>/exp.yaml: the environment's exp.yaml with a unique name, one seed and the candidate's
    overrides. The environment's own file is not changed."""
    tid = trial_id(c, seed, exp_sha256(run))
    exp = copy.deepcopy(load_exp(run))
    _absolutize(exp, run.exp.parent)
    exp["name"] = f"{exp.get('name', 'exp')}-{tid}"
    exp["seeds"] = [seed]
    exp["overrides"] = {k: {"value": o.value, "reason": o.reason} for k, o in sorted(c.overrides.items())}
    for k in ("base", "gpus", "recipe"):  # given on the ojh run command line
        exp.pop(k, None)
    d = run.home / "work" / tid
    d.mkdir(parents=True, exist_ok=True)
    out = d / "exp.yaml"
    tmp = out.with_suffix(".yaml.tmp")
    tmp.write_text("# written by trsi; the environment is " + str(run.exp) + "\n" + yaml.safe_dump(exp, sort_keys=False))
    os.replace(tmp, out)
    return out


# ---- running ojh ----------------------------------------------------------------------------------------------

def ojh_command(run: RunConfig, args: list[str]) -> list[str]:
    if run.ojh.env_file:
        # non-interactive shells do not read the user's profile: source the env file first, then expand variables
        # left in the ojh path (e.g. $OJH_REPO, set by that env file)
        script = 'source "$0" && b=$1 && shift && eval "b=\\"$b\\"" && exec "$b" "$@"'
        return ["bash", "-c", script, str(run.ojh.env_file), run.ojh.bin, *args]
    return [run.ojh.bin, *args]


def _json_out(text: str) -> dict:
    i = text.find("{")
    if i < 0:
        raise HarnessError(f"ojh printed no JSON:\n{text[-2000:]}")
    return json.loads(text[i:])


def _ojh(run: RunConfig, args: list[str]) -> dict:
    p = subprocess.run(ojh_command(run, args), capture_output=True, text=True)
    if p.returncode != 0:
        raise HarnessError(f"ojh {args[0]} exited with code {p.returncode}. Its message (tell the user; alternatives "
                           f"are the user's choice):\n{(p.stderr or p.stdout)[-4000:]}")
    return _json_out(p.stdout)


def recipe_gpus(recipe: Path) -> int:
    return int((yaml.safe_load(Path(recipe).read_text()) or {}).get("gpus", 1))


def _args(run: RunConfig, cmd: str, exp: Path, c: Candidate, gpus: list[int]) -> list[str]:
    args = [cmd, str(exp), "--base", c.base, "--recipe", str(c.recipe), "--gpus", ",".join(str(g) for g in gpus)]
    if run.ojh.machine:
        args += ["--machine", str(run.ojh.machine)]
    return args


def launch(run: RunConfig, c: Candidate, seed: int, gpus: list[int]) -> dict:
    """ojh run for one seed; returns immediately with ojh's JSON (run_dir, session, log, done, result, ...)."""
    exp = write_workspace(run, c, seed)
    args = _args(run, "run", exp, c, gpus)
    if run.gpus.allow_shared:
        args.append("--allow-shared")
    out = _ojh(run, args)
    if "run_dir" not in out:
        raise HarnessError(f"ojh run returned no run_dir: {out}")
    return out


def check(run: RunConfig, c: Candidate, seed: int, gpus: list[int]) -> dict:
    """ojh check: prepare, config, adapter, plan-only and environment checks, without the cards."""
    return _ojh(run, _args(run, "check", write_workspace(run, c, seed), c, gpus))


# ---- reading results ----------------------------------------------------------------------------------------

def state(run_dir: Path) -> str:
    """running | done | failed, from the harness's DONE marker."""
    done = Path(run_dir) / "DONE"
    if not done.is_file():
        return "running"
    try:
        return "done" if json.loads(done.read_text()).get("exit_code") == 0 else "failed"
    except json.JSONDecodeError:
        return "running"  # being written


def _read_json(p: Path) -> dict:
    return json.loads(p.read_text()) if p.is_file() else {}


def _cost(run_dir: Path, result: dict) -> tuple[float, list[str]]:
    cards = (result.get("gpus") or {}).get("cards") or _read_json(run_dir / "gpus.json") or []
    names = [c.get("name", "") for c in cards if isinstance(c, dict)]
    gh = (result.get("wall_time") or {}).get("gpu_hours")
    if gh is None:  # failed runs have no result.json: training seconds x cards
        secs = sum(float(s.get("seconds") or 0) for s in (_read_json(run_dir / "timing.json").get("seeds") or {}).values())
        gh = secs * max(len(names), 1) / 3600
    return float(gh), names


def read_result(run_dir: Path, positive: str, seed: int) -> TrialResult:
    """Metrics of the selected checkpoint on the select part of the eval sets, from predictions_select.jsonl."""
    run_dir = Path(run_dir)
    st = state(run_dir)
    result = _read_json(run_dir / "result.json")
    gh, names = _cost(run_dir, result)
    if st != "done":
        done = _read_json(run_dir / "DONE")
        return TrialResult(str(run_dir), seed, "failed", gpu_hours=gh, gpu_names=names,
                           error=f"stage {done.get('stage')}: {done.get('error')}")
    y, p = [], []
    with open(run_dir / "predictions_select.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if int(r.get("seed", seed)) != seed:
                continue
            labels = list(r["labels"])
            if positive not in labels:
                raise HarnessError(f"{run_dir}: label {positive!r} not among {labels}")
            y.append(int(round(float(r["target"][labels.index(positive)]))))
            p.append(float(r["positive"]) if "positive" in r else float(r["probs"][labels.index(positive)]))
    s = (result.get("seeds") or {}).get(str(seed)) or {}
    curve = s.get("curve") or []
    return TrialResult(str(run_dir), seed, "done", evaluate(y, p), gh, names, s.get("selected_step"),
                       max((e["step"] for e in curve), default=None), s.get("checkpoint"))
