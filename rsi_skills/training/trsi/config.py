"""Run configuration (run.yaml): the environment (a harness exp.yaml), the start recipe, the search space the user
approved, GPUs, the comparison rules and the budget. Relative paths are relative to the run file.
Every key is documented in configs/run.example.yaml."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .candidate import Candidate, CandidateError, parse_candidate, resolve_path
from .metrics import HIGHER_IS_BETTER

OVERRIDE_PREFIXES = ("onejev.", "lora.")
# Flags the harness derives itself and refuses as overrides (training-with-onejev SKILL.md); the training mode
# (mode, fsdp) changes only through a recipe, and the GPU count through the recipe's gpus.
DERIVED = {"full_finetune", "fsdp", "attention", "readout", "eval_every", "save_every", "seed", "model"}


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class OjhSettings:
    bin: str                       # the ojh executable, e.g. $OJH_REPO/.venv/bin/ojh
    env_file: Path | None = None   # sourced in the same shell command before ojh (harness README "环境配置")
    machine: Path | None = None    # optional harness machine.yaml


@dataclass(frozen=True)
class OverrideSpace:
    values: tuple = ()
    range: tuple[float, float] | None = None
    dead: dict = field(default_factory=dict)   # value -> reason it is a dead end


@dataclass(frozen=True)
class Space:
    bases: tuple[str, ...]
    recipes: tuple[Path, ...]
    overrides: dict[str, OverrideSpace]


@dataclass(frozen=True)
class GpuPolicy:
    ids: tuple[int, ...]
    allow_shared: bool = False


@dataclass(frozen=True)
class MetricSpec:
    primary: str = "auprc"
    guards: dict[str, float] = field(default_factory=lambda: {"ece": 0.02, "brier": 0.005})


@dataclass(frozen=True)
class Rules:
    seeds: tuple[int, ...] = (0, 1, 2)   # the first one is the probe seed
    probe_margin: float = 0.035
    accept_margin: float = 0.035


@dataclass(frozen=True)
class RunConfig:
    path: Path
    run_name: str
    out_dir: Path
    exp: Path
    ojh: OjhSettings
    start: Candidate
    space: Space
    gpus: GpuPolicy
    metric: MetricSpec
    rules: Rules
    budget_gpu_h: float
    gpu_factor: dict[str, float]       # substring of the GPU name -> A100-equivalent factor (default 1)
    estimate_gpu_h: dict[str, float]   # base -> expected GPU hours of one seed, before any result exists

    @property
    def home(self) -> Path:
        """Where this run keeps its records: <out_dir>/<run_name>/."""
        return self.out_dir / self.run_name


def _need(d: dict, key: str, where: str):
    if key not in d or d[key] in (None, ""):
        raise ConfigError(f"{where}: missing '{key}'")
    return d[key]


def _float(v, where: str, lo: float | None = None) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ConfigError(f"{where} must be a number (got {v!r})")
    if lo is not None and v < lo:
        raise ConfigError(f"{where} must be >= {lo} (got {v})")
    return float(v)


def _space(d: dict, base_dir: Path) -> Space:
    bases = tuple(str(b) for b in _need(d, "bases", "space"))
    recipes = []
    for r in _need(d, "recipes", "space"):
        p = resolve_path(base_dir, r)
        if not p.is_file():
            raise ConfigError(f"space.recipes: {p} does not exist")
        recipes.append(p)
    ovs = {}
    for k, s in (d.get("overrides") or {}).items():
        k = str(k)
        if not k.startswith(OVERRIDE_PREFIXES):
            raise ConfigError(f"space.overrides.{k}: only onejev.<flag> and lora.<field> can be searched; the training "
                              "mode and GPU count change through recipes")
        if k.split(".", 1)[1] in DERIVED:
            raise ConfigError(f"space.overrides.{k}: the harness derives this flag itself and refuses it as an override")
        s = s or {}
        values = tuple(s.get("values") or ())
        rng = s.get("range")
        if rng is not None:
            if not (isinstance(rng, list) and len(rng) == 2):
                raise ConfigError(f"space.overrides.{k}.range must be [low, high]")
            rng = (_float(rng[0], f"space.overrides.{k}.range"), _float(rng[1], f"space.overrides.{k}.range"))
        if not values and rng is None:
            raise ConfigError(f"space.overrides.{k}: give values or range")
        # keys may arrive as strings (e.g. JSON); read them as YAML scalars so "3" matches the value 3
        dead = {(yaml.safe_load(v) if isinstance(v, str) else v): str(why) for v, why in (s.get("dead") or {}).items()}
        ovs[k] = OverrideSpace(values, rng, dead)
    return Space(bases, tuple(recipes), ovs)


def load_run(path: Path) -> RunConfig:
    path = Path(path).resolve()
    d = yaml.safe_load(path.read_text()) or {}
    base_dir = path.parent
    name = str(_need(d, "run_name", "run"))
    if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
        raise ConfigError(f"run_name {name!r}: use letters, digits, '.', '_' and '-' only")
    try:
        out_dir = resolve_path(base_dir, _need(d, "out_dir", "run"))
        exp = resolve_path(base_dir, _need(d, "exp", "run"))
        o = _need(d, "ojh", "run")
        ojh = OjhSettings(os.path.expandvars(os.path.expanduser(str(_need(o, "bin", "ojh")))),  # unset vars: see harness
                          resolve_path(base_dir, o["env_file"]) if o.get("env_file") else None,
                          resolve_path(base_dir, o["machine"]) if o.get("machine") else None)
        start = parse_candidate(_need(d, "start", "run"), base_dir, "start")
        space = _space(_need(d, "space", "run"), base_dir)
    except CandidateError as e:
        raise ConfigError(str(e)) from None
    if not exp.is_file():
        raise ConfigError(f"exp {exp} does not exist")
    g = _need(d, "gpus", "run")
    ids = tuple(int(i) for i in _need(g, "ids", "gpus"))
    gpus = GpuPolicy(ids, bool(g.get("allow_shared", False)))
    m = d.get("metric") or {}
    primary = str(m.get("primary", "auprc"))
    if not HIGHER_IS_BETTER.get(primary):
        raise ConfigError(f"metric.primary must be one of auprc, auc (got {primary!r})")
    guards = {str(k): _float(v, f"metric.guards.{k}", 0) for k, v in (m.get("guards") or {"ece": 0.02, "brier": 0.005}).items()}
    for k in guards:
        if HIGHER_IS_BETTER.get(k) is not False:
            raise ConfigError(f"metric.guards.{k}: guards are lower-is-better metrics (ece, brier)")
    r = d.get("rules") or {}
    seeds = tuple(int(s) for s in r.get("seeds", (0, 1, 2)))
    if not seeds or len(set(seeds)) != len(seeds):
        raise ConfigError("rules.seeds must be distinct and non-empty")
    rules = Rules(seeds, _float(r.get("probe_margin", 0.035), "rules.probe_margin", 0),
                  _float(r.get("accept_margin", 0.035), "rules.accept_margin", 0))
    b = _need(d, "budget", "run")
    return RunConfig(
        path=path, run_name=name, out_dir=out_dir, exp=exp, ojh=ojh, start=start, space=space, gpus=gpus,
        metric=MetricSpec(primary, guards), rules=rules,
        budget_gpu_h=_float(_need(b, "gpu_hours", "budget"), "budget.gpu_hours", 0),
        gpu_factor={str(k): _float(v, f"budget.gpu_factor.{k}", 0) for k, v in (b.get("gpu_factor") or {}).items()},
        estimate_gpu_h={str(k): _float(v, f"budget.estimate.{k}", 0) for k, v in (b.get("estimate") or {}).items()},
    )
