"""A candidate training scheme: base model + harness recipe + overrides (each with a reason) + a note.

Its id is computed from content (base, recipe file contents, override values, environment exp.yaml contents), so the
same scheme is never trained twice; reasons and notes do not change the id.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from .config import RunConfig, Space


class CandidateError(ValueError):
    pass


@dataclass(frozen=True)
class Override:
    value: object
    reason: str


@dataclass(frozen=True)
class Candidate:
    base: str
    recipe: Path
    overrides: dict[str, Override] = field(default_factory=dict)
    note: str = ""

    def values(self) -> dict:
        return {k: o.value for k, o in sorted(self.overrides.items())}

    def label(self) -> str:
        """Short human-readable description, e.g. 'mammoth-1 lora_1g lr=5e-05'."""
        ov = " ".join(f"{k.split('.', 1)[-1]}={v}" for k, v in self.values().items())
        return " ".join(x for x in (self.base, self.recipe.stem, ov) if x)

    def to_dict(self) -> dict:
        return {"base": self.base, "recipe": str(self.recipe), "recipe_sha256": file_sha256(self.recipe),
                "overrides": {k: {"value": o.value, "reason": o.reason} for k, o in sorted(self.overrides.items())},
                "note": self.note}


@dataclass(frozen=True)
class Prediction:
    metric: str        # the run's primary metric, e.g. auprc
    value: float       # expected value of the primary metric for the probe seed
    p_accept: float    # probability that the candidate ends up accepted


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_path(base_dir: Path, value) -> Path:
    import os
    s = os.path.expandvars(os.path.expanduser(str(value)))
    if "$" in s:
        raise CandidateError(f"unset environment variable in path {value!r}")
    p = Path(s)
    return p if p.is_absolute() else (base_dir / p).resolve()


def parse_candidate(d: dict, base_dir: Path, where: str) -> Candidate:
    """Candidate from a mapping {base, recipe, overrides: {key: {value, reason}}, note}; recipe relative to base_dir."""
    if not isinstance(d, dict):
        raise CandidateError(f"{where}: expected a mapping")
    for k in ("base", "recipe"):
        if not d.get(k):
            raise CandidateError(f"{where}: missing '{k}'")
    ovs = {}
    for k, o in (d.get("overrides") or {}).items():
        if not isinstance(o, dict) or "value" not in o:
            raise CandidateError(f"{where}: override {k} must be {{value: ..., reason: ...}}")
        if not str(o.get("reason") or "").strip():
            raise CandidateError(f"{where}: override {k} needs a reason")
        ovs[str(k)] = Override(o["value"], str(o["reason"]).strip())
    recipe = resolve_path(base_dir, d["recipe"])
    if not recipe.is_file():
        raise CandidateError(f"{where}: recipe {recipe} does not exist")
    return Candidate(str(d["base"]), recipe, ovs, str(d.get("note") or "").strip())


def load_proposal(path: Path, run: "RunConfig") -> tuple[Candidate, Prediction]:
    """A proposal YAML: the candidate fields plus prediction: {<primary metric>: x, p_accept: q}.
    Recipe paths are relative to the run file, as in the run's search space."""
    path = Path(path)
    d = yaml.safe_load(path.read_text()) or {}
    c = parse_candidate(d, run.path.parent, str(path))
    if not c.note:
        raise CandidateError(f"{path}: write a note (which evidence it answers and why it may help)")
    pr = d.get("prediction")
    m = run.metric.primary
    if not isinstance(pr, dict) or not isinstance(pr.get(m), (int, float)) or not isinstance(pr.get("p_accept"), (int, float)):
        raise CandidateError(f"{path}: prediction must be {{{m}: <expected value for seed {run.rules.seeds[0]}>, "
                             "p_accept: <0..1>}}")
    if not 0 <= pr["p_accept"] <= 1:
        raise CandidateError(f"{path}: prediction.p_accept must be in [0, 1]")
    return c, Prediction(m, float(pr[m]), float(pr["p_accept"]))


def _norm(v):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return v
    return float(v)


def candidate_id(c: Candidate, exp_sha256: str) -> str:
    key = {"base": c.base, "recipe": file_sha256(c.recipe), "overrides": {k: _norm(v) for k, v in c.values().items()},
           "exp": exp_sha256}
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:10]


def trial_id(c: Candidate, seed: int, exp_sha256: str) -> str:
    return f"{candidate_id(c, exp_sha256)}-s{seed}"


def _same(a, b) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
    return a == b


def outside_space(c: Candidate, space: "Space") -> list[str]:
    """Why the candidate is not in the space the user approved; empty when it is."""
    out = []
    if c.base not in space.bases:
        out.append(f"base {c.base} is not approved (approved: {', '.join(space.bases)})")
    if c.recipe not in space.recipes:
        out.append(f"recipe {c.recipe} is not approved (approved: {', '.join(str(r) for r in space.recipes)})")
    for k, o in sorted(c.overrides.items()):
        s = space.overrides.get(k)
        if s is None:
            out.append(f"override {k} is not in the approved space")
            continue
        in_values = any(_same(o.value, v) for v in s.values)
        in_range = (s.range is not None and isinstance(o.value, (int, float)) and not isinstance(o.value, bool)
                    and s.range[0] <= o.value <= s.range[1])
        if not (in_values or in_range):
            allowed = ", ".join(repr(v) for v in s.values) + (f"; range {list(s.range)}" if s.range else "")
            out.append(f"override {k} = {o.value!r} is not approved (approved: {allowed})")
    return out


def dead_ends(c: Candidate, space: "Space") -> list[str]:
    """Approved values that are marked as dead ends, with their reasons."""
    out = []
    for k, o in sorted(c.overrides.items()):
        s = space.overrides.get(k)
        for v, why in (s.dead.items() if s else ()):
            if _same(o.value, v):
                out.append(f"{k} = {o.value!r} is a dead end: {why}")
    return out
