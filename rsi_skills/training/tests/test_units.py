"""Metrics, rules, the search space and the workspace exp.yaml (design tests 1, 2, 4, 5, 6, 8)."""
import json

import pytest
import yaml

from trsi import harness
from trsi.candidate import Candidate, Override, dead_ends, outside_space
from trsi.config import ConfigError, MetricSpec, Rules, load_run
from trsi.harness import TrialResult
from trsi.metrics import auc, auprc, brier, ece, evaluate
from trsi.rules import after_probe, decide


# ---- 4. metrics on hand-computed examples ----

def test_metrics_hand_examples():
    y, p = [1, 0, 1, 0], [0.9, 0.8, 0.4, 0.1]
    assert auprc(y, p) == pytest.approx(0.5 * 1 + 0.5 * 2 / 3)       # sklearn: 0.8333
    assert auc(y, p) == pytest.approx(0.75)
    assert brier(y, p) == pytest.approx((0.01 + 0.64 + 0.36 + 0.01) / 4)
    assert ece(y, p) == pytest.approx((0.1 + 0.8 + 0.6 + 0.1) / 4)    # one row per bin
    ties = [0.5] * 4
    assert auprc(y, ties) == pytest.approx(0.5) and auc(y, ties) == pytest.approx(0.5)
    assert evaluate(y, p)["positives"] == 2


def test_metrics_reject_bad_input():
    with pytest.raises(ValueError):
        evaluate([0, 0], [0.1, 0.2])        # no positive
    with pytest.raises(ValueError):
        evaluate([1, 0], [1.2, 0.1])        # not a probability


# ---- 5, 6. rules ----

def _r(seed, auprc_, ece_=0.02, brier_=0.02):
    return TrialResult("rd", seed, "done", {"auprc": auprc_, "ece": ece_, "brier": brier_})


RULES, METRIC = Rules(), MetricSpec()


def test_probe_margin():
    assert after_probe(_r(0, 0.913), _r(0, 0.883), RULES, METRIC)[0] == "reject"       # +0.030
    v, g = after_probe(_r(0, 0.923), _r(0, 0.883), RULES, METRIC)                       # +0.040
    assert v == "more_seeds" and g == pytest.approx(0.04)
    with pytest.raises(ValueError):
        after_probe(_r(1, 0.95), _r(0, 0.883), RULES, METRIC)


def test_decide_guards_and_accept():
    inc = {0: _r(0, 0.883), 1: _r(1, 0.905), 2: _r(2, 0.918)}
    better = {0: _r(0, 0.923), 1: _r(1, 0.945), 2: _r(2, 0.958)}
    ok, why = decide(better, inc, RULES, METRIC)
    assert ok and "3/3 higher" in why
    worse_ece = {s: _r(s, r.metrics["auprc"], ece_=0.05) for s, r in better.items()}     # ECE +0.03 > 0.02
    ok, why = decide(worse_ece, inc, RULES, METRIC)
    assert not ok and "ece +0.0300 (beyond 0.02)" in why
    small = {s: _r(s, r.metrics["auprc"] + 0.03) for s, r in inc.items()}
    assert not decide(small, inc, RULES, METRIC)[0]
    with pytest.raises(ValueError):
        decide({0: better[0]}, inc, RULES, METRIC)


# ---- 1. the approved space ----

def test_outside_space_and_dead_ends(env):
    run = load_run(env.run)
    el = env.root / "recipes" / "el_lora_1g.yaml"
    ok = Candidate("elephant-1", el, {"onejev.lr": Override(5e-5, "r"), "lora.r": Override(64, "r")})
    assert outside_space(ok, run.space) == []
    bad = Candidate("joey-1", env.root / "exp.yaml", {"onejev.lr": Override(3e-4, "r"), "onejev.epochs": Override(2, "r"),
                                                       "lora.r": Override(256, "r")})
    msgs = " | ".join(outside_space(bad, run.space))
    for part in ("base joey-1", "recipe", "onejev.lr = 0.0003", "onejev.epochs is not in", "lora.r = 256"):
        assert part in msgs
    dead = Candidate("elephant-1", el, {"onejev.brier_weight": Override(3.0, "r")})
    assert outside_space(dead, run.space) == [] and "dead end" in dead_ends(dead, run.space)[0]


def test_config_refuses_mode_and_derived_overrides(env):
    d = yaml.safe_load(env.run.read_text())
    d["space"]["overrides"]["mode"] = {"values": ["full"]}
    env.run.write_text(yaml.safe_dump(d))
    with pytest.raises(ConfigError, match="through recipes"):
        load_run(env.run)
    del d["space"]["overrides"]["mode"]
    d["space"]["overrides"]["onejev.attention"] = {"values": ["causal"]}
    env.run.write_text(yaml.safe_dump(d))
    with pytest.raises(ConfigError, match="derives"):
        load_run(env.run)


def test_launch_refusals(env):
    assert env.trsi("init", env.run)[0] == 0
    cases = [
        (env.proposal("a", base="joey-1"), "outside the approved space"),
        (env.proposal("b", overrides={"onejev.brier_weight": 3}), "dead end"),
        (env.proposal("c", prediction={"p_accept": 0.5}), "prediction must be"),
        (env.proposal("d", prediction={"auprc": 0.9, "p_accept": 1.5}), "p_accept must be in"),
        (env.proposal("e", note=""), "write a note"),
    ]
    for p, msg in cases:
        code, out, err = env.trsi("launch", env.run, p, "--seed", 0, "--gpus", 0)
        assert code == 2 and msg in err, (p.name, err)
    d = yaml.safe_load(env.proposal("f").read_text())
    d["overrides"] = {"onejev.lr": {"value": 5e-5}}
    (env.root / "f2.yaml").write_text(yaml.safe_dump(d))
    code, _, err = env.trsi("launch", env.run, env.root / "f2.yaml", "--seed", 0, "--gpus", 0)
    assert code == 2 and "needs a reason" in err
    code, _, err = env.trsi("launch", env.run, env.proposal("g"), "--seed", 0, "--gpus", 7)
    assert code == 2 and "not all approved" in err
    assert env.n_calls() == 0


# ---- 2. the workspace exp.yaml ----

def test_workspace(env):
    run = load_run(env.run)
    before = (env.root / "exp.yaml").read_text()
    c = Candidate("mammoth-1", env.root / "recipes" / "mm_lora_1g.yaml", {"onejev.lr": Override(5e-5, "27B needs less")})
    p = harness.write_workspace(run, c, 1)
    w = yaml.safe_load(p.read_text())
    assert (env.root / "exp.yaml").read_text() == before
    assert w["name"].startswith("elliptic-") and w["name"].endswith("-s1") and w["seeds"] == [1]
    assert w["overrides"] == {"onejev.lr": {"value": 5e-5, "reason": "27B needs less"}}
    assert w["data"]["train"] == str(env.root / "data" / "train.jsonl")
    assert w["data"]["eval"]["val"]["train_labels"] == str(env.root / "data" / "val.jsonl")
    assert w["fields"] == {"label": "target.yes", "group": "meta.state_id"} and w["evaluator"]["gate"] is True
    p0 = harness.write_workspace(run, c, 0)
    assert yaml.safe_load(p0.read_text())["name"] != w["name"]


def test_environment_with_overrides_is_refused(env):
    d = yaml.safe_load((env.root / "exp.yaml").read_text())
    d["overrides"] = {"onejev.lr": {"value": 1, "reason": "x"}}
    (env.root / "exp.yaml").write_text(yaml.safe_dump(d))
    code, _, err = env.trsi("init", env.run)
    assert code == 2 and "must not have overrides" in err
