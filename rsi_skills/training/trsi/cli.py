"""python -m trsi <command> <run.yaml> ...

check    validate the run config (and with --ojh, `ojh check` the start without using cards)
init     make the start the incumbent
launch   train one seed of the start (--start) or of a proposal, through `ojh run`; returns immediately
adopt    register a finished harness run as one seed of the start or of a proposal
collect  read finished runs (metrics, GPU hours)
next     apply the rules (probe, more seeds, accept), update the incumbent, print what to do next
report   write <out_dir>/<run_name>/report.md
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

from . import __version__, archive, harness, report
from .candidate import (Candidate, CandidateError, candidate_id, dead_ends, file_sha256, load_proposal,
                        outside_space)
from .config import ConfigError, RunConfig, load_run
from .harness import HarnessError, TrialResult
from .rules import after_probe, decide


class Refused(RuntimeError):
    pass


# ---- cost ---------------------------------------------------------------------------------------------------

def _factor(run: RunConfig, names: list[str]) -> float:
    if not names:
        return 1.0
    fs = [next((f for k, f in run.gpu_factor.items() if k in n), 1.0) for n in names]
    return sum(fs) / len(fs)


def cost_of(run: RunConfig):
    def f(rec: dict) -> float:
        r = rec.get("result") or {}
        return float(r.get("gpu_hours") or 0) * _factor(run, r.get("gpu_names") or [])
    return f


def spent(run: RunConfig) -> float:
    c = cost_of(run)
    total = sum(c(rec) for rec in archive.trials(run).values() if rec["charged"] and not rec["start"])
    total += sum(e["gpu_h"] for e in archive.events(run, "retry") if e.get("charged"))
    return total


def estimate(run: RunConfig, c: Candidate) -> float | None:
    """Expected cost of one seed: mean of finished trials with the same base and recipe, else budget.estimate."""
    sha = file_sha256(c.recipe)
    cf = cost_of(run)
    seen = [cf(r) for r in archive.trials(run).values()
            if r["status"] == "done" and r["candidate"]["base"] == c.base and r["candidate"]["recipe_sha256"] == sha]
    est = sum(seen) / len(seen) if seen else run.estimate_gpu_h.get(c.base)
    if est is None:
        return None
    ep = c.overrides.get("onejev.epochs")
    if ep is not None:  # cost scales with epochs
        base_ep = float(((yaml.safe_load(c.recipe.read_text()) or {}).get("onejev") or {}).get("epochs", 1))
        est *= float(ep.value) / base_ep
    return est


# ---- helpers ------------------------------------------------------------------------------------------------

def _gpus(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _need_init(run: RunConfig) -> tuple[Candidate, str]:
    inc = archive.incumbent(run)
    if inc is None:
        raise Refused("run `python -m trsi init <run.yaml>` first")
    return inc


def _results(run: RunConfig) -> tuple[dict[str, dict[int, TrialResult]], dict[str, dict[int, dict]]]:
    done: dict[str, dict[int, TrialResult]] = defaultdict(dict)
    recs: dict[str, dict[int, dict]] = defaultdict(dict)
    for rec in archive.trials(run).values():
        recs[rec["candidate_id"]][rec["seed"]] = rec
        if rec["status"] == "done":
            done[rec["candidate_id"]][rec["seed"]] = TrialResult.from_dict(rec["result"])
    return done, recs


def _probe_event(run: RunConfig, cid: str, against: str) -> dict | None:
    return next((e for e in reversed(archive.events(run, "probe"))
                 if e["candidate"] == cid and e["against"] == against), None)


def _candidate(run: RunConfig, a) -> tuple[Candidate, object, bool]:
    if a.start:
        return run.start, None, True
    if not a.proposal:
        raise Refused("give a proposal YAML or --start")
    c, pred = load_proposal(Path(a.proposal), run)
    return c, pred, False


def _record_proposal(run: RunConfig, c: Candidate, cid: str, pred, path: str) -> None:
    if pred is None or any(e["candidate"] == cid for e in archive.events(run, "propose")):
        return
    archive.append(run, {"event": "propose", "candidate": cid, "candidate_dict": c.to_dict(), "proposal": path,
                         "prediction": {"metric": pred.metric, "value": pred.value, "p_accept": pred.p_accept}})


# ---- commands -----------------------------------------------------------------------------------------------

def cmd_check(a) -> int:
    run = load_run(Path(a.run))
    harness.load_exp(run)
    problems = [f"start: {p}" for p in outside_space(run.start, run.space)]
    print(f"run       {run.run_name} -> records in {run.home}")
    print(f"exp       {run.exp} (positive label {harness.positive_label(run)!r})")
    print(f"ojh       {run.ojh.bin}" + (f" (after sourcing {run.ojh.env_file})" if run.ojh.env_file else ""))
    print(f"start     {run.start.label()}")
    print(f"space     bases {list(run.space.bases)}; {len(run.space.recipes)} recipe(s); overrides "
          + (", ".join(f"{k} {list(s.values) or list(s.range)}" for k, s in run.space.overrides.items()) or "none"))
    for r in run.space.recipes:
        d = yaml.safe_load(r.read_text()) or {}
        print(f"recipe    {r.name}: mode {d.get('mode')}, {d.get('gpus', 1)} GPU(s)")
        if d.get("mode") not in ("full", "lora"):
            problems.append(f"recipe {r}: mode must be full or lora")
    for b in run.space.bases:
        if b not in run.estimate_gpu_h:
            print(f"warning   no budget.estimate for {b}: launch is refused until a trial of it has finished")
    print(f"gpus      {list(run.gpus.ids)}" + (" (shared cards allowed)" if run.gpus.allow_shared else ""))
    print(f"rules     {run.rules}; metric {run.metric.primary}, guards {run.metric.guards}; budget {run.budget_gpu_h} GPU h")
    for d in dead_ends(run.start, run.space):
        print(f"warning   start uses {d}")
    if a.ojh and not problems:
        n = harness.recipe_gpus(run.start.recipe)
        out = harness.check(run, run.start, run.rules.seeds[0], list(run.gpus.ids[:n]))
        print("ojh check " + json.dumps(out)[:400])
    for p in problems:
        print(f"PROBLEM   {p}")
    return 1 if problems else 0


def cmd_init(a) -> int:
    run = load_run(Path(a.run))
    harness.load_exp(run)
    if archive.incumbent(run):
        print(f"{run.home} is already initialised; the incumbent is in incumbent.yaml")
        return 0
    cid = candidate_id(run.start, harness.exp_sha256(run))
    (run.home / "proposals").mkdir(parents=True, exist_ok=True)
    archive.append(run, {"event": "init", "run_file": str(run.path), "run_sha256": file_sha256(run.path),
                         "exp_sha256": harness.exp_sha256(run), "trsi": __version__})
    archive.set_incumbent(run, run.start, cid, "start (the recipe the run begins from)")
    print(f"initialised {run.home}; incumbent = start `{cid}` {run.start.label()}")
    print(f"next: train the start's seeds {list(run.rules.seeds)} (not charged to the budget):")
    for s in run.rules.seeds:
        print(f"  python -m trsi launch {a.run} --start --seed {s} --gpus <ids>")
    print("or register finished harness runs of the same scheme: python -m trsi adopt <run.yaml> --start --seed S --run-dir <RUN>")
    return 0


def cmd_launch(a) -> int:
    run = load_run(Path(a.run))
    inc_c, inc_cid = _need_init(run)
    c, pred, is_start = _candidate(run, a)
    exp_sha = harness.exp_sha256(run)
    cid = candidate_id(c, exp_sha)
    is_start = is_start or cid == candidate_id(run.start, exp_sha)
    tid = f"{cid}-s{a.seed}"
    if not is_start:
        bad = outside_space(c, run.space)
        if bad:
            raise Refused("outside the approved space (the user approves the space in run.yaml):\n  " + "\n  ".join(bad))
        dead = dead_ends(c, run.space)
        if dead and not a.force:
            raise Refused("uses a dead end (add --force only with a reason the dead end does not apply):\n  "
                          + "\n  ".join(dead))
    if a.seed not in run.rules.seeds:
        raise Refused(f"seed {a.seed} is not in rules.seeds {list(run.rules.seeds)}")
    probe = run.rules.seeds[0]
    if not is_start and a.seed != probe and cid != inc_cid:
        pe = _probe_event(run, cid, inc_cid)
        if not (pe and pe["verdict"] == "more_seeds") and not a.force:
            raise Refused(f"seed {a.seed} comes after the probe: seed {probe} must beat the incumbent by more than "
                          f"{run.rules.probe_margin} first (run `trsi collect` and `trsi next`)")
    if any(e["trial"] == tid for e in archive.events(run, "use")):
        rec = archive.trial_record(run, tid)
        if rec and rec["status"] != "failed":
            print(f"{tid} is already part of this run ({rec['status']}): {rec['run_dir']}")
            return 0
    rec = archive.trial_record(run, tid)
    if rec and rec["status"] in ("running", "done"):
        _record_proposal(run, c, cid, pred, str(a.proposal))
        archive.append(run, {"event": "use", "trial": tid, "start": is_start, "charged": not is_start, "reused": True})
        print(f"reused {tid} ({rec['status']}) from {rec['run_dir']}; no new training")
        return 0
    gpus = _gpus(a.gpus)
    if not set(gpus) <= set(run.gpus.ids):
        raise Refused(f"cards {gpus} are not all approved (approved: {list(run.gpus.ids)})")
    need = harness.recipe_gpus(c.recipe)
    if len(gpus) != need:
        raise Refused(f"recipe {c.recipe.name} trains on {need} card(s); got {len(gpus)}")
    if not is_start:
        est = estimate(run, c)
        if est is None:
            raise Refused(f"no cost estimate for {c.base}: add budget.estimate.{c.base} to the run file")
        sp = spent(run)
        if sp + est > run.budget_gpu_h:
            raise Refused(f"budget: {sp:.1f} spent + about {est:.1f} for this seed > {run.budget_gpu_h:.0f} GPU hours; "
                          "stop and write the review")
    _record_proposal(run, c, cid, pred, str(a.proposal))
    if rec and rec["status"] == "failed":
        archive.append(run, {"event": "retry", "trial": tid, "gpu_h": cost_of(run)(rec), "charged": not is_start})
    out = harness.launch(run, c, a.seed, gpus)
    archive.write_trial(run, {"trial": tid, "candidate_id": cid, "candidate": c.to_dict(), "seed": a.seed,
                              "status": "running", "run_dir": out["run_dir"], "session": out.get("session"),
                              "log": out.get("log"), "attempts": (rec or {}).get("attempts", 0) + 1, "result": None})
    archive.append(run, {"event": "launch", "trial": tid, "run_dir": out["run_dir"], "gpus": gpus})
    archive.append(run, {"event": "use", "trial": tid, "start": is_start, "charged": not is_start})
    print(json.dumps(out, indent=2))
    print(f"launched {tid}. Hand back now; later run `python -m trsi collect {a.run}` (do not poll in a loop).")
    return 0


def cmd_adopt(a) -> int:
    run = load_run(Path(a.run))
    _need_init(run)
    c, pred, is_start = _candidate(run, a)
    rd = Path(a.run_dir).resolve()
    if a.seed not in run.rules.seeds:
        raise Refused(f"seed {a.seed} is not in rules.seeds {list(run.rules.seeds)}")
    lock = yaml.safe_load((rd / "run.lock.yaml").read_text())
    probs = []
    if lock["base"]["name"] != c.base:
        probs.append(f"base {lock['base']['name']} != {c.base}")
    if (lock.get("recipe") or {}).get("sha256") != file_sha256(c.recipe):
        probs.append("recipe contents differ")
    lov = {k: v.get("value") for k, v in (lock.get("overrides") or {}).items()}
    if lov != c.values():
        probs.append(f"overrides {lov} != {c.values()}")
    if probs:
        raise Refused(f"{rd} is not this scheme: " + "; ".join(probs))
    if harness.state(rd) != "done":
        raise Refused(f"{rd} has not finished successfully (DONE)")
    exp_sha = harness.exp_sha256(run)
    cid = candidate_id(c, exp_sha)
    is_start = is_start or cid == candidate_id(run.start, exp_sha)
    tid = f"{cid}-s{a.seed}"
    res = harness.read_result(rd, harness.positive_label(run), a.seed)
    _record_proposal(run, c, cid, pred, str(a.proposal))
    archive.write_trial(run, {"trial": tid, "candidate_id": cid, "candidate": c.to_dict(), "seed": a.seed,
                              "status": res.status, "run_dir": str(rd), "adopted": True, "result": res.to_dict()})
    archive.append(run, {"event": "adopt", "trial": tid, "run_dir": str(rd),
                         "note": "data and evaluator are not compared; adopt only runs of the same exp.yaml"})
    archive.append(run, {"event": "use", "trial": tid, "start": is_start, "charged": not is_start})
    print(f"adopted {rd} as {tid}: {run.metric.primary} {res.metrics.get(run.metric.primary):.4f}")
    return 0


def cmd_collect(a) -> int:
    run = load_run(Path(a.run))
    _need_init(run)
    pos = harness.positive_label(run)
    m = run.metric.primary
    for tid, used in archive.trials(run).items():
        rec = archive.trial_record(run, tid)
        if rec["status"] == "running":
            st = harness.state(Path(rec["run_dir"]))
            if st == "running":
                print(f"{tid}  running  {rec['run_dir']}")
                continue
            res = harness.read_result(Path(rec["run_dir"]), pos, rec["seed"])
            rec.update(status=res.status, result=res.to_dict())
            archive.write_trial(run, rec)
            archive.append(run, {"event": "result", "trial": tid, "status": res.status,
                                 m: res.metrics.get(m), "gpu_h": res.gpu_hours, "error": res.error})
        r = rec.get("result") or {}
        val = (r.get("metrics") or {}).get(m)
        print(f"{tid}  {rec['status']:<8} {m} {val if val is None else round(val, 4)}  "
              f"{float(r.get('gpu_hours') or 0):.1f} GPU h" + (f"  {r.get('error')}" if r.get("error") else ""))
    return 0


def cmd_next(a) -> int:
    run = load_run(Path(a.run))
    inc_c, inc_cid = _need_init(run)
    seeds = run.rules.seeds
    probe = seeds[0]
    m = run.metric.primary
    todo: list[str] = []
    done, recs = _results(run)
    inc_missing = [s for s in seeds if s not in done.get(inc_cid, {})]
    if inc_missing:
        waiting = [s for s in inc_missing if s in recs.get(inc_cid, {}) and recs[inc_cid][s]["status"] == "running"]
        todo.append(f"incumbent seeds {inc_missing} not finished" + (f" (running: {waiting})" if waiting else "")
                    + "; candidates are compared once they are")
    else:
        if not any(e["candidate"] == inc_cid for e in archive.events(run, "curve")):
            mean = sum(done[inc_cid][s].metrics[m] for s in seeds) / len(seeds)
            archive.append(run, {"event": "curve", "candidate": inc_cid, "spent": spent(run),
                                 "incumbent_mean": mean, "why": "incumbent complete"})
        verdicts = {e["candidate"] for e in archive.events(run, "verdict")}
        for e in archive.events(run, "propose"):
            cid = e["candidate"]
            inc_c, inc_cid = archive.incumbent(run)  # may change inside this loop
            if cid in verdicts or cid == inc_cid:
                continue
            c = archive._cand_from(e["candidate_dict"])
            res = done.get(cid, {})
            if probe not in res:
                st = recs.get(cid, {}).get(probe, {}).get("status")
                todo.append(f"`{cid}` {c.label()}: probe seed {probe} {st or 'not launched'}"
                            + (" -> relaunch it or drop the candidate" if st == "failed" else ""))
                continue
            pe = _probe_event(run, cid, inc_cid)
            if pe is None:
                v, g = after_probe(res[probe], done[inc_cid][probe], run.rules, run.metric)
                pe = {"event": "probe", "candidate": cid, "against": inc_cid, "verdict": v, "gain": g}
                archive.append(run, pe)
            if pe["verdict"] == "reject":
                detail = f"probe seed {probe} gain {pe['gain']:+.4f} <= {run.rules.probe_margin}"
                archive.append(run, {"event": "verdict", "candidate": cid, "against": inc_cid, "verdict": "rejected",
                                     "stage": "probe", "detail": detail})
                todo.append(f"`{cid}` {c.label()}: rejected ({detail})")
                continue
            missing = [s for s in seeds if s not in res]
            if missing:
                running = [s for s in missing if recs.get(cid, {}).get(s, {}).get("status") == "running"]
                launch = [s for s in missing if s not in running]
                todo.append(f"`{cid}` {c.label()}: probe passed ({pe['gain']:+.4f}); "
                            + (f"launch seeds {launch}: python -m trsi launch {a.run} <its proposal> --seed S --gpus <ids>"
                               if launch else f"waiting for seeds {running}"))
                continue
            ok, detail = decide(res, done[inc_cid], run.rules, run.metric)
            archive.append(run, {"event": "verdict", "candidate": cid, "against": inc_cid,
                                 "verdict": "accepted" if ok else "rejected", "stage": "decide", "detail": detail})
            todo.append(f"`{cid}` {c.label()}: {'ACCEPTED' if ok else 'rejected'} ({detail})")
            if ok:
                archive.set_incumbent(run, c, cid, detail)
                archive.append(run, {"event": "curve", "candidate": cid, "spent": spent(run),
                                     "incumbent_mean": sum(res[s].metrics[m] for s in seeds) / len(seeds),
                                     "why": f"accepted {cid}"})
    sp = spent(run)
    left = run.budget_gpu_h - sp
    ests = [x for x in (estimate(run, Candidate(b, r)) for b in run.space.bases for r in run.space.recipes) if x]
    inc_c, inc_cid = archive.incumbent(run)
    print(f"incumbent `{inc_cid}` {inc_c.label()}")
    print(f"budget    {sp:.1f} / {run.budget_gpu_h:.0f} GPU hours spent, {left:.1f} left")
    for t in todo:
        print(f"- {t}")
    open_c = [e for e in archive.events(run, "propose")
              if e["candidate"] not in {v["candidate"] for v in archive.events(run, "verdict")}]
    if ests and left < min(ests):
        print("STOP: the budget cannot pay for another seed. Write the review and report (python -m trsi report).")
    elif not inc_missing and not open_c:
        print("no open candidate: diagnose the incumbent and write 1-2 proposals (SKILL.md, step 4)")
    return 0


def cmd_report(a) -> int:
    run = load_run(Path(a.run))
    _need_init(run)
    print(report.write(run, cost_of(run), spent(run)))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m trsi", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"trsi {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("check")
    p.add_argument("run")
    p.add_argument("--ojh", action="store_true", help="also run `ojh check` on the start (no cards used)")
    p.set_defaults(fn=cmd_check)
    for name, fn in (("init", cmd_init), ("collect", cmd_collect), ("next", cmd_next), ("report", cmd_report)):
        p = sub.add_parser(name)
        p.add_argument("run")
        p.set_defaults(fn=fn)
    p = sub.add_parser("launch")
    p.add_argument("run")
    p.add_argument("proposal", nargs="?")
    p.add_argument("--start", action="store_true")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--gpus", required=True, help="card ids, e.g. 0 or 2,3")
    p.add_argument("--force", action="store_true", help="allow a dead end, or seeds after a probe that did not pass")
    p.set_defaults(fn=cmd_launch)
    p = sub.add_parser("adopt")
    p.add_argument("run")
    p.add_argument("proposal", nargs="?")
    p.add_argument("--start", action="store_true")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--run-dir", required=True, help="a finished harness run directory (has run.lock.yaml and DONE)")
    p.set_defaults(fn=cmd_adopt)
    a = ap.parse_args(argv)
    try:
        return a.fn(a)
    except (Refused, ConfigError, CandidateError, HarnessError) as e:
        print(f"trsi: {type(e).__name__.lower()}: {e}", file=sys.stderr)
        return 2
