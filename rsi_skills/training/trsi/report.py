"""report.md: the incumbent and its checkpoints (the hand-off to the evaluation side), every candidate's prediction
against its result, the budget, the best metric against GPU hours, and runs that are not part of the incumbent."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from . import __version__, archive
from .config import RunConfig
from .harness import TrialResult, exp_sha256


def _f(v, nd=4) -> str:
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def render(run: RunConfig, cost_of, spent: float) -> str:
    m = run.metric.primary
    inc = archive.incumbent(run)
    T = archive.trials(run)
    by_cid: dict[str, dict[int, dict]] = defaultdict(dict)
    for rec in T.values():
        by_cid[rec["candidate_id"]][rec["seed"]] = rec
    proposals = {e["candidate"]: e for e in archive.events(run, "propose")}
    verdicts = {e["candidate"]: e for e in archive.events(run, "verdict")}
    probes = defaultdict(list)
    for e in archive.events(run, "probe"):
        probes[e["candidate"]].append(e)

    L = [f"# Training RSI report: {run.run_name}", "",
         f"- environment: `{run.exp}` (sha256 {exp_sha256(run)[:12]}); trsi {__version__}",
         f"- rules: probe seed {run.rules.seeds[0]}, probe margin {run.rules.probe_margin}, seeds "
         f"{list(run.rules.seeds)}, accept margin {run.rules.accept_margin}, guards {run.metric.guards}",
         f"- budget: {spent:.1f} / {run.budget_gpu_h:.0f} GPU hours spent (the start's trials are not charged)", ""]

    L += ["## Incumbent (hand-off to the evaluation side)", ""]
    if inc:
        c, cid = inc
        L.append(f"`{cid}` {c.label()}" + (f": {c.note}" if c.note else ""))
        L += ["", "| seed | " + m + " | ece | brier | selected step | checkpoint | run dir |", "|---|---|---|---|---|---|---|"]
        for s, rec in sorted(by_cid.get(cid, {}).items()):
            r = TrialResult.from_dict(rec["result"]) if rec.get("result") else None
            L.append(f"| {s} | {_f(r and r.metrics.get(m))} | {_f(r and r.metrics.get('ece'))} | "
                     f"{_f(r and r.metrics.get('brier'))} | {_f(r and r.selected_step)} | "
                     f"`{r.checkpoint if r else '-'}` | `{rec['run_dir']}` |")
        L.append("")

    L += ["## Candidates", "",
          f"| candidate | scheme | note | predicted {m} / p_accept | probe gain | seeds ({m}) | verdict | GPU h |",
          "|---|---|---|---|---|---|---|---|"]
    for cid, e in proposals.items():
        pr = e.get("prediction") or {}
        seeds = by_cid.get(cid, {})
        res = ", ".join(f"s{s} {_f((rec.get('result') or {}).get('metrics', {}).get(m))}"
                        if rec["status"] == "done" else f"s{s} {rec['status']}" for s, rec in sorted(seeds.items()))
        pg = probes.get(cid)
        v = verdicts.get(cid)
        cost = sum(cost_of(rec) for rec in seeds.values())
        label = archive._cand_from(e["candidate_dict"]).label()
        L.append(f"| `{cid}` | {label} | {e['candidate_dict'].get('note', '')} | {_f(pr.get('value'))} / "
                 f"{_f(pr.get('p_accept'), 2)} | {_f(pg[-1]['gain']) if pg else '-'} | {res or '-'} | "
                 f"{(v['verdict'] + ': ' + v['detail']) if v else 'open'} | {cost:.1f} |")
    L.append("")

    L += ["## Best scheme against GPU hours", "", f"| GPU hours spent | incumbent mean {m} | event |", "|---|---|---|"]
    for e in archive.events(run, "curve"):
        L.append(f"| {e['spent']:.1f} | {_f(e.get('incumbent_mean'))} | {e['why']} |")
    L.append("")

    used = set(by_cid.get(inc[1], {}).keys()) if inc else set()
    other = [rec for rec in T.values() if not (inc and rec["candidate_id"] == inc[1] and rec["seed"] in used)]
    if other:
        L += ["## Runs outside the incumbent", "",
              "Their checkpoints can be cleaned up with `ojh cleanup <RUN>` (list first; `--confirm` only after the "
              "user approves).", ""]
        L += [f"- `{rec['run_dir']}` ({rec['candidate_id']} seed {rec['seed']}, {rec['status']})" for rec in other]
    return "\n".join(L) + "\n"


def write(run: RunConfig, cost_of, spent: float) -> Path:
    p = run.home / "report.md"
    archive._write(p, render(run, cost_of, spent))
    return p
