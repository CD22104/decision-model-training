"""End to end with the fake ojh (design tests 3, 7, 9, 10 and the whole probe -> seeds -> accept path)."""
import json

import yaml

from trsi import archive, harness
from trsi.config import load_run


def _start(env):
    assert env.trsi("init", env.run)[0] == 0
    for s in (0, 1, 2):
        code, out, err = env.trsi("launch", env.run, "--start", "--seed", s, "--gpus", 0)
        assert code == 0, err
    assert env.trsi("collect", env.run)[0] == 0


def test_probe_more_seeds_then_accept(env, monkeypatch):
    monkeypatch.setenv("FAKE_OJH_QUALITY", json.dumps({"elephant-1": 0.8, "mammoth-1|onejev.lr=5e-05": 2.5}))
    _start(env)
    p = env.proposal("mm1-lr-half", base="mammoth-1", recipe="recipes/mm_lora_1g.yaml", overrides={"onejev.lr": 5e-5},
                     prediction={"auprc": 0.91, "p_accept": 0.3})
    code, _, err = env.trsi("launch", env.run, p, "--seed", 1, "--gpus", 0)   # seed 1 before the probe
    assert code == 2 and "comes after the probe" in err
    assert env.trsi("launch", env.run, p, "--seed", 0, "--gpus", 1)[0] == 0
    env.trsi("collect", env.run)
    code, out, _ = env.trsi("next", env.run)
    assert "probe passed" in out and "launch seeds [1, 2]" in out
    for s in (1, 2):
        assert env.trsi("launch", env.run, p, "--seed", s, "--gpus", 2)[0] == 0
    env.trsi("collect", env.run)
    code, out, _ = env.trsi("next", env.run)
    assert "ACCEPTED" in out
    run = load_run(env.run)
    inc, cid = archive.incumbent(run)
    assert inc.base == "mammoth-1" and inc.overrides["onejev.lr"].value == 5e-5
    assert [e["event"] for e in archive.events(run)].count("incumbent") == 2
    assert env.n_calls() == 6
    rep = env.root / "out" / "r1" / "report.md"
    assert env.trsi("report", env.run)[0] == 0 and cid in rep.read_text() and "checkpoints/step-800" in rep.read_text()


def test_probe_reject(env, monkeypatch):
    monkeypatch.setenv("FAKE_OJH_QUALITY", json.dumps({"elephant-1": 1.5, "elephant-1|onejev.lr=0.0002": 1.2}))
    _start(env)
    p = env.proposal("lr-double", overrides={"onejev.lr": 2e-4})   # weaker than the start
    env.trsi("launch", env.run, p, "--seed", 0, "--gpus", 0)
    env.trsi("collect", env.run)
    code, out, _ = env.trsi("next", env.run)
    assert "rejected (probe seed 0 gain" in out
    code, _, err = env.trsi("launch", env.run, p, "--seed", 1, "--gpus", 0)
    assert code == 2 and "comes after the probe" in err


def test_reuse_without_training(env):
    _start(env)
    p = env.proposal("lr-half", overrides={"onejev.lr": 5e-5})
    env.trsi("launch", env.run, p, "--seed", 0, "--gpus", 0)
    env.trsi("collect", env.run)
    n = env.n_calls()
    code, out, _ = env.trsi("launch", env.run, p, "--seed", 0, "--gpus", 0)
    assert code == 0 and "already part of this run" in out and env.n_calls() == n
    env.write_run(run_name="r2")                       # another run under the same out_dir
    env.trsi("init", env.run)
    code, out, _ = env.trsi("launch", env.run, "--start", "--seed", 0, "--gpus", 0)
    assert "reused" in out and env.n_calls() == n
    code, out, _ = env.trsi("launch", env.run, p, "--seed", 0, "--gpus", 0)
    assert "reused" in out and env.n_calls() == n
    run = load_run(env.run)
    assert archive.trials(run)  # the reused probe is charged to run r2
    from trsi.cli import spent
    assert spent(run) > 0


def test_budget(env):
    env.write_run(budget={"gpu_hours": 10, "estimate": {"elephant-1": 8, "mammoth-1": 23}})
    _start(env)                                         # the start is not charged
    run = load_run(env.run)
    from trsi.cli import spent
    assert spent(run) == 0
    code, _, err = env.trsi("launch", env.run, env.proposal("mm", base="mammoth-1", recipe="recipes/mm_lora_1g.yaml"),
                            "--seed", 0, "--gpus", 0)
    assert code == 2 and "budget" in err
    code, _, err = env.trsi("launch", env.run, env.proposal("lr", overrides={"onejev.lr": 5e-5}), "--seed", 0, "--gpus", 0)
    assert code == 0, err                               # 2 GPU h measured < 10


def test_failed_run_is_charged_and_not_compared(env, monkeypatch):
    _start(env)
    monkeypatch.setenv("FAKE_OJH_QUALITY", json.dumps({"elephant-1|onejev.lr=0.0002": "fail"}))
    p = env.proposal("lr-double", overrides={"onejev.lr": 2e-4})
    env.trsi("launch", env.run, p, "--seed", 0, "--gpus", 0)
    code, out, _ = env.trsi("collect", env.run)
    assert "failed" in out and "CUDA out of memory" in out
    from trsi.cli import spent
    assert spent(load_run(env.run)) == 1.0               # 3600 s x 1 card
    code, out, _ = env.trsi("next", env.run)
    assert "probe seed 0 failed" in out and "ACCEPTED" not in out and "rejected" not in out
    monkeypatch.setenv("FAKE_OJH_QUALITY", "{}")
    assert env.trsi("launch", env.run, p, "--seed", 0, "--gpus", 0)[0] == 0   # relaunch; the failed hour stays charged
    env.trsi("collect", env.run)
    assert spent(load_run(env.run)) == 3.0


def test_pending_run_and_read_result(env, monkeypatch):
    env.trsi("init", env.run)
    monkeypatch.setenv("FAKE_OJH_PENDING", "1")
    env.trsi("launch", env.run, "--start", "--seed", 0, "--gpus", 0)
    code, out, _ = env.trsi("collect", env.run)
    assert "running" in out
    rd = next(iter(archive.trials(load_run(env.run)).values()))["run_dir"]
    assert harness.state(rd) == "running"
    monkeypatch.delenv("FAKE_OJH_PENDING")
    env.write_run(run_name="r3")
    env.trsi("init", env.run)
    env.trsi("launch", env.run, "--start", "--seed", 1, "--gpus", 0)
    rd = next(iter(archive.trials(load_run(env.run)).values()))["run_dir"]
    r = harness.read_result(rd, "yes", 1)
    assert r.status == "done" and r.metrics["n"] == 300 and r.metrics["positives"] == 30
    assert r.gpu_hours == 2.0 and r.gpu_names == ["NVIDIA H200"] and r.selected_step == 800 and r.last_step == 1200


def test_gpu_factor(env):
    env.write_run(budget={"gpu_hours": 100, "estimate": {"elephant-1": 8}, "gpu_factor": {"H200": 1.7}})
    _start(env)
    env.trsi("launch", env.run, env.proposal("lr", overrides={"onejev.lr": 5e-5}), "--seed", 0, "--gpus", 0)
    env.trsi("collect", env.run)
    from trsi.cli import spent
    assert abs(spent(load_run(env.run)) - 3.4) < 1e-9


def test_adopt(env):
    _start(env)
    env.write_run(run_name="r4", out_dir="out2")
    env.trsi("init", env.run)
    run1_dir = next(r for r in (env.root / "ojh_runs").glob("*-s0/r0"))
    code, out, err = env.trsi("adopt", env.run, "--start", "--seed", 0, "--run-dir", run1_dir)
    assert code == 0, err
    p = env.proposal("other", overrides={"onejev.lr": 5e-5})
    code, _, err = env.trsi("adopt", env.run, p, "--seed", 0, "--run-dir", run1_dir)
    assert code == 2 and "not this scheme" in err


def test_check(env):
    code, out, _ = env.trsi("check", env.run, "--ojh")
    assert code == 0 and "ojh check" in out and "mode lora" in out


def test_env_file_sets_the_ojh_path(env, monkeypatch):
    monkeypatch.delenv("TRSI_TEST_OJH_DIR", raising=False)
    (env.root / "env.sh").write_text(f"export TRSI_TEST_OJH_DIR={env.root}\n")
    env.write_run(ojh={"bin": "$TRSI_TEST_OJH_DIR/ojh", "env_file": "env.sh"})
    env.trsi("init", env.run)
    code, out, err = env.trsi("launch", env.run, "--start", "--seed", 0, "--gpus", 0)
    assert code == 0, err
    assert env.n_calls() == 1
