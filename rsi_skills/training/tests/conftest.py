import json
import os
import stat
import sys
from pathlib import Path

import pytest
import yaml

from trsi.cli import main

HERE = Path(__file__).parent

RUN = {
    "run_name": "r1",
    "out_dir": "out",
    "exp": "exp.yaml",
    "ojh": {"bin": "ojh"},
    "start": {"base": "elephant-1", "recipe": "recipes/el_lora_1g.yaml", "note": "team recipe"},
    "space": {
        "bases": ["elephant-1", "mammoth-1"],
        "recipes": ["recipes/el_lora_1g.yaml", "recipes/mm_lora_1g.yaml"],
        "overrides": {
            "onejev.lr": {"values": [5.0e-5, 1.0e-4, 2.0e-4]},
            "onejev.brier_weight": {"values": [0, 1, 3], "dead": {3: "Elliptic: 3 seeds mean -0.082"}},
            "lora.r": {"range": [8, 128]},
        },
    },
    "gpus": {"ids": [0, 1, 2, 3]},
    "metric": {"primary": "auprc", "guards": {"ece": 0.02, "brier": 0.005}},
    "rules": {"seeds": [0, 1, 2], "probe_margin": 0.035, "accept_margin": 0.035},
    "budget": {"gpu_hours": 100, "estimate": {"elephant-1": 8, "mammoth-1": 23}},
}

EXP = {
    "name": "elliptic",
    "data": {"adapter": "onejev_rows", "train": "data/train.jsonl",
             "eval": {"val": {"train_labels": "data/val.jsonl"}}},
    "fields": {"label": "target.yes", "group": "meta.state_id"},
    "evaluator": {"type": "binary_decision", "gate": True},
    "seeds": [42],
    "overrides": {},
}

RECIPE = {"mode": "lora", "gpus": 1, "onejev": {"lr": 1.0e-4, "epochs": 1, "brier_weight": 1},
          "lora": {"r": 32, "alpha": 64, "dropout": 0.05}}


class Env:
    def __init__(self, root: Path, capsys):
        self.root = root
        self.capsys = capsys
        self.run = root / "run.yaml"
        self.calls = root / "calls.txt"

    def write_run(self, **changes):
        d = json.loads(json.dumps(RUN))
        d["ojh"]["bin"] = str(self.root / "ojh")
        for k, v in changes.items():
            d[k] = v
        self.run.write_text(yaml.safe_dump(d))

    def trsi(self, *args) -> tuple[int, str, str]:
        self.capsys.readouterr()
        code = main([str(a) for a in args])
        out = self.capsys.readouterr()
        return code, out.out, out.err

    def proposal(self, name, base="elephant-1", recipe="recipes/el_lora_1g.yaml", overrides=None,
                 note="a reason", prediction=None):
        d = {"base": base, "recipe": recipe, "note": note,
             "overrides": {k: {"value": v, "reason": f"try {k}"} for k, v in (overrides or {}).items()},
             "prediction": prediction if prediction is not None else {"auprc": 0.9, "p_accept": 0.3}}
        p = self.root / "out" / "r1" / "proposals" / f"{name}.yaml"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(yaml.safe_dump(d))
        return p

    def n_calls(self) -> int:
        return len(self.calls.read_text().splitlines()) if self.calls.is_file() else 0


@pytest.fixture
def env(tmp_path, monkeypatch, capsys):
    (tmp_path / "recipes").mkdir()
    (tmp_path / "recipes" / "el_lora_1g.yaml").write_text(yaml.safe_dump(RECIPE))
    mm = json.loads(json.dumps(RECIPE))
    mm["onejev"]["lr"] = 5.0e-5
    (tmp_path / "recipes" / "mm_lora_1g.yaml").write_text(yaml.safe_dump(mm))
    (tmp_path / "exp.yaml").write_text(yaml.safe_dump(EXP))
    ojh = tmp_path / "ojh"
    ojh.write_text(f"#!/bin/sh\nexec {sys.executable} {HERE / 'fake_ojh.py'} \"$@\"\n")
    ojh.chmod(ojh.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("FAKE_OJH_OUT", str(tmp_path / "ojh_runs"))
    monkeypatch.setenv("FAKE_OJH_CALLS", str(tmp_path / "calls.txt"))
    monkeypatch.delenv("FAKE_OJH_PENDING", raising=False)
    monkeypatch.setenv("FAKE_OJH_QUALITY", "{}")
    e = Env(tmp_path, capsys)
    e.write_run()
    return e
