# Training RSI Skill（训练方向）

在卡时预算内，自动选择一个 OneJev 决策模型**怎么训练**：用哪个基座、全参还是 LoRA、哪些超参。训练一律通过 onejev-harness（`ojh run`），不改 harness。评测（阈值、校准、决策策略、test）归评测方向的 skill；这里只交出选中的训练方案和它的 checkpoint。

给 agent 的流程在 [`SKILL.md`](SKILL.md)。版本和依据在 [`VERSIONS.md`](VERSIONS.md)。

## 一个例子

一个候选 = 基座 + 配方 + 带原因的 overrides + 预测：

```yaml
base: mammoth-1
recipe: ../../configs/recipes/mammoth_lora_1g.yaml
overrides:
  onejev.lr: {value: 5.0e-5, reason: "27B 沿用 9B 的学习率偏高：验证分数在第 1200 步后一路下降"}
note: 换大模型时学习率跟着调小
prediction: {auprc: 0.91, p_accept: 0.3}
```

```bash
python -m trsi launch run.yaml proposals/mm1-lr-half.yaml --seed 0 --gpus 3   # 检查范围、预算、预测，然后 ojh run
python -m trsi collect run.yaml   # 训练结束后：验证 AUPRC 0.907、ECE、Brier、卡时
python -m trsi next run.yaml      # 和当前方案的种子 0（0.883）比：+0.024，没过 0.035，不补种子
```

## 安装和测试

```bash
cd rsi_skills/training && pip install -e '.[dev]' && pytest -q     # 只需要 CPU；测试用一个假的 ojh
```

## 文件

| 位置 | 内容 |
|---|---|
| `SKILL.md` | agent 的流程、规则、故障表 |
| `trsi/` | 命令行工具：`check`、`init`、`launch`、`adopt`、`collect`、`next`、`report` |
| `configs/run.example.yaml` | 运行配置模板，每个键都有注释 |
| `configs/recipes/` | 起点配方（harness 格式）：elephant-1、mammoth-1 的 1 卡 LoRA |
| `reference/search_space.md` | 能改什么、属于哪一类、先验和代价 |
| `reference/protocol.md` | 怎么比较训练方案、为什么这样 |
| `reference/lessons.md` | Elliptic 上已经学到的：结果、死路、成本、环境的坑 |
| `experiments/elliptic_txhist/` | Elliptic++ 交易历史输入的环境和运行配置 |
| `tests/` | 单元测试和端到端测试（假的 `ojh`） |

## 记录（都在 `<out_dir>/` 下，不进 git）

| 文件 | 内容 |
|---|---|
| `<run_name>/archive.jsonl` | 每次提出、启动、复用、结果、判定、换方案各一行，只追加 |
| `<run_name>/incumbent.yaml` | 当前方案，只有 `trsi next` 判接受时才改 |
| `<run_name>/proposals/` | agent 写的候选 |
| `<run_name>/work/<试验编号>/exp.yaml` | 给 harness 的工作区 |
| `<run_name>/report.md` | 当前方案和 checkpoint、每个候选的预测和结果、预算、最好方案随卡时 |
| `trials/<试验编号>.json` | 每个训练过的（方案, 种子），同一 `out_dir` 下的 run 共用，做过的不重做 |

## 和别的部分的接口

- 输入：一份 harness 的 exp.yaml（数据方向或 `adapting-data-for-onejev` skill 产出），harness 的基座和配方。
- 输出：`report.md` 第一节，当前方案每个种子的 harness run 目录和选中的 checkpoint。评测方向用自己的打分脚本接着做（Elliptic 上是 `pipelines/elliptic_elephant/03_score_checkpoint.sh`）。
