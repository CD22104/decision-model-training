---
name: training-rsi
description: Use when the question is HOW to train a OneJev decision model within a GPU budget — which base (joey-1, elephant-1, mammoth-1), full fine-tune or LoRA, which hyperparameters (learning rate, epochs, Brier weight, batch, LoRA rank, trained modules) — or when a training-scheme search is running and needs a diagnosis, a candidate, a comparison, an accept/stop decision or a review. Training itself goes through onejev-harness (ojh). 训练方案选择：选基座、全参或 LoRA、超参，在卡时预算内比较新旧训练方案、决定接不接受、复盘。
---

# Training RSI（Skill 1.0，`trsi` 1.0.0）

版本和每版的依据见 `VERSIONS.md`。

## 概览

训练由 onejev-harness 完成（`ojh run`）。这个 skill 决定**怎么训练**：用哪个基座、全参还是 LoRA（配方）、哪些超参（带原因的 override），并在卡时预算内比较新旧方案。

```
环境 exp.yaml（数据、字段、评测器；整个 run 不变）+ 候选（基座 + 配方 + overrides）
  ──► trsi launch   生成 harness 工作区，ojh run 训练一个种子，立即返回
  ──► trsi collect  读 harness 的逐行预测：验证集 AUPRC、ECE、Brier，以及卡时
  ──► trsi next     先比种子 0；高过门槛才补种子；所有种子平均也高过门槛、校准不变差才接受
```

`trsi` 做机械的事：检查范围、预算和预测，生成工作区，启动，读结果，按规则判，记账，写报告。agent 做判断：读训练曲线和数据、诊断、提候选、写预测、复盘。

**不做：** 不改 harness 和 OneJev；不做评测这一层（阈值、校准、决策策略、test 上的评测），那是评测方向的 skill 的事；不读 test。

## 前提

- harness 能用：按 `training-with-onejev` skill 装好，`ojh doctor` 通过，用到的基座已经 `ojh fetch-base`。
- 一份环境 exp.yaml：数据、字段、评测器，`overrides: {}`。没有就先用 `adapting-data-for-onejev` skill 写。
- 本目录 `pip install -e .`（只依赖 pyyaml，CPU）。
- 运行配置 run.yaml：从 `configs/run.example.yaml` 复制，每个键都有注释。完整例子：`experiments/elliptic_txhist/run.yaml`。
- 命令写 `python -m trsi <命令> <run.yaml>`。run.yaml 里的 `$VAR` 要在当前 shell 里有值（先 source 环境变量文件）。

## 主流程

1. **定范围，一次问完用户**：
   - 起点：一般是团队现成的配方（基座 + 配方文件）；
   - 允许的基座、配方，每个 override 键允许的取值（常用键和先验见 `reference/search_space.md`）；
   - 能用的卡号、预算（卡时）、主指标（默认 AUPRC）。
   写进 run.yaml，**把 `space` 给用户看并得到批准**。批准的意义：范围内的候选不再逐个问（等于对 harness 的 override 审批做了一次性授权）；范围外的候选 `trsi launch` 会拒绝，要扩大范围就再问用户。
2. **检查**：`python -m trsi check run.yaml --ojh`，不占卡。
3. **起点**：`python -m trsi init run.yaml`，然后把起点跑满 `rules.seeds`（不算进预算）：
   `python -m trsi launch run.yaml --start --seed 0 --gpus 0`。已经有同一方案、同一 exp.yaml 的 harness run，用 `python -m trsi adopt run.yaml --start --seed 0 --run-dir <RUN>` 接过来，不重跑。
4. **诊断**：用 3–5 句话写当前方案的问题，每条带证据。
   - 训练曲线（每个种子的 `RESULT.md`；`report.md` 里有选中步数）：最好的步很早、后面一路变差 → 学习率偏大或过拟合；到最后还在变好 → 训练不够。
   - 种子之间差多少：差得大时，小改进分不出来，先试大改动。
   - 数据事实：正例比例、输入长度和标签的关系、重复行。它们会让曲线出现和"训练好坏"无关的现象。
   - 对照 `reference/lessons.md`（别的任务上已经学到的）。
5. **提候选**：在 `<out_dir>/<run_name>/proposals/` 写 1–2 个 YAML（格式见下），一次只改一处想法。
   - `note`：针对哪条诊断、为什么可能有效。
   - `prediction`：主指标在种子 0 上的预期值，以及被接受的概率。复盘时和实际对照。
   - 先验高的先试：结构类（换基座、全参或 LoRA、训练哪些层）优先于拟合类微调；换大模型时连学习率一起调小；从别处借来的改法，先写下它在原处为什么有效、这里满不满足；证据差不多时选便宜的。
6. **跑探针**：`python -m trsi launch run.yaml <proposal.yaml> --seed 0 --gpus <ids>`。立即返回，**交回，不轮询**。过一段时间再 `python -m trsi collect run.yaml`、`python -m trsi next run.yaml`。
7. **照 `next` 的输出走**：
   - 探针没过门槛：记为不接受；
   - 过了：按提示补种子（同一个 proposal，`--seed 1`、`--seed 2`）；
   - 种子都完成：`next` 判接受或不接受，接受时自动更新 `incumbent.yaml`。
   然后回到第 4 步。
8. **停**：`next` 打印 STOP（预算不够再跑一个种子）；或用户叫停；或诊断里剩下的都是数据或评测的问题（写下来交给对应方向）。
9. **复盘和交接**：`python -m trsi report run.yaml` 生成 `report.md`。逐条写"诊断 → 改动 → 预测 → 结果"，以及预测准不准。把当前方案的 run 目录和 checkpoint（`report.md` 第一节）交给评测方向。

候选 YAML：

```yaml
base: mammoth-1
recipe: recipes/mammoth_lora_1g.yaml      # 相对 run.yaml；必须在 space.recipes 里
overrides:
  onejev.lr: {value: 5.0e-5, reason: "27B 沿用 9B 的学习率偏高：验证分数在第 1200 步后一路下降"}
note: 换大模型时学习率跟着调小
prediction: {auprc: 0.91, p_accept: 0.3}
```

## 判输赢（细节和依据见 `reference/protocol.md`）

- 探针：候选的种子 0 比当前方案的种子 0 高过 `probe_margin`（默认 0.035），才补其余种子；否则不接受。
- 接受：所有种子平均比当前方案高过 `accept_margin`（默认 0.035），且 ECE、Brier 的平均变差不超过护栏（默认 0.02、0.005）。
- 为什么不用"一个种子 + 按行 bootstrap"：在 Elliptic 上，同一配方换种子的标准差是 0.0175 AUPRC，种子之间约九成的差来自训练本身，按行重抽看不到这部分。
- 拿不准就不接受：留着当前方案没有成本；靠噪声赢的方案会让后面每一步都建立在错的起点上。

## 规则

- 候选只经 `trsi launch` 训练，不直接 `ojh run`，否则记账、复用和比较都会漏掉它。
- 不改 harness、`run.lock.yaml`、环境 exp.yaml。要改 `rules`、`metric`、`budget`、`space`，就开一个新的 `run_name`，并告诉用户。
- 不手改 `archive.jsonl`、`incumbent.yaml`、`trials/`。
- 不读 test：不传 `--allow-test`、`--allow-report`。
- 卡：只用 run.yaml 里批准的卡。harness 说卡忙或放不下时，把原因和它列的替代方案原样告诉用户，由用户选；不自己换卡、换配方。
- 死路（`space` 里标了 `dead` 的取值）默认拒绝。只有能说明前提已经变了时才加 `--force`，并写进 note。
- 探针没过的候选不补种子（`--force` 只在用户要求时用）。
- 清理：`report.md` 列出不属于当前方案的 run。`ojh cleanup <RUN>` 先不带 `--confirm` 列出来，用户批准后才删。

## 长时间运行

不做后台循环，规则也不会自己开新候选。要连续跑，用定时检查（例如 Claude Code 的定时任务）每 1–2 小时执行一次 `collect` 和 `next`，按输出补种子；新候选只在 agent 做完第 4、5 步后提出。

## 故障

| 现象 | 处理 |
|---|---|
| `launch` 说 outside the approved space | 候选超出用户批准的范围。换候选，或者问用户是否扩大 `space` |
| `launch` 说 dead end | 先看死路的原因；前提没变就换候选 |
| `launch` 说 budget | 预算用完：停，写复盘 |
| `launch` 说 no cost estimate | 在 `budget.estimate` 写这个基座一个种子大约多少卡时 |
| `launch` 说 comes after the probe | 先 `collect`、`next`，探针过了才补种子 |
| `ojh run` 非零退出（卡忙、不可行、没有默认配方） | 原样告诉用户，由用户选替代方案 |
| `collect` 显示 failed | 看 run 目录的 `DONE`、`runner.log`，按 `training-with-onejev` 的故障表处理。卡时照记。修好后用同一条命令重跑 |
| H100、H200 上第 1 步反向就失败，日志要求装 tilelang（`DONE` 的 stage 是 train） | harness 的安装脚本在 Hopper 卡上漏了 tilelang 和 causal-conv1d。告诉用户，经同意后按 `reference/lessons.md` 的"环境上踩过的坑"补装，并在机器配置的 `env` 里设 `CUDA_HOME` |
| 训练已经完成，收尾被 signal 15 杀掉（`DONE` 的 stage 是 finalize） | 机器上的显卡看门狗杀了长时间不用卡的进程：exp.yaml 有 `fields.text` 时，收尾会在 CPU 上训文本基线，同时占着卡。告诉用户。去掉 `text` 要改环境 exp.yaml，等于换了环境：经用户同意后开新的 `run_name`，旧试验不会复用 |
| `next` 说 incumbent seeds not finished | 起点的种子还没跑完或没接过来，先补齐 |
| 两个 run.yaml 的结果互相"复用" | 复用要求基座、配方内容、overrides、种子和环境 exp.yaml 内容都相同，这是预期行为；不想复用就换 `out_dir` |
