# Elliptic++ 地址任务，交易历史输入

判断一个比特币地址是不是非法地址。输入是这个地址的交易历史（写成文本），输出是"非法"的概率。按时间切分：第 1–29 步训练、30–34 步验证、35–49 步测试（团队统一规则）。

## 文件

- `exp.yaml`：环境（数据、字段、评测器）。数据由 `pipelines/elliptic_elephant/01_build_data.sh` 生成到 `$DMRSI_DATA`。评测集用完整验证集（24,225 个地址），理由写在文件开头。
- `run.yaml`：起点是团队配方（elephant-1 LoRA 1 卡）；范围包括两个基座和 7 个超参键；v0 已经证明不行的取值标成了死路。`gpus.ids` 和批准日期要在真跑前填。

## 怎么跑

```bash
source $DMRSI_HOME/dmrsi-env.sh          # 设 DMRSI_HOME、DMRSI_DATA、OJH_REPO
cd rsi_skills/training
python -m trsi check experiments/elliptic_txhist/run.yaml --ojh
python -m trsi init  experiments/elliptic_txhist/run.yaml
python -m trsi launch experiments/elliptic_txhist/run.yaml --start --seed 0 --gpus 0   # 种子 1、2 同理
# 之后照 SKILL.md：诊断 → proposals → launch 探针 → collect → next
```

## 之前的结果（直接用 OneJev，没经过 harness）

见 `reference/lessons.md`：团队配方 3 个种子平均 0.902；试过的 7 个改动都没过接受线，最接近的是 mammoth-1 + 学习率减半（种子 0 +0.024）。经 harness 重跑后，checkpoint 改为按 NLL 选，数会有出入。
