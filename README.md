# 基于 Transformer 的英译中机器翻译（cmn-eng-simple）

用 PyTorch **从零自实现** Transformer Encoder-Decoder，完成英文→中文翻译。
输入：一句英文（如 `tom is a student .`）；输出：中文（`汤姆是个学生。`）。

- 数据集：`cmn-eng-simple`（train 18000 / valid 500 / test 2636）
- 评测指标：**BLEU**（词级）
- **测试集结果：词级 BLEU 25.72 / token 准确率 62.4%**

## 目录结构

```text
.
├── configs/cmn.yaml          # 训练/模型/解码配置
├── data/
│   ├── cmn-eng-simple/       # 原始数据集（已分词，制表符分隔）
│   └── cmn/                  # prepare_cmn.py 生成的 jsonl + 词表
├── scripts/                  # 命令行入口
│   ├── prepare_cmn.py        # 数据 → jsonl + 词表
│   ├── train.py / evaluate.py / translate.py
│   └── average_checkpoints.py
├── src/nmt/                  # 自实现源码：model/data/vocab/tokenizer/train/evaluate/inference/...
└── reports/
    ├── report.md             # 实验报告（方法/结果/分析）
    ├── cmn_training_log.txt   # 逐轮训练日志
    └── demo_script.md         # 1–2 分钟演示视频脚本
```

## 环境

```bash
conda create -n nmt python=3.10 -y
conda activate nmt
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"   # 期望 True
```

## 运行流程

```bash
# 1) 数据准备（cmn-eng-simple 已放在 data/ 下）
python scripts/prepare_cmn.py
#    train=18000 valid=500 test=2636 ; en vocab=4404 zh vocab=9886

# 2) 训练（pre-norm + 权重绑定的小模型，~4 分钟）
python scripts/train.py --config configs/cmn.yaml --device cuda

# 3) 权重平均（取后段 10 轮）
python scripts/average_checkpoints.py \
  --checkpoints checkpoints/cmn/epoch_{51,52,53,54,55,56,57,58,59,60}.pt \
  --output checkpoints/cmn/avg.pt

# 4) 评估（词级 BLEU，2636 句测试集）
python scripts/evaluate.py --checkpoint checkpoints/cmn/avg.pt \
  --config checkpoints/cmn/config.yaml --split test --device cuda --beam-size 5 --bleu-samples 3000

# 5) 翻译
python scripts/translate.py --checkpoint checkpoints/cmn/avg.pt \
  --config checkpoints/cmn/config.yaml --beam-size 5 --device cuda --text "tom is a student ."
# → 汤姆是个学生。
```

## 实现要点

- **自实现 Transformer**：多头注意力、正弦位置编码、Encoder/Decoder Layer、前馈网络、残差 + LayerNorm（pre-norm）、线性生成器、权重绑定。
- **训练**：teacher forcing + label smoothing 交叉熵；AdamW + Noam warmup；bf16；梯度裁剪。
- **解码**：自实现 beam search（长度归一化）+ checkpoint averaging。
- 数据已分词（英文 BPE、中文 jieba 词），按空格切分（`pretokenized`），**BLEU 以词为单位**统计 n 元词匹配。

## 提交内容

| 内容 | 位置 |
| --- | --- |
| 可运行源代码 | `src/nmt/`、`scripts/`、`configs/cmn.yaml` |
| 实验报告（方法/结果/分析） | [`reports/report.md`](reports/report.md) |
| 训练日志 | [`reports/cmn_training_log.txt`](reports/cmn_training_log.txt) |
| 演示视频脚本 | [`reports/demo_script.md`](reports/demo_script.md) |
