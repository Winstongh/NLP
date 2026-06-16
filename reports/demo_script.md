# 1-2 分钟程序演示视频脚本

## 0:00-0:15 项目简介

展示项目目录，说明本项目实现了一个 PyTorch 自实现 Transformer 英译中系统，包含数据准备、训练、评估和推理。

建议画面：

```bash
tree -L 3
```

## 0:15-0:35 数据准备

说明使用 IWSLT 2017 `iwslt2017-en-zh`，英文为源语言，中文为目标语言。展示数据准备命令和生成的 JSONL、词表文件。

建议画面：

```bash
python scripts/prepare_data.py --output-dir data/iwslt2017 --max-train-samples 20000
ls data/iwslt2017
head -n 2 data/iwslt2017/train.jsonl
```

## 0:35-0:55 模型训练

展示训练命令和日志，说明 loss 下降、保存 checkpoint。

建议画面：

```bash
python scripts/train.py --config configs/rtx4090.yaml --device cuda
tail -n 3 checkpoints/iwslt2017_en_zh/train_log.jsonl
```

如果正式训练时间太长，视频中可以展示已经训练好的日志和 checkpoint。

## 0:55-1:15 模型评估

展示测试集评估命令，说明输出包含 loss、token accuracy、BLEU 和翻译样例。

建议画面：

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/iwslt2017_en_zh/best.pt \
  --config checkpoints/iwslt2017_en_zh/config.yaml \
  --split test \
  --device cuda
```

## 1:15-1:45 交互式翻译

展示输入英文句子并输出中文翻译。

建议画面：

```bash
python scripts/translate.py \
  --checkpoint checkpoints/iwslt2017_en_zh/best.pt \
  --config checkpoints/iwslt2017_en_zh/config.yaml \
  --device cuda
```

示例输入：

```text
This talk is about machine translation.
We use a transformer model.
```

## 1:45-2:00 总结

总结项目完成了：

- Transformer 模型自实现。
- IWSLT 数据准备和词表构建。
- 训练、评估和推理完整流程。
- 可复现实验报告和演示视频材料。
