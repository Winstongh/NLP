# 基于 Transformer 的英译中机器翻译

这是一个用于 NLP 课程实践的 PyTorch 项目：从零实现 Transformer Encoder-Decoder，并完成英文到中文的机器翻译训练、评估和推理。

项目默认使用 IWSLT 2017 `iwslt2017-en-zh` 数据集。数据集主页：https://huggingface.co/datasets/IWSLT/iwslt2017

## 目录结构

```text
.
├── configs/                 # 训练配置
│   ├── debug.yaml           # 小样例 smoke test
│   └── rtx4090.yaml         # Ubuntu + RTX 4090 推荐配置
├── examples/sample_data/    # 极小样例数据，方便本地检查代码是否跑通
├── reports/                 # 报告模板和演示视频讲稿
├── scripts/                 # 命令行入口
└── src/nmt/                 # 数据、词表、模型、训练、评估、推理源码
```

## 环境安装

建议在 Ubuntu GPU 环境中运行训练。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

如果 `torch` 安装后不能识别 CUDA，请按 PyTorch 官网为你的 CUDA 版本生成安装命令，再重新安装 `torch`。

## 快速自检

项目内置了一个很小的样例数据集，用于检查代码、mask、checkpoint 和推理流程。

```bash
python scripts/train.py --config configs/debug.yaml --device cpu
python scripts/evaluate.py --checkpoint checkpoints/debug/best.pt --config checkpoints/debug/config.yaml --split test --device cpu
python scripts/translate.py --checkpoint checkpoints/debug/best.pt --config checkpoints/debug/config.yaml --text "hello world" --device cpu
```

这个样例数据太小，翻译效果没有实际意义，只用于确认程序可运行。

## 准备 IWSLT 数据

在 Ubuntu 训练机上运行：

```bash
python scripts/prepare_data.py \
  --dataset IWSLT/iwslt2017 \
  --config iwslt2017-en-zh \
  --output-dir data/iwslt2017 \
  --src-lang en \
  --tgt-lang zh \
  --src-vocab-size 32000 \
  --tgt-vocab-size 12000
```

如果第一次下载较慢，可以先用子集快速验证：

```bash
python scripts/prepare_data.py \
  --output-dir data/iwslt2017 \
  --max-train-samples 20000 \
  --max-valid-samples 1000 \
  --max-test-samples 1000
```

生成文件：

```text
data/iwslt2017/train.jsonl
data/iwslt2017/validation.jsonl
data/iwslt2017/test.jsonl
data/iwslt2017/vocab.en.json
data/iwslt2017/vocab.zh.json
```

每行格式为：

```json
{"src": "English sentence.", "tgt": "中文句子。"}
```

## 训练

RTX 4090 48GB 推荐命令：

```bash
python scripts/train.py --config configs/rtx4090.yaml --device cuda
```

继续训练：

```bash
python scripts/train.py \
  --config checkpoints/iwslt2017_en_zh/config.yaml \
  --resume checkpoints/iwslt2017_en_zh/last.pt \
  --device cuda
```

训练输出：

```text
checkpoints/iwslt2017_en_zh/best.pt
checkpoints/iwslt2017_en_zh/last.pt
checkpoints/iwslt2017_en_zh/train_log.jsonl
checkpoints/iwslt2017_en_zh/config.yaml
```

## 评估

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/iwslt2017_en_zh/best.pt \
  --config checkpoints/iwslt2017_en_zh/config.yaml \
  --split test \
  --device cuda \
  --output outputs/test_metrics.json
```

输出包含：

- `loss`：测试集交叉熵损失。
- `token_accuracy`：忽略 `<pad>` 后的 token 级准确率。
- `bleu`：基于中文字符 token 的 BLEU 分数。
- `samples`：若干英文、参考中文、模型输出对照样例。

## 推理演示

单句翻译：

```bash
python scripts/translate.py \
  --checkpoint checkpoints/iwslt2017_en_zh/best.pt \
  --config checkpoints/iwslt2017_en_zh/config.yaml \
  --text "This talk is about machine translation." \
  --device cuda
```

交互式翻译：

```bash
python scripts/translate.py \
  --checkpoint checkpoints/iwslt2017_en_zh/best.pt \
  --config checkpoints/iwslt2017_en_zh/config.yaml \
  --device cuda
```

## 实现要点

- 自实现 Transformer：多头注意力、位置编码、Encoder/Decoder Layer、前馈网络、残差连接和 LayerNorm。
- 训练使用 teacher forcing：目标序列右移作为 decoder 输入，下一个 token 作为监督信号。
- Mask 包含 source padding mask、target padding mask 和 decoder causal mask。
- 优化器使用 AdamW，学习率使用 Transformer 原论文的 warmup + inverse square-root decay。
- 中文默认字符级建模，避免额外中文分词模型依赖，便于复现。

## 常见问题

**1. 为什么不用预训练模型？**  
课程要求是“基于 Transformer 的机器翻译”，本项目主线是从零实现 Transformer，更能体现模型结构和训练流程。

**2. 为什么中文按字符切分？**  
字符级中文 tokenizer 简单稳定，不需要 jieba 或 SentencePiece 训练。缺点是序列更长、BLEU 可能偏低，可以在报告中作为分析点。

**3. 训练效果不好怎么办？**  
先确认 loss 能下降；再增加训练 epoch、使用完整数据、适当调大 batch size，或改用更大的词表和更长训练时间。

**4. 要提交哪些内容？**  
提交源代码、README、`reports/report_template.md` 填写后的实验报告，以及按 `reports/demo_script.md` 录制的 1-2 分钟演示视频。
