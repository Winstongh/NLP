# 基于 Transformer 的英译中机器翻译实验报告

## 1. 实验目标

本实验实现一个基于 Transformer Encoder-Decoder 的英文到中文机器翻译系统。项目不使用预训练翻译模型，而是基于 PyTorch 自行实现模型结构、训练流程、评估和推理脚本。

## 2. 数据集

本实验使用 IWSLT 2017 `iwslt2017-en-zh` 中英平行语料。

- 数据来源：https://huggingface.co/datasets/IWSLT/iwslt2017
- 任务方向：English -> Chinese
- 数据格式：每条样本包含英文源句 `src` 和中文目标句 `tgt`
- 划分：train / validation / test

可在此处填写实际样本数：

| Split | 样本数 |
| --- | ---: |
| Train | 待填写 |
| Validation | 待填写 |
| Test | 待填写 |

## 3. 方法

### 3.1 预处理

英文使用正则分词，将单词、数字和标点分为 token；中文使用字符级 tokenization，并保留连续 ASCII 字符。词表仅由训练集构建，避免验证集和测试集信息泄漏。

### 3.2 模型结构

模型采用标准 Transformer Encoder-Decoder 架构：

- Token embedding + sinusoidal positional encoding
- Multi-head self-attention
- Encoder-Decoder cross-attention
- Position-wise feed-forward network
- Residual connection + LayerNorm
- Linear generator 输出目标词表概率

### 3.3 训练策略

训练使用 teacher forcing。目标句加入 `<bos>` 和 `<eos>`，decoder 输入为右移后的目标序列，监督信号为下一个 token。损失函数为带 label smoothing 的交叉熵，并忽略 `<pad>`。

优化器使用 AdamW，学习率采用 Transformer warmup 策略：

```text
lr = d_model^-0.5 * min(step^-0.5, step * warmup_steps^-1.5)
```

## 4. 实验设置

| 项目 | 配置 |
| --- | --- |
| GPU | RTX 4090 48GB |
| d_model | 512 |
| heads | 8 |
| encoder layers | 6 |
| decoder layers | 6 |
| feed-forward dim | 2048 |
| dropout | 0.1 |
| batch size | 96 |
| epochs | 30 |
| label smoothing | 0.1 |
| warmup steps | 4000 |

## 5. 实验结果

填写 `python scripts/evaluate.py ...` 的输出结果。

| Split | Loss | Token Accuracy | BLEU |
| --- | ---: | ---: | ---: |
| Validation | 待填写 | 待填写 | 待填写 |
| Test | 待填写 | 待填写 | 待填写 |

训练 loss 变化可从 `checkpoints/iwslt2017_en_zh/train_log.jsonl` 整理绘图。

## 6. 翻译样例

| 英文输入 | 参考中文 | 模型输出 |
| --- | --- | --- |
| 待填写 | 待填写 | 待填写 |
| 待填写 | 待填写 | 待填写 |
| 待填写 | 待填写 | 待填写 |

## 7. 误差分析

可从以下角度分析：

- 长句翻译是否出现漏译或重复。
- 专有名词、数字、标点是否保留正确。
- 中文语序是否自然。
- 字符级中文建模是否导致词边界信息不足。
- 训练数据规模和训练轮数对 BLEU 的影响。

## 8. 总结

本实验完成了从数据准备、词表构建、Transformer 实现、训练、评估到推理的完整机器翻译流程。实验说明 Transformer 的注意力机制可以有效建模源语言和目标语言之间的对应关系，但翻译质量仍受数据规模、分词方式、训练时长和模型容量影响。

## 参考资料

1. Vaswani et al. Attention Is All You Need. 2017.
2. IWSLT 2017 Dataset: https://huggingface.co/datasets/IWSLT/iwslt2017
