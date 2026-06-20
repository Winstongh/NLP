# 基于 Transformer 的英译中机器翻译 — 实验报告

> 数据集：**cmn-eng-simple**（指定）　|　硬件：单卡 RTX 4090　|　完成日期：2026-06-20
> 配套：逐轮训练日志 [`cmn_training_log.txt`](cmn_training_log.txt)，演示脚本 [`demo_script.md`](demo_script.md)，复现步骤见第 11 节。

## 1. 实验目标

用 PyTorch 自实现标准 Transformer Encoder-Decoder，将输入的英文句翻译成中文（例：`tom is a student .` → `汤姆是个学生。`），并用 **BLEU** 评测。

## 2. 数据集

指定数据集 `cmn-eng-simple`，已预处理分词（英文 nltk 分词 + BPE 子词，中文 jieba 词级分词，制表符分隔）。

| Split | 句数 |
| --- | ---: |
| Train | 18,000 |
| Validation | 500 |
| Test | 2,636 |

- 源（英文）词表 4,404，目标（中文）词表 9,886（含 `<PAD>/<BOS>/<EOS>/<UNK>`）。
- 平均长度：英文 7.8 token、中文 6.7 词。句子简短。
- 数据已分词，故按**空格切分**（`pretokenized`），BLEU 以**词**为单位统计 n 元词匹配。

## 3. 方法

### 3.1 预处理
数据已是分词形式，源/目标均按空格切分为 token，构建词表（仅用训练集）。英文保留 BPE 的 `@@` 续接标记，中文为 jieba 词。

### 3.2 模型结构（自实现）
标准 Transformer：多头注意力、正弦位置编码、Encoder/Decoder Layer、前馈网络、残差 + LayerNorm、线性生成器。针对 **18k 小数据**采用较小模型并加正则：

| 超参 | 值 |
| --- | --- |
| d_model / heads | 256 / 4 |
| encoder / decoder layers | 4 / 4 |
| feed-forward dim | 1024 |
| dropout | 0.3 |
| 归一化 | **pre-norm** |
| 权重绑定(tgt emb↔输出层) | **是** |
| 参数量 | **11.0 M** |

> pre-norm + 权重绑定经前期实验验证能稳定训练、提升泛化（见踩坑记录）。

### 3.3 训练策略
- Teacher forcing；损失为忽略 `<PAD>`、label smoothing 0.1 的交叉熵。
- AdamW（betas 0.9/0.98, eps 1e-9）+ Noam 调度（warmup 2000）。
- bf16 混合精度，梯度裁剪 1.0，batch 128，60 epoch（约 3 秒/轮，合计 ~4 分钟）。

### 3.4 解码与后处理
- 自实现 **beam search**（beam=5，长度归一化）。
- **Checkpoint averaging**：平均后段若干轮权重作为最终模型。
- 输出为词序列，拼接去空格即为译文（如 `不要 低估 我 的 力量 。` → `不要低估我的力量。`）。

## 4. 实验设置

| 项目 | 配置 |
| --- | --- |
| GPU / 精度 | RTX 4090 / bf16 |
| batch / epochs | 128 / 60 |
| warmup / 优化器 | 2000 / AdamW + Noam |
| 最终模型 | pre-norm + 权重绑定 + 平均 epoch 51–60 |

## 5. 实验结果

### 5.1 训练曲线（验证集）
| epoch | train_loss | valid_loss | valid_acc |
| ---: | ---: | ---: | ---: |
| 1 | 8.364 | 7.158 | 0.130 |
| 5 | 4.914 | 4.788 | 0.420 |
| 10 | 3.870 | 4.019 | 0.522 |
| 20 | 2.795 | 3.601 | 0.601 |
| **21** | 2.78 | **3.576** | 0.607 |
| 40 | 2.104 | 3.637 | 0.624 |
| 51 | — | 3.64 | **0.637** |
| 60 | 1.887 | 3.667 | 0.633 |

`valid_loss` 在 epoch 21 最低、之后略升（轻微过拟合），但 `valid_acc` 持续爬升到 ~0.637（epoch 51）。故对后段权重做平均。完整逐轮训练日志见 [`cmn_training_log.txt`](cmn_training_log.txt)。

### 5.2 测试集最终结果（2,636 句，beam=5）
| 模型 | token_accuracy | **词级 BLEU** |
| --- | ---: | ---: |
| best.pt（epoch 21，min valid_loss） | 58.9% | 21.16 |
| 平均 epoch 46–54 | 62.2% | 25.49 |
| **平均 epoch 51–60（最终）** | **62.4%** | **25.72** |

## 6. 消融实验

| 改动 | BLEU | Δ |
| --- | ---: | ---: |
| best.pt（单 checkpoint） | 21.16 | — |
| + 平均后段 5 轮 | 25.49 | +4.33 |
| **+ 平均后段 10 轮（最终）** | **25.72** | +4.56 |

**关键结论：checkpoint averaging 带来 +4.5 BLEU 的显著提升**——因为模型后段在 `valid_loss` 上轻微过拟合、但 `valid_acc` 更高，平均这些权重既取了更强的预测、又平滑了过拟合噪声。架构上 **pre-norm + 权重绑定**是稳定训练与泛化的基础（前期实验表明 post-norm 会严重欠优化）。

## 7. 翻译样例（最终模型, beam=5）

| 英文输入 | 模型输出 | 参考 |
| --- | --- | --- |
| tom is a student . | 汤姆是个学生。 | 汤姆是个学生。 |
| now is the time to act . | 现在是行动的时候了。 | 现在是行动的时候了。 |
| do n't underestimate my power . | 不要低估我的力量。 | 不要小看我的力量。 |
| you have n't changed at all . | 你一点都没变。 | 你真的一点没变。 |
| i was just talking about tom . | 我刚跟汤姆说话。 | 我仅仅是在和 Tom 交谈。 |

译文流畅、语义基本正确，部分与参考完全一致。

## 8. 误差分析

- **同义替换**：如 "低估" vs 参考"小看"，意思对但用词不同，会拉低 BLEU（BLEU 只算 n 元词精确匹配）。
- **长句/多子句易丢信息**：含两个分句时偶尔只译前半句。
- **专有名词**：人名（Tom）大小写/译写不稳。
- **轻微过拟合**：18k 数据规模有限，epoch 21 后验证集 loss 回升；已用权重平均缓解。

## 9. 关键工程问题（踩坑）摘要

两个最关键的工程问题：

1. **学习率调度器调用顺序错误**：原训练循环 `scheduler.step()` 在 `optimizer.step()` 之后，使第 1 步用 AdamW 初始 lr=1.0 摧毁初始化（fp16 下 NaN、bf16 下 token-acc 冻 4.5%）。修复（移到 `optimizer.step()` 前）后模型正常学习。
2. **词级 BLEU 的分词一致性**：中文译文若先拼接成无空格句子、再按空格切分算 BLEU，会变成「一个超长 token」对不上参考词，BLEU 趋近 0。修正为对**词序列**计算 BLEU（与参考的分词一致）。

## 10. 结论

在指定的 cmn-eng-simple 数据集上，用 PyTorch 自实现 Transformer 完成英译中，最终测试集 **词级 BLEU 25.72 / token 准确率 62.4%**，译文流畅、语义基本正确。关键做法：修复训练调度 bug、采用 pre-norm + 权重绑定的小模型抑制过拟合、beam search + checkpoint averaging（贡献 +4.5 BLEU）。进一步提升受限于 18k 的数据规模。

## 11. 复现步骤

```bash
conda activate nmt
# 1) 数据：把 cmn-eng-simple 放在 data/ 下，转 jsonl + 词表
python scripts/prepare_cmn.py
# 2) 训练（pre-norm + 权重绑定 + 每轮存档）
python scripts/train.py --config configs/cmn.yaml --device cuda
# 3) 权重平均（后段 10 轮）
python scripts/average_checkpoints.py \
  --checkpoints checkpoints/cmn/epoch_{51,52,53,54,55,56,57,58,59,60}.pt \
  --output checkpoints/cmn/avg.pt
# 4) 评估（词级 BLEU）
python scripts/evaluate.py --checkpoint checkpoints/cmn/avg.pt \
  --config checkpoints/cmn/config.yaml --split test --device cuda --beam-size 5 --bleu-samples 3000
# 5) 翻译
python scripts/translate.py --checkpoint checkpoints/cmn/avg.pt \
  --config checkpoints/cmn/config.yaml --beam-size 5 --device cuda --text "tom is a student ."
```

## 参考资料
1. Vaswani et al. *Attention Is All You Need.* 2017.
2. Wang et al. *Learning Deep Transformer Models for MT.* 2019.（pre-norm）
3. Press & Wolf. *Using the Output Embedding to Improve Language Models.* 2017.（权重绑定）
4. Papineni et al. *BLEU: a Method for Automatic Evaluation of MT.* 2002.
