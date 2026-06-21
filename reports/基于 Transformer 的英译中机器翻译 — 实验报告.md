# 基于 Transformer 的英译中机器翻译实验报告

## 1. 实验目的

本实验完成一个英文到中文的神经机器翻译系统。在 PyTorch 中实现 Transformer Encoder-Decoder 的主要结构，并完成数据处理、训练、解码、测试集评估。

数据集是 `cmn-eng-simple`，其中训练集 18,000 句，验证集 500 句，测试集 2,636 句。最终模型在测试集上达到 **词级 BLEU 25.72**，token 准确率为 **62.4%**。

## 2. 实验环境与文件说明

| 项目 | 内容 |
| --- | --- |
| 深度学习框架 | PyTorch |
| 训练设备 | CUDA + bf16 混合精度；脚本也支持 CPU 推理 |
| 数据集 | `data/cmn-eng-simple` |
| 训练配置 | `configs/cmn.yaml` |
| 模型源码 | `src/nmt/` |
| 训练日志 | `reports/cmn_training_log.txt` |
| 最终评估结果 | `outputs/cmn_final.json` |

主要运行命令如下：

```bash
python scripts/prepare_cmn.py
python scripts/train.py --config configs/cmn.yaml --device cuda
python scripts/average_checkpoints.py \
  --checkpoints checkpoints/cmn/epoch_{51,52,53,54,55,56,57,58,59,60}.pt \
  --output checkpoints/cmn/avg.pt
python scripts/evaluate.py --checkpoint checkpoints/cmn/avg.pt \
  --config checkpoints/cmn/config.yaml --split test --device cuda --beam-size 5
```

## 3. 数据处理

数据本身已经做过预分词，所以这里没有重新训练 BPE 或 jieba 分词器。处理流程如下：

1. 读取原始英中句对。
2. 按数据集划分生成 `train.jsonl`、`validation.jsonl` 和 `test.jsonl`。
3. 只使用训练集构建英文词表和中文词表，避免把验证集、测试集的信息提前放进词表。
4. 训练和评估时按空格切 token。

训练集中的样例如下：

```json
{"src": "it 's none of your concern .", "tgt": "这不关 你 的 事 。"}
{"src": "she has a habit of bi@@ ting her na@@ ils .", "tgt": "她 有 咬 指甲 的 习惯 。"}
```

英文端保留数据中已有的 `@@` 续接标记，中文端保留已经切好的词。模型输出仍然是中文 token 序列，展示译文时再去掉 token 之间的空格，例如 `不要 低估 我 的 力量 。` 最后显示为 `不要低估我的力量。`。

## 4. 模型与训练方法

### 4.1 模型结构

模型是自实现的 Transformer，没有使用 PyTorch 封装好的完整翻译模型。实现内容包括：

- 多头自注意力和 Encoder-Decoder cross attention；
- 正弦位置编码；
- Encoder layer 和 Decoder layer；
- 前馈网络、残差连接和 LayerNorm；
- target embedding 与输出层权重绑定；
- greedy search 和 beam search 解码。

模型规模没有设得很大，主要是考虑到数据集只有 18k 条训练样本。最终配置如下：

| 超参 | 值 |
| --- | ---: |
| `d_model` | 256 |
| attention heads | 4 |
| encoder layers | 4 |
| decoder layers | 4 |
| feed-forward dim | 1024 |
| dropout | 0.3 |
| LayerNorm 方式 | pre-norm |
| target embedding / output 权重绑定 | 是 |
| 参数量 | 约 11.0 M |

前期试过 post-norm，收敛不如 pre-norm 稳定。这个数据规模下，pre-norm 加较大的 dropout 更容易训起来。target embedding 和输出层做权重绑定后，验证集表现也更稳一些。

### 4.2 训练策略

训练时采用 teacher forcing，损失函数为交叉熵，忽略 `<PAD>`，并加入 0.1 的 label smoothing。优化器使用 AdamW，学习率采用 Noam 调度，warmup steps 为 2000，梯度裁剪为 1.0。

训练共 60 个 epoch，batch size 为 128。日志中记录的单轮时间约 1.8 到 2.1 秒，完整训练时间约 4 分钟。训练脚本使用 bf16 混合精度，因此在有 CUDA 的机器上速度比较快。

### 4.3 解码和模型选择

测试时使用 beam search，beam size 设为 5，并加入长度归一化。最终提交的模型不是单个 epoch 的 checkpoint，而是平均 epoch 51 到 60 的权重。

这样做的原因是：epoch 21 的 `valid_loss` 最低，但后面的 `valid_acc` 仍然继续上升。直接取最低 loss 的模型，BLEU 只有 21.16；平均后段权重后，BLEU 提升到 25 分以上。

## 5. 实验结果

### 5.1 训练过程

训练日志中可以看到，前期 loss 下降很明显，验证集准确率也从 0.13 提升到 0.63 左右。

| epoch | train_loss | valid_loss | valid_acc |
| ---: | ---: | ---: | ---: |
| 1 | 8.364 | 7.158 | 0.130 |
| 5 | 4.914 | 4.788 | 0.420 |
| 10 | 3.870 | 4.019 | 0.522 |
| 20 | 2.795 | 3.601 | 0.601 |
| 21 | 2.732 | **3.576** | 0.607 |
| 40 | 2.104 | 3.637 | 0.624 |
| 51 | 1.962 | 3.657 | **0.637** |
| 60 | 1.887 | 3.667 | 0.633 |

这里也能看出一个问题：epoch 21 之后训练 loss 继续下降，但验证 loss 不再下降，说明模型已经开始轻微过拟合。不过验证准确率还在小幅波动上升，所以我没有简单停在 epoch 21，而是用 checkpoint averaging 降低单个 checkpoint 的波动。

### 5.2 测试集指标

最终测试集结果来自 `outputs/cmn_final.json`：

| 模型 | 测试句数 | beam size | token accuracy | 词级 BLEU |
| --- | ---: | ---: | ---: | ---: |
| best.pt（epoch 21，valid_loss 最低） | 2,636 | 5 | 58.9% | 21.16 |
| 平均 epoch 46-54 | 2,636 | 5 | 62.2% | 25.49 |
| 平均 epoch 51-60（最终） | 2,636 | 5 | **62.4%** | **25.72** |

从结果看，checkpoint averaging 是本实验中最明显的提升点，带来了约 4.5 BLEU 的提高。

### 5.3 翻译样例

| 英文输入 | 模型输出 | 参考译文 |
| --- | --- | --- |
| `tom is a student .` | 汤姆是个学生。 | 汤姆是个学生。 |
| `now is the time to act .` | 现在是行动的时候了。 | 现在是行动的时候了。 |
| `do n't underestimate my power .` | 不要低估我的力量。 | 不要小看我的力量。 |
| `you have n't changed at all .` | 你一点都没变。 | 你真的一点没变。 |
| `could you please talk a bit louder ? i ca n't hear very well .` | 请你说话大声一点吗？ | 你能大声点讲吗？我听不太清。 |

短句的效果比较稳定，常见句式可以翻得比较准确。长一点的句子更容易丢信息，比如最后一个例子只翻出了“请说大声一点”，没有翻出“我听不太清”。

## 6. 分析

### 6.1 有效的部分

本实验里比较有效的改动主要有三个：

1. **学习率调度顺序修正**：一开始训练不稳定，检查后发现学习率更新顺序会影响 Noam schedule 的实际效果，修正后模型才正常收敛。
2. **pre-norm + 权重绑定**：小数据集上训练更稳，参数量也更合理。
3. **beam search + checkpoint averaging**：beam search 让输出比 greedy 更稳定；checkpoint averaging 明显提升 BLEU。

### 6.2 主要问题

1. **BLEU 对同义表达不友好**：例如模型输出“低估”，参考译文是“小看”，意思接近，但 n-gram 匹配还是会扣分。
2. **长句容易漏译**：两个分句以上的句子，模型有时只保留前半部分信息。
3. **专有名词不统一**：Tom 有时翻译成“汤姆”，有时保留英文 `Tom`。
4. **数据规模有限**：训练集只有 18k 句，后期已经能看到过拟合迹象。

## 8. 结论

本实验用 PyTorch 自实现 Transformer 完成了 cmn-eng-simple 数据集上的英译中任务。最终模型使用 4 层 Encoder、4 层 Decoder，参数量约 11M，在测试集 2,636 句上取得 **词级 BLEU 25.72 / token 准确率 62.4%**。从实验过程看，模型已经能比较稳定地处理简单短句，但长句漏译、专有名词不统一和同义表达评分偏低仍然明显。