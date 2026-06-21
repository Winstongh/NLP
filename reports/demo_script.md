# 1–2 分钟程序演示视频脚本（cmn-eng-simple）

> 目标：1–2 分钟内展示「数据 → 训练 → 评估(BLEU) → 翻译」完整流程。

## 已自动录制（终端实录，约 53 秒）

整个演示已用一键脚本跑通并录制：

```bash
bash scripts/demo.sh                                  # 直接看演示（终端里运行）
asciinema rec reports/demo.cast -c "bash scripts/demo.sh"   # 重新录制
asciinema play reports/demo.cast                      # 回放录制
```

产物：
- `reports/demo.cast` — asciinema 终端录像（可回放 / 上传 asciinema.org 得分享链接）
- `reports/demo.svg` — 动画 SVG，**浏览器直接打开即可播放**

**得到最终 .mp4 视频**（任选其一）：
1. 浏览器打开 `reports/demo.svg`，用任意录屏软件录这 53 秒 → 导出 mp4（最简单）；
2. 或终端 `asciinema play reports/demo.cast` 时录屏；
3. 如需配音解说，照下面分镜讲解即可。

> 注：我（助手）无法直接产出带语音的屏幕录像 mp4，上面是已经录好的程序实跑画面，你做最后一步录屏/配音即可。

---

## 分镜讲稿（可照着配音）

> 录屏前先 `conda activate nmt`，工作目录 `/home/zj/cw/NLP`。

## 0:00–0:15 项目简介
说明：PyTorch **自实现** Transformer Encoder-Decoder，英文→中文翻译，数据集 cmn-eng-simple（train 18000 / valid 500 / test 2636），评测指标 **BLEU**。

建议画面：
```bash
ls src/nmt/                 # 自实现模型/训练/评估/推理源码
head -2 data/cmn/train.jsonl   # 数据样例（英文/中文）
```

## 0:15–0:35 数据准备
说明：数据已分词（英文 BPE、中文 jieba 词），脚本转成 jsonl 并构建词表。

```bash
python scripts/prepare_cmn.py
# train=18000 valid=500 test=2636 ; en vocab=4404 zh vocab=9886
```

## 0:35–0:55 训练
说明：小模型（d256/4+4，pre-norm+权重绑定）抑制过拟合；展示训练日志中 loss 下降、valid_acc 上升。

```bash
python scripts/train.py --config configs/cmn.yaml --device cuda
# 实时日志见 reports/cmn_training_log.txt：valid_acc 0.13 → 0.63
tail -n 5 reports/cmn_training_log.txt
```
（视频中可直接展示已训练好的日志，无需现场等训。）

## 0:55–1:20 评估（BLEU）
说明：权重平均 + beam search，在 2636 句测试集上算**词级 BLEU**。

```bash
python scripts/average_checkpoints.py \
  --checkpoints checkpoints/cmn/epoch_{51,52,53,54,55,56,57,58,59,60}.pt \
  --output checkpoints/cmn/avg.pt

python scripts/evaluate.py --checkpoint checkpoints/cmn/avg.pt \
  --config checkpoints/cmn/config.yaml --split test --device cuda --beam-size 5 --bleu-samples 3000
# 输出：token_accuracy ≈ 0.62, bleu ≈ 25.7
```

## 1:20–1:50 翻译演示
说明：现场输入英文句子，输出中文译文。

```bash
python scripts/translate.py --checkpoint checkpoints/cmn/avg.pt \
  --config checkpoints/cmn/config.yaml --beam-size 5 --device cuda \
  --text "now is the time to act ."
# → 现在是行动的时候了。
```
可多翻几句（输入用小写、空格分词的形式）：
```text
do n't underestimate my power .      → 不要低估我的力量。
you have n't changed at all .        → 你一点都没变。
tom is a student .                   → 汤姆是学生。
```

## 1:50–2:00 总结
- 自实现 Transformer，完整数据/训练/评估/推理流程。
- 测试集 **词级 BLEU 25.72 / token 准确率 62.4%**。
- 关键点：修复训练调度 bug、pre-norm+权重绑定+checkpoint averaging。
