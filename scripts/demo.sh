#!/usr/bin/env bash

PY=/home/zj/anaconda3/envs/nmt/bin/python
p(){ sleep "${1:-1.5}"; }
say(){ printf "\n\033[1;36m▶ %s\033[0m\n" "$1"; p 1.2; }

clear
printf "\033[1;33m基于 Transformer 的英译中机器翻译\033[0m\n"
printf "\033[0;33mPyTorch 自实现 · 数据集 cmn-eng-simple (train 18000 / valid 500 / test 2636)\033[0m\n"
p 3

say "1) 自实现的 Transformer 源码"
ls src/nmt/*.py | xargs -n1 basename | tr '\n' ' ' ; echo ; p 3.5

say "2) 数据集样例（英文 → 中文，已分词）"
head -n 2 data/cmn/train.jsonl ; p 4

say "3) 训练日志：loss 持续下降，valid_acc 从 0.13 升到 0.63"
sed -n '5p;9p;14p;24p;44p;64p' reports/cmn_training_log.txt ; p 4.5

say "4) 测试集评估结果（2636 句, beam search, 词级 BLEU）"
$PY -c "import json;d=json.load(open('outputs/cmn_final.json'));print('   token 准确率 = %.1f%%\n   词级 BLEU   = %.2f'%(d['token_accuracy']*100,d['bleu']))"
p 4

say "5) 现场翻译演示"
for s in \
  "tom is a student ." \
  "i like music ." \
  "what time is it ?" \
  "she is reading a book ." \
  "now is the time to act ." \
  "i want to go home ." \
  "you have n't changed at all ." ; do
  printf "   \033[0;37mEN:\033[0m  %s\n" "$s" ; p 0.7
  printf "   \033[1;32mZH:\033[0m  "
  $PY scripts/translate.py --checkpoint checkpoints/cmn/avg.pt --config checkpoints/cmn/config.yaml \
      --beam-size 5 --device cuda --text "$s" 2>/dev/null
  p 1.8
done
p 1

printf "\n\033[1;32m✓ 完成 — 测试集 词级 BLEU 25.72 / token 准确率 62.4%%\033[0m\n"
p 3
