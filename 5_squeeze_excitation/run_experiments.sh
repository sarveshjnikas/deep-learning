#!/bin/bash
# REPRODUCES EVERY NUMBER IN THE README. RUN FROM THIS FOLDER: bash run_experiments.sh
set -e
PY=${PY:-python}
DATA=${DATA:-data}

# 1. LR SELECTION ON THE BASELINE ONLY (BEST VAL ACCURACY, TIES BROKEN BY VAL MACRO-F1, SEED 0)
for lr in 3e-4 1e-3 3e-3 1e-2 3e-2; do $PY train.py --data $DATA --model base --lr $lr --seed 0; done

# 2. MAIN COMPARISON: SAME LR/EPOCHS/AUGMENTATION/SEEDS, ONLY THE SE BLOCKS DIFFER
LR=${LR:-$($PY -c "import json,glob;r=[json.load(open(f)) for f in glob.glob('results/base_lr*_s0.json')];print(max(r,key=lambda x:(x['val_acc'],x['val_f1']))['args']['lr'])")}
for s in 0 1 2; do
  $PY train.py --data $DATA --model base --lr $LR --seed $s --save
  $PY train.py --data $DATA --model se --r 4 --lr $LR --seed $s --save
done

# 3. ABLATION: SE REDUCTION RATIO r
for r in 2 8; do for s in 0 1 2; do $PY train.py --data $DATA --model se --r $r --lr $LR --seed $s; done; done

# 4. TABLES AND FIGURES
$PY analysis.py $DATA
