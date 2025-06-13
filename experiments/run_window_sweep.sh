#!/usr/bin/env bash
for W in 1 3 5; do
  for N in 1000 2500 4000; do
    python -m src.train --window $W --n_train $N --epochs 5 \
      | tee experiments/w${W}_n${N}.log
  done
done
