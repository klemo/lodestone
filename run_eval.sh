#!/bin/bash

for i in 64 96 128 160 192 224 256
  do
    python lodestone.py --i  ~/Downloads/project_gutenberg/b100_out/ --o "outputs_b100/${i}_44_char.csv" --k 5 5 --l $i
done
