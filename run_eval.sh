#!/bin/bash

for i in 64 96 128 160 192 224 256
  do
    python lodestone.py --i  ~/Downloads/project_gutenberg/b100_out_p4/ --o "outputs_b100_p4/char_sha_44_${i}.csv" --k 4 4 --l $i
done