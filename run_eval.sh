#!/bin/bash

# analyze ngram sizes
for i in 2 3 4 5
  do
    python lodestone.py --i "~/Downloads/project_gutenberg/datasets/b1k_out_p5/" --o "outputs_b1k_p5_ngram/${i}.csv" --k $i $i --l 128
done

# analyze hash length over different corruption rates
# for i in 64 96 128 160 192 224 256
#   do
#   for j in 05 1 2 3 4 5 6 7
#     do
#       python lodestone.py --i "~/Downloads/project_gutenberg/datasets/b1k_out_p${j}/" --o "hashes/b1k/${j}/${i}.csv" --k 4 4 --l $i
#   done
# done