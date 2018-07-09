#!/bin/bash

source ~/begin.sh

echo "${GF}, ${CPROB}"
python train_ours.py ~/data cifar10 -s ./snapshots --log ./logs --gold_fraction ${GF} --corruption_prob ${CPROB} --corruption_type "unif" --seed ${SEED}
