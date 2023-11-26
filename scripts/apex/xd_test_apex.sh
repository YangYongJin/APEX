#!/bin/bash

#cd ../..

# custom config
DATA=../datasets
TRAINER=APEX

DATASET=$1
SEED=$2

CFG=cross_eval
SHOTS=16


DIR=output/evaluation/${TRAINER}/${CFG}_${SHOTS}shots/${DATASET}/seed${SEED}

rm -rf $DIR
echo "Run this job and save the output to ${DIR}"

python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--output-dir ${DIR} \
--model-dir output/imagenet/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED} \
--load-epoch 6 \
--eval-only
