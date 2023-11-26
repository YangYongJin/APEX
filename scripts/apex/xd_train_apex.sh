#!/bin/bash

#cd ../..

# custom config
DATA=../datasets
TRAINER=APEX

DATASET=$1
SEED=$2

CFG=cross_eval
SHOTS=16


DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED}

rm -rf $DIR
echo "Run this job and save the output to ${DIR}"

python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--output-dir ${DIR} \
DATASET.NUM_SHOTS ${SHOTS}
