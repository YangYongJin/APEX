#!/bin/bash

#cd ../..

# custom config
# DATA=/path/to/datasets
TRAINER=ZeroshotCLIP
DATA=../datasets
# CFG=$2  # rn50, rn101, vit_b32 or vit_b16
CFG=vit_b16
SEED=$2

if [ "${1}" == "0" ]; then
    DATASET=oxford_pets
elif [ "${1}" == "1" ]; then
    DATASET=oxford_flowers
elif [ "${1}" == "2" ]; then
    DATASET=dtd
elif [ "${1}" == "3" ]; then
    DATASET=eurosat
elif [ "${1}" == "4" ]; then
    DATASET=caltech101
elif [ "${1}" == "5" ]; then
    DATASET=fgvc_aircraft
elif [ "${1}" == "6" ]; then
    DATASET=ucf101
elif [ "${1}" == "7" ]; then
    DATASET=imagenet_a
elif [ "${1}" == "8" ]; then
    DATASET=imagenet_r
elif [ "${1}" == "9" ]; then
    DATASET=food101
elif [ "${1}" == "10" ]; then
    DATASET=food101
elif [ "${1}" == "11" ]; then
    DATASET=imagenetv2
elif [ "${1}" == "12" ]; then
    DATASET=imagenet_sketch
elif [ "${1}" == "13" ]; then
    DATASET=imagenet
elif [ "${1}" == "14" ]; then
    DATASET=stanford_cars
fi

python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/CoOp/${CFG}.yaml \
--output-dir output/${TRAINER}/${CFG}/${DATASET} \
--eval-only