#!/bin/bash

#cd ../..

# custom config
DATA=../datasets
TRAINER=MaPLe

SEED=$2
WEIGHTSPATH=$3

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
fi

CFG=vit_b16_c2_ep5_batch4_2ctx
SHOTS=16
LOADEP=5
SUB_base=base
SUB_novel=new

COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
MODEL_DIR=${WEIGHTSPATH}/base/seed${SEED}
DIR_base=output/base2new/test_${SUB_base}/${COMMON_DIR}
DIR_novel=output/base2new/test_${SUB_novel}/${COMMON_DIR}
if [ -d "$DIR" ]; then
    echo "Results are already available in ${DIR}. Skipping..."
else
    echo "Evaluating model"
    echo "Runing the first phase job and save the output to ${DIR}"
    # Evaluate on base classes
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR_base} \
    --model-dir ${MODEL_DIR} \
    --load-epoch ${LOADEP} \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB_base}

    # Evaluate on novel classes
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR_novel} \
    --model-dir ${MODEL_DIR} \
    --load-epoch ${LOADEP} \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB_novel}



fi