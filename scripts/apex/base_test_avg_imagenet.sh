#!/bin/bash

# custom config
DATA=../datasets
TRAINER=APEX
DATASET=$1
CFG=vit_b16_c2_ep5_batch16_2+2ctx_imagenet
SHOTS=16
LOADEP=5
SUB=new
SEEDS=$(seq 1 20) # Add your seed values here
NUM_SEEDS=20 # Add the number of seeds here

COMMON_DIR_PREFIX=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed
OUTPUT_DIR=output/base2new/test_${SUB}/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}

# initialize variables to hold the sum of all accuracies and macro_f1 scores
sum_accuracy=0
sum_macro_f1=0

for SEED in $SEEDS; do
  COMMON_DIR=${COMMON_DIR_PREFIX}${SEED}
  MODEL_DIR=output/base2new/train_base/${COMMON_DIR}
  DIR=output/base2new/test_${SUB}/${COMMON_DIR}
  if [ -d "$DIR" ]; then
      rm -rf $DIR
      echo "Remove dir and Run this job and save the output to ${DIR}"
      python train.py \
      --root ${DATA} \
      --seed ${SEED} \
      --trainer ${TRAINER} \
      --dataset-config-file configs/datasets/${DATASET}.yaml \
      --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
      --output-dir ${DIR} \
      --model-dir ${MODEL_DIR} \
      --load-epoch ${LOADEP} \
      --eval-only \
      DATASET.NUM_SHOTS ${SHOTS} \
      DATASET.SUBSAMPLE_CLASSES ${SUB}

  else
      echo "Evaluating model"
      echo "Running the first phase job and save the output to ${DIR}"

      python train.py \
      --root ${DATA} \
      --seed ${SEED} \
      --trainer ${TRAINER} \
      --dataset-config-file configs/datasets/${DATASET}.yaml \
      --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
      --output-dir ${DIR} \
      --model-dir ${MODEL_DIR} \
      --load-epoch ${LOADEP} \
      --eval-only \
      DATASET.NUM_SHOTS ${SHOTS} \
      DATASET.SUBSAMPLE_CLASSES ${SUB}
  fi

  # extract accuracy and macro_f1 from log.txt
  accuracy=$(tail -n 5 $DIR/log.txt | grep accuracy | awk -F': ' '{print $2}' | sed 's/%//g')
  macro_f1=$(tail -n 5 $DIR/log.txt | grep macro_f1 | awk -F': ' '{print $2}' | sed 's/%//g')
  # add them to their respective sums
  sum_accuracy=$(echo $sum_accuracy + $accuracy | bc)
  sum_macro_f1=$(echo $sum_macro_f1 + $macro_f1 | bc)
done

# calculate averages
average_accuracy=$(echo "scale=2; $sum_accuracy / $NUM_SEEDS" | bc)
average_macro_f1=$(echo "scale=2; $sum_macro_f1 / $NUM_SEEDS" | bc)

# print the results
echo "Average Accuracy: $average_accuracy%"
echo "Average Macro F1: $average_macro_f1%"

# write averages to file
echo "Average Accuracy: $average_accuracy%" >> $OUTPUT_DIR/average_evaluation_results.txt
echo "Average Macro F1: $average_macro_f1%" >> $OUTPUT_DIR/average_evaluation_results.txt
