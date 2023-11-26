#!/bin/bash

# custom config
DATA=../datasets
TRAINER=APEX
CFG=cross_eval
SHOTS=16
SEED=$1

# Declare an associative array to hold accuracy for each dataset
declare -A ACCURACIES

# List of datasets to iterate over
DATASETS=("caltech101" "oxford_pets" "stanford_cars" "oxford_flowers" "food101" "fgvc_aircraft" "sun397" "dtd" "eurosat" "ucf101" "imagenetv2" "imagenet_a" "imagenet_r" "imagenet_sketch")

# Iterate over each dataset in the list
for DATASET in "${DATASETS[@]}"
do
  DIR=output/evaluation/${TRAINER}/${CFG}_${SHOTS}shots/${DATASET}/seed${SEED}

  # Remove the output directory if it exists
  rm -rf $DIR
  echo "Running job for ${DATASET} and saving the output to ${DIR}"

  # Run the training script for the current dataset
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

  # Get the accuracy from the log file
  accuracy=$(tail -n 5 $DIR/log.txt | grep accuracy | awk -F': ' '{print $2}' | sed 's/%//g')

  # Store the accuracy in the associative array with the dataset name as the key
  ACCURACIES[$DATASET]=$accuracy
done



# Report all accuracies at the end
echo "Accuracies for each dataset:"
for DATASET in "${!ACCURACIES[@]}"
do
  echo "${DATASET}: ${ACCURACIES[$DATASET]}%"
done