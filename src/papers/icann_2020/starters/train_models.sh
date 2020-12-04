#!/bin/sh

model_type=$1 # one of mlt, ata, sae
dataset=$2 # one of ARR, TREC
dataset_path="../gs_configs/${model_type}/${dataset}.yml"

python ../../../outlier_detection/run.py --model_type $model_type \
                                    --run_mode TRAIN \
                                    --process_count 3 \
                                    --dashify_logging_path ../../../../dashify_logs/ \
                                    --text_logging_path ../../../../general_logs/ \
                                    --gs_config_path $dataset_path \
                                    --num_epochs 1