#!/bin/sh

dataset=$1 # one of ARR, TREC
dataset_path="../gs_configs/SAE/${dataset}.yml"

python ../../../outlier_detection/run.py --model_type SAE \
                                    --run_mode TRAIN \
                                    --process_count 1 \
                                    --dashify_logging_path ../../../../dashify_logs/ \
                                    --text_logging_path ../../../../general_logs/ \
                                    --gs_config_path $dataset_path \
                                    --num_epochs 2