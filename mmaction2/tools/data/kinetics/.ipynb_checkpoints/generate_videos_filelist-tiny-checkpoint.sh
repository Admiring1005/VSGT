#!/usr/bin/env bash

DATASET="tiny-kinetics-400"
cd ../../../
PYTHONPATH=. python tools/data/build_file_list.py ${DATASET} data/${DATASET}/train_256/ --level 2 --format rawframes --num-split 1 --subset train --shuffle
echo "Train filelist for video generated."

PYTHONPATH=. python tools/data/build_file_list.py ${DATASET} data/${DATASET}/val_256/ --level 2 --format rawframes --num-split 1 --subset val --shuffle
echo "Val filelist for video generated."
cd tools/data/kinetics/
