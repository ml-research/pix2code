#!/bin/bash

#----------------------------------------------------------#
N_TASKS=100
CPUS=96
TASK_FOLDER="color_count"
EVAL=0
DATE=`date +%Y-%m-%d-%H-%M`
COMPRESSOR="ocaml"
SEED=0
#----------------------------------------------------------#

echo "Start experiment for ${TASK_FOLDER} with time stamp ${DATE}"
python -u bin/clevr_revise_color_count.py --task_folder $TASK_FOLDER --number_tasks $N_TASKS -c $CPUS --compressor $COMPRESSOR --seed $SEED --eval $EVAL> consoleOutputs/clevr_revised/$TASK_FOLDER/train/$SEED/${DATE}_clevr_${N_TASKS}.out 2> consoleOutputs/clevr_revised/$TASK_FOLDER/train/$SEED/${DATE}_clevr_${N_TASKS}.err
