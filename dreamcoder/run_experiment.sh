#!/bin/bash

#----------------------------------------------------------#
N_TASKS=100
CPUS=96
TASK_FOLDER=$1
EVAL=0
DATE=`date +%Y-%m-%d-%H-%M`
COMPRESSOR="ocaml"
SEED=$2
#----------------------------------------------------------#

echo "Start experiment for ${TASK_FOLDER} with time stamp ${DATE}" 
python -u bin/clevr.py --task_folder $TASK_FOLDER --number_tasks $N_TASKS -c $CPUS --compressor $COMPRESSOR --seed $SEED --eval $EVAL> consoleOutputs/clevr/$TASK_FOLDER/${DATE}_clevr_${N_TASKS}.out 2> consoleOutputs/clevr/$TASK_FOLDER/${DATE}_clevr_${N_TASKS}.err
