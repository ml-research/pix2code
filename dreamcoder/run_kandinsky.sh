#!/bin/bash

#----------------------------------------------------------#
N_TASKS=100
CPUS=96
TASK_FOLDER="kandinsky"
EVAL=false
DATE=`date +%Y-%m-%d-%H-%M`
COMPRESSOR="ocaml"
SEED=$1
#----------------------------------------------------------#

echo "Start experiment for ${TASK_FOLDER} with time stamp ${DATE}"
python -u bin/relations.py --task_folder $TASK_FOLDER --number_tasks $N_TASKS -c $CPUS --compressor $COMPRESSOR --seed $SEED> consoleOutputs/$TASK_FOLDER/${DATE}_kandinsky_${N_TASKS}_${COMPRESSOR}${EVAL_STR}.out 2> consoleOutputs/$TASK_FOLDER/${DATE}_kandinsky_${N_TASKS}_${COMPRESSOR}${EVAL_STR}.err