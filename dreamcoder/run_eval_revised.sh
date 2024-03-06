#!/bin/bash

#----------------------------------------------------------#
N_TASKS=100
CPUS=96
TASK_FOLDER=$1
EVAL=1
DATE=`date +%Y-%m-%d-%H-%M`
SEED=$2
DOMAIN="clevr_revised"
#----------------------------------------------------------#

c=0
echo "Start evaluation ${c} for ${TASK_FOLDER} with time stamp ${DATE}"
python -u bin/clevr_revise_color_count.py --task_folder $TASK_FOLDER --number_tasks $N_TASKS -c $CPUS --seed $SEED --eval $EVAL --test_idx $c> consoleOutputs/$DOMAIN/$TASK_FOLDER/eval/$SEED/${DATE}_eval_${c}.out 2> consoleOutputs/$DOMAIN/$TASK_FOLDER/eval/$SEED/${DATE}_eval_${c}.err
