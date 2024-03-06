#!/bin/bash

#----------------------------------------------------------#
N_TASKS=100
CPUS=96
TASK_FOLDER="kandinsky"
EVAL=1
DATE=`date +%Y-%m-%d-%H-%M`
SEED=$1
DOMAIN="kandinsky"
#----------------------------------------------------------#

c=0
echo "Start evaluation ${c} for ${TASK_FOLDER} with time stamp ${DATE}"
python -u bin/relations.py --task_folder $TASK_FOLDER --number_tasks $N_TASKS -c $CPUS --seed $SEED --eval $EVAL --test_idx $c> consoleOutputs/$DOMAIN/eval/$SEED/${DATE}_eval_${c}.out 2> consoleOutputs/$DOMAIN/eval/$SEED/${DATE}_eval_${c}.err
