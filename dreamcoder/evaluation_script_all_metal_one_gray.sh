#!/bin/bash

SPLIT="all_metal_one_gray"
SEED=$1

# Set the path to the folder you want to monitor
TARGET_FOLDER="consoleOutputs/clevr/all_metal_one_gray/eval"

# Set the target number of files
TARGET_FILE_COUNT=$(ls -1 "$TARGET_FOLDER" | wc -l)

# Index for eval
IDX=0

echo $SPLIT $SEED $IDX

# Function to run a file inside the container
run_file() {
  sh $1 $2 $3 $4
}

while [ "$IDX" -le 0 ]; do
  echo start loop with $IDX

  # Run the initial file
  run_file run_eval.sh $IDX $SPLIT $SEED
  IDX=$((IDX+1))

done
