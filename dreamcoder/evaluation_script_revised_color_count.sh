#!/bin/bash

SPLIT="color_count"
SEED=0

# Set the path to the folder you want to monitor
TARGET_FOLDER="consoleOutputs/clevr_revised/$SPLIT/eval"

# Set the target number of files
TARGET_FILE_COUNT=$(ls -1 "$TARGET_FOLDER" | wc -l)

# Index for eval
IDX=0

echo $SPLIT $SEED $IDX

# Function to run a file inside the container
run_file() {
  docker exec -it pix2code sh $1 $2 $3 $4
}

# Build the Docker image
docker stop pix2code
docker rm pix2code
docker compose build
docker compose up -d

while [ "$IDX" -le 125 ]; do
  echo start loop with $IDX

  # Run the initial file
  run_file run_eval_revised.sh $IDX $SPLIT $SEED
  IDX=$((IDX+1))

  docker stop pix2code
  docker rm pix2code
  docker compose build
  docker compose up -d

done
