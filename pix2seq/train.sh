#!/bin/bash

python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --coco_path "data/pattern_free_clevr" --pix2seq_lr --large_scale_jitter --rand_target $@
