# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
source release/bin/activate
source paths.sh

MODEL_OR_ORACLE="model"

JOB_NAME="sweep_models"

# Sweep that trains all the models explained in the paper.
for SPLIT_TYPE in "iid" #"comp" "color_count" "color_location" "color_material" \
    #"color" "shape" "color_boolean" "length_threshold_10"
do
    for MODALITY in "image" 
    do
        for POOLING in  "trnsf" "rel_net" "gap" "concat"
        do
            for JOB_TYPE in "train" # "eval"
            do
                for LANGUAGE_ALPHA in "0.0"
                do 
                    for NEGATIVE_TYPE in "alternate_hypotheses"
                    do
                        for REPLICA_JOBS in "0" "1" "2"
                        do
    				    EVAL_STR=""
                        if [ $JOB_TYPE = "eval" ];
                        then
                            EVAL_STR="eval_cfg.write_raw_metrics=True"
                        fi
                        CMD="python hydra_${JOB_TYPE}.py \
    	     				    ${EVAL_STR}\
                                hydra.job.name=${JOB_NAME}\
                                mode=${JOB_TYPE} \
                                model_or_oracle_metrics=${MODEL_OR_ORACLE} \
                                modality=${MODALITY}\
                                pooling=${POOLING}\
                                data.split_type=${SPLIT_TYPE} \
                                data.negative_type=${NEGATIVE_TYPE}\
                                loss.params.alpha=${LANGUAGE_ALPHA} \
                                job_replica=${REPLICA_JOBS}"
                        echo ${CMD}
                        eval ${CMD}
                        done
                    done
                done
            done
        done
    done
done