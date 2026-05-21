#!/bin/bash

PROJECT_ROOT=$(pwd)
experiment_home_dir="experiments"
config="tbsn_confocal_512.json"

experiment_name=$(python ${PROJECT_ROOT}/train/experiment_name.py --config ${PROJECT_ROOT}/option/${config})
#experiment_name='test'

experiment_dir="${PROJECT_ROOT}/${experiment_home_dir}/${experiment_name}"
echo "experiment dir: ${experiment_dir}"

mkdir -p "${experiment_dir}/log"

cd ${experiment_dir}
export PYTHONPATH=${PROJECT_ROOT}:$PYTHONPATH
python ${PROJECT_ROOT}/train/base.py --config ${PROJECT_ROOT}/option/${config}