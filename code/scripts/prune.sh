#!/bin/sh

work_dir="work_dir"

mkdir -p ${work_dir}
echo save results to ${work_dir}

python lstmModel.py \
    pruneFunction \
    ${work_dir}/"$1" \
    "$2" \
    "$3"  