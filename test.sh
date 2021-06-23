#!/bin/bash

# Copyright  2021 Matt Haynes <matt.haynes@bbc.co.uk>
#            2021 Misa Ogura <misa.ogura01@gmail.com>
# Apache 2.0
#
# Simple script to test the repo confirms to various coding standards and the
# the VAD and segmentation scripts function as expected.

set -eo pipefail

# -----------------------------------------------------------------------------
# Check .py files for coding standards

echo "Checking .py files look good"

for py_file in recipe/local/*.py; do

    flake8 "$py_file"

    if ! grep -q "# Copyright" "$py_file"; then
        echo "No Copyright notice in $py_file"
        exit 1
    fi

    if ! grep -q "# Apache 2.0" "$py_file"; then
        echo "No Apache 2.0 license notice in $py_file"
        exit 1
    fi

    if ! grep -q "#!/usr/bin/env python3" "$py_file"; then
        echo "No shebang in $py_file"
        exit 1
    fi

done

# -----------------------------------------------------------------------------
# Check .sh files for coding standards

echo "Checking .sh files look good"

for sh_file in test.sh recipe/*.sh recipe/local/*.sh; do

    if ! grep -q "# Copyright" "$sh_file"; then
        echo "No Copyright notice in $sh_file"
        exit 1
    fi

    if ! grep -q "# Apache 2.0" "$sh_file"; then
        echo "No Apache 2.0 license notice in $sh_file"
        exit 1
    fi

    if ! grep -q "#!/bin/bash" "$sh_file"; then
        echo "No shebang in $sh_file"
        exit 1
    fi

done

# -----------------------------------------------------------------------------
# Run segmentation and check if all works

cd recipe

tmp_dir=$(mktemp -d)

# --help

echo "Checking run-segmentation.sh --help"

./run-segmentation.sh --help 2>&1 \
    | grep -q "usage: run-segmentation.sh"

# --no-vad

echo "Checking run-segmentation.sh --no-vad"

./run-segmentation.sh --no-vad true ../test.wav ../test.stm "$tmp_dir" 2>&1 \
    | grep "INFO 5 Skipping" > /dev/null

./run-segmentation.sh --no-vad true ../test.wav ../test.stm "$tmp_dir" 2>&1 \
    | grep "INFO 8 Skipping" > /dev/null

ls $tmp_dir/diarize.rttm $tmp_dir/diarize.stm > /dev/null

rm -rf $tmp_dir

# --vad-method individual

echo "Checking run-segmentation.sh --vad-method individual"

./run-segmentation.sh --vad-method individual   \
        ../test.wav ../test.stm "$tmp_dir" 2>&1 \
    | grep "INFO 5 Filtering unvoiced xvectors" \
    > /dev/null

./run-segmentation.sh --vad-method individual   \
        ../test.wav ../test.stm "$tmp_dir" 2>&1 \
    | grep "INFO 8 Skipping" > /dev/null

ls $tmp_dir/diarize.rttm $tmp_dir/diarize.stm > /dev/null

rm -rf $tmp_dir

# --vad-method segment

echo "Checking run-segmentation.sh --vad-method segment"

./run-segmentation.sh --vad-method segment      \
        ../test.wav ../test.stm "$tmp_dir" 2>&1 \
    | grep "INFO 8 Filtering unvoiced clusters" \
    > /dev/null

./run-segmentation.sh --vad-method segment      \
        ../test.wav ../test.stm "$tmp_dir" 2>&1 \
    | grep "INFO 5 Skipping" > /dev/null

ls $tmp_dir/diarize.rttm $tmp_dir/diarize.stm > /dev/null

rm -rf $tmp_dir

# -----------------------------------------------------------------------------
# Ensure xvector_utils.py commands work as expected

mkdir -p $tmp_dir

echo "test /wrk/test.wav" > $tmp_dir/test.scp

# make-xvectors

echo "Checking local/xvector_utils.py make-xvectors"

python3 local/xvector_utils.py make-xvectors                            \
        $tmp_dir/test.scp /wrk/test.stm $tmp_dir/test.xvectors.ark 2>&1 \
    | grep "INFO 4 X-vector extraction completed" > /dev/null

if [ ! -s "$tmp_dir/test.xvectors.ark" ]; then
    echo "X-vectors empty"
    exit 1
fi

copy-vector ark:$tmp_dir/test.xvectors.ark \
    ark,t:$tmp_dir/test.xvectors.ark.txt >/dev/null 2>&1

num_xvectors=$( wc -l $tmp_dir/test.xvectors.ark.txt | awk '{print $1}' )

if (( "$num_xvectors" != 38 )); then
    echo "The number of x-vectors extracted doesn't match the expected"
    exit 1
fi

# visualize

echo "Checking local/xvector_utils.py visualize"

python3 local/xvector_utils.py visualize                           \
        /wrk/test.stm $tmp_dir/test.xvectors.ark $tmp_dir/viz 2>&1 \
    | grep "INFO:xvector_utils.main:Saving the dataset" > /dev/null

if [ ! -s "$tmp_dir/viz/visualization.png" ]; then
    echo "Visualization empty"
    exit 1
fi

# visualize

echo "Checking local/xvector_utils.py train"

python3 local/xvector_utils.py train                                     \
        /wrk/test.stm $tmp_dir/test.xvectors.ark $tmp_dir/model.pkl 2>&1 \
    | grep "INFO:xvector_utils.main:Saving the model" > /dev/null

if [ ! -s "$tmp_dir/model.pkl" ]; then
    echo "Model empty"
    exit 1
fi

echo "Checking local/xvector_utils.py evaluate"

python3 local/xvector_utils.py evaluate               \
        /wrk/recipe/data/bbc-vad-eval/reference.stm   \
        /wrk/recipe/data/bbc-vad-eval/xvectors.ark    \
        /wrk/recipe/model/xvector-classifier.pkl 2>&1 \
    | grep "Accuracy: 0.974" > /dev/null

rm -rf $tmp_dir

cd ..

echo "All checks passed"
