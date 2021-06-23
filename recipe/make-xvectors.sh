#!/bin/bash

# Copyright  2021 Matt Haynes <matt.haynes@bbc.co.uk>
#            2021 Misa Ogura <misa.ogura@bbc.co.uk>
# Apache 2.0

set -eo pipefail

# -----------------------------------------------------------------------------
# General variables

help_message=$(cat <<'EOF'
usage: make-xvectors.sh input.scp input.stm output-dir

options:
  --help                   Print this message
EOF
)

export log_level=4
export log_format="text"
export nj=$(nproc)
export stage=1

export DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export MODEL_DIR=$DIR/model

. $DIR/path.sh
. $DIR/local/logging.sh

. $DIR/utils/parse_options.sh

WAV_SCP=$1
STM=$2
OUTPUT_DIR=$3

logs_dir=$OUTPUT_DIR/logs
rm -rf $logs_dir
mkdir -p $logs_dir

# -----------------------------------------------------------------------------
# Main

if [[ $# -ne 3 ]]; then
  echo "$help_message" | head -n 1
  exit 1
fi

# Stage 1: Create entru point files

info $stage "Creating entry point files"

mkdir -p $OUTPUT_DIR

cat $WAV_SCP | sort -g -k1 > $OUTPUT_DIR/wav.scp

cat $STM                                                             \
  | sort -k1,1 -k4,4g                                                \
  | awk '{
      count++;
      printf "%s_%010.3f_%010.3f %s %f %f\n",$1,$4,$5,$1,$4,$5
    }' > $OUTPUT_DIR/segments

cat $STM                                                                     \
  | sort -k1,1 -k4,4g                                                        \
  | awk '{
      count++;
      printf "%s_%010.3f_%010.3f %s_%05d\n",$1,$4,$5,$1,count
    }' > $OUTPUT_DIR/utt2spk

$DIR/utils/fix_data_dir.sh $OUTPUT_DIR > $logs_dir/stage_1.log 2>&1 || exit 66

$DIR/utils/utt2spk_to_spk2utt.pl < $OUTPUT_DIR/utt2spk > $OUTPUT_DIR/spk2utt

speakers_length=$(cat $OUTPUT_DIR/spk2utt 2> /dev/null | wc -l)

# Reduce concurrent jobs to number of speakers

if [ "$nj" -gt "$speakers_length" ] ; then
    debug $stage "Reducing number of jobs to $speakers_length for safety"
    nj=$speakers_length
fi

# Stage 2: Extract MFCC features

stage=2

info $stage "Extracting MFCC features"

mfccdir=$OUTPUT_DIR/mfcc

$DIR/steps/make_mfcc.sh --mfcc-config conf/mfcc.conf \
  --nj 1 --write-utt2num-frames true                 \
  $OUTPUT_DIR $mfccdir/log $mfccdir                  \
  > $logs_dir/stage_2.log 2>&1 || exit 66

# Stage 3: Apply CMVN

stage=3

info $stage "Applying CMVN smoothing"

cmndir=$OUTPUT_DIR/cmn

$DIR/local/prepare-feats.sh --nj $nj \
  $OUTPUT_DIR $cmndir $cmndir        \
  > $logs_dir/stage_3.log 2>&1 || exit 66

cp $OUTPUT_DIR/segments $cmndir
$DIR/utils/fix_data_dir.sh $cmndir > $logs_dir/stage_3.log 2>&1 || exit 66

# Stage 4: Extract xvectors

stage=4

info $stage "Extracting x-vectors"

xvectordir=$OUTPUT_DIR/xvectors
nnet_dir=$MODEL_DIR

$DIR/local/extract-xvectors.sh                          \
  --nj $nj --window 1.5 --period 0.75 --apply-cmn false \
  --min-segment 0.5 $nnet_dir $cmndir $xvectordir       \
  > $logs_dir/stage_4.log 2>&1 || exit 66

# Combine ark files across jobs

cat $xvectordir/xvector.*.ark > $xvectordir/xvector.all.ark

# Convert ark to text format

ark=$xvectordir/xvector.all.ark
arktxt=$xvectordir/xvector.all.ark.txt

copy-vector ark:$ark ark,t:$arktxt > $logs_dir/stage_7.log 2>&1 || exit 6

info $stage "X-vector extraction completed"
