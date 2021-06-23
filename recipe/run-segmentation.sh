#!/bin/bash

# Copyright  2021 Matt Haynes <matt.haynes@bbc.co.uk>
#            2021 Misa Ogura <misa.ogura@bbc.co.uk>
# Apache 2.0

set -eo pipefail

# -----------------------------------------------------------------------------
# General variables

help_message=$(cat <<'EOF'
usage: run-segmentation.sh [options] input.wav input.stm output-dir

options:
  --nj NUM                 Maximum number of CPU cores to use
  --stage STAGE            Start from this stage
  --cluster-threshold THR  Cluster stopping criteria. Default: -0.3
  --vad-threshold THR      Xvector classifier threshold. Lower the number the
                           more speech segments shall be returned at the
                           expense of accuracy. Default: 0.2
  --vad-method             Filter segments on an individual or segment basis.
                           Default: individual
  --no-vad                 Skip xvector vad stages. Default: false
  --help                   Print this message
EOF
)

export nj=1                       # Max number of CPU cores to use
export stage=1                    # Start from this stage
export cluster_threshold=-0.3     # Xvector clustering threshold
export vad_threshold=0.2          # Xvector classifier threshold
export vad_method=individual      # Xvector vad method
export no_vad=false               # Apply no VAD method at all

export DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

. $DIR/path.sh
. $DIR/local/logging.sh

. $DIR/utils/parse_options.sh

# -----------------------------------------------------------------------------
# Main

if [[ $# -ne 3 ]]; then
  echo "$help_message" | head -n 1
  exit 1
fi

export input_wav="$1"
export input_stm="$2"
export output_dir="$3"
export log_file=""

if [ $vad_method != "individual" ] && [ $vad_method != "segment" ]; then
   echo "--vad-method must be one of individual or segment"
   exit 1
fi

# Stage 1: Setup entry point files

if [ $stage -le 1 ]; then

  info 1 "Creating entry point files"

  log_file=$output_dir/logs/stage_01.log

  rm -rf $output_dir

  mkdir -p $output_dir $output_dir/logs $output_dir/times

  fileid=`basename $input_wav .wav`;

  echo "$fileid $input_wav" > $output_dir/wav.scp

  cat $input_stm                                                       \
    | awk '{printf "%s_%010.3f_%010.3f %s %f %f\n",$1,$4,$5,$1,$4,$5}' \
    > $output_dir/segments

  cat $input_stm                                                 \
    | awk '{printf "%s_%010.3f_%010.3f %s_%s\n",$1,$4,$5,$1,$3}' \
    > $output_dir/utt2spk

  $DIR/utils/fix_data_dir.sh $output_dir > $log_file 2>&1 \
    || fail 1 "Error running stage 1, see $log_file for details"

fi

# Stage 2: Extract MFCC features

if [ $stage -le 2 ]; then

  info 2 "Extracting MFCC features"

  log_file=$output_dir/logs/stage_02.log

  # Reduce concurrent jobs to number of speakers

  speakers_length=$(cat $output_dir/spk2utt 2> /dev/null | wc -l)

  if [ "$nj" -gt "$speakers_length" ] ; then
    debug 2 "Reducing number of jobs to $speakers_length for safety"
    nj=$speakers_length
  fi

  $DIR/steps/make_mfcc.sh --mfcc-config $DIR/conf/mfcc.conf \
    --nj $nj --write-utt2num-frames true                    \
    --write-utt2dur false                                   \
    $output_dir $output_dir/mfcc/log $output_dir/mfcc       \
    > $log_file 2>&1                                        \
  || fail 2 "Error running stage 2, see $log_file for details"

fi

# Stage 3: Apply CMVN

if [ $stage -le 3 ]; then

  info 3 "Applying CMVN smoothing"

  log_file=$output_dir/logs/stage_03.log

  # Reduce concurrent jobs to number of speakers

  speakers_length=$(cat $output_dir/spk2utt 2> /dev/null | wc -l)

  if [ "$nj" -gt "$speakers_length" ] ; then
    debug 2 "Reducing number of jobs to $speakers_length for safety"
    nj=$speakers_length
  fi

  $DIR/local/prepare-feats.sh --nj $nj            \
    $output_dir $output_dir/cmvn $output_dir/cmvn \
    > $log_file 2>&1                              \
  || fail 3 "Error running stage 3, see $log_file for details"

  cp $output_dir/segments $output_dir/cmvn

  $DIR/utils/fix_data_dir.sh $output_dir/cmvn >> $log_file 2>&1 \
    || fail 3 "Error running stage 3, see $log_file for details"

  cp $output_dir/segments $output_dir/cmvn

  $DIR/utils/fix_data_dir.sh $output_dir/cmvn >> $log_file 2>&1 \
    || fail 3 "Error running stage 3, see $log_file for details"

fi

# Stage 4: Extract xvectors

if [ $stage -le 4 ]; then

  info 4 "Extracting xvectors"

  log_file=$output_dir/logs/stage_04.log

  # Reduce concurrent jobs to number of speakers

  speakers_length=$(cat $output_dir/spk2utt 2> /dev/null | wc -l)

  if [ "$nj" -gt "$speakers_length" ] ; then
    debug 2 "Reducing number of jobs to $speakers_length for safety"
    nj=$speakers_length
  fi

  rm -rf $output_dir/xvectors

  $DIR/local/extract-xvectors.sh    \
    --nj $nj                        \
    --window 1.5                    \
    --period 0.75                   \
    --apply-cmn false               \
    --min-segment 0.5               \
    $DIR/model                      \
    $output_dir/cmvn                \
    $output_dir/xvectors            \
    > $log_file 2>&1                \
  || fail 4 "Error running stage 4, see $log_file for details"

  cat $output_dir/xvectors/xvector.*.ark   \
    > $output_dir/xvectors/xvector.all.ark

fi

# Stage 5: Filter unvoiced xvectors

if [ $stage -le 5 ]; then

  if [ $no_vad == 'true' ]; then

    info 5 "Skipping xvector vad stage as requested"

  elif [ $vad_method == 'segment' ]; then

    info 5 "Skipping individual xvector vad as --vad-method segment applied"

  else

    info 5 "Filtering unvoiced xvectors"

    log_file=$output_dir/logs/stage_05.log

    python $DIR/local/xvector_utils.py  \
      predict                                \
      $output_dir/xvectors/xvector.all.ark   \
      $DIR/model/xvector-classifier.pkl      \
      > $output_dir/xvectors/predictions.txt \
      2> $log_file                           \
    || fail 5 "Error running stage 5, see $log_file for details"

    python $DIR/local/filter_unvoiced_xvectors.py \
      $vad_threshold                              \
      $output_dir/xvectors/predictions.txt        \
      $output_dir/xvectors/segments               \
      > $output_dir/xvectors/segments.filt        \
      2> $log_file                                \
    || fail 5 "Error running stage 5, see $log_file for details"

    cp $output_dir/xvectors/segments.filt $output_dir/xvectors/segments

  fi

fi

# Stage 6: Score xvectors

if [ $stage -le 5 ]; then

  info 6 "Scoring xvectors"

  log_file=$output_dir/logs/stage_06.log

  $DIR/local/score-plda.sh            \
    --target-energy 0.9 --nj 1        \
    $DIR/model                        \
    $output_dir/xvectors              \
    $output_dir/plda                  \
    > $log_file 2>&1                  \
  || fail 5 "Error running stage 6, see $log_file for details"

fi

# Stage 7: Cluster xvectors

if [ $stage -le 7 ]; then

  info 7 "Clustering xvectors"

  log_file=$output_dir/logs/stage_07.log

  $DIR/local/cluster.sh --nj 1     \
    --threshold $cluster_threshold \
    $output_dir/plda               \
    $output_dir/clusters           \
    > $log_file 2>&1               \
  || fail 6 "Error running stage 7, see $log_file for details"

  python $DIR/local/merge_single_xvector_clusters.py \
    $output_dir/clusters/rttm                        \
    > $output_dir/clusters/rttm.merged 2>> $log_file \
  || fail 6 "Error running stage 7, see $log_file for details"

  cp $output_dir/clusters/rttm $output_dir/clusters/rttm.bak

  cp $output_dir/clusters/rttm.merged $output_dir/clusters/rttm

fi

# Stage 8: Filter unvoiced clusters

if [ $stage -le 8 ]; then

  if [ $no_vad == 'true' ]; then

    info 8 "Skipping xvector vad stage as requested"

  elif [ $vad_method == 'individual' ]; then

    info 8 "Skipping segment xvector vad as --vad-method individual applied"

  else

    info 8 "Filtering unvoiced clusters"

    log_file=$output_dir/logs/stage_08.log

    copy-vector ark:$output_dir/xvectors/xvector.all.ark         \
                ark,t:$output_dir/xvectors/xvector.all.ark.txt   \
      > $log_file 2>&1                                           \
    || fail 7 "Error running stage 8, see $log_file for details"

    python $DIR/local/filter_unvoiced_segments.py \
      $vad_threshold                              \
      $DIR/model/xvector-classifier.pkl           \
      $output_dir/clusters                        \
      $output_dir/xvectors/xvector.all.ark.txt    \
      >> $log_file 2>&1                           \
    || fail 7 "Error running stage 8, see $log_file for details"

    old_segs=$(cat $output_dir/clusters/rttm | wc -l)
    new_segs=$(cat $output_dir/clusters/rttm.filt | wc -l)
    dif_segs=$((old_segs-new_segs))

    debug 8 "Removed $dif_segs unvoiced segments"

    cp $output_dir/clusters/rttm.filt $output_dir/clusters/rttm
  fi

fi

# Stage 9: Pad segments

if [ $stage -le 9 ]; then

  info 9 "Padding segments"

  log_file=$output_dir/logs/stage_09.log

  python $DIR/local/pad_segments.py \
    $output_dir/clusters/rttm       \
    > $output_dir/clusters/rttm.pad \
    2> $log_file                    \
  || fail 8 "Error running stage 9, see $log_file for details"

  cp $output_dir/clusters/rttm.pad $output_dir/clusters/rttm

fi

# Stage 10: Convert RTTM to STM

if [ $stage -le 10 ]; then

  info 10 "Converting RTTM to STM"

  python $DIR/local/convert_rttm_to_stm.py \
    $output_dir/clusters/rttm              \
    $output_dir/diarize.stm                \
    > $log_file 2>&1                       \
  || fail 9 "Error running stage 10, see $log_file for details"

  cp $output_dir/clusters/rttm $output_dir/diarize.rttm

fi
