#!/bin/bash

# Copyright  2021 Matt Haynes <matt.haynes@bbc.co.uk>
#            2021 Misa Ogura <misa.ogura@bbc.co.uk>
# Apache 2.0

if [ -z "${KALDI_ROOT:-}" ]; then
  . kaldi-path
else
  [ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh

  export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH

  if [ ! -f $KALDI_ROOT/tools/config/common_path.sh ]; then
    echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!"
    exit 1
  fi

  . $KALDI_ROOT/tools/config/common_path.sh

  export LC_ALL=C.UTF-8
fi

export DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

# This ridiculously ugly regex removes any paths set by Kaldi 
# in the $PATH env variable:

export PATH=$DIR:$DIR/local:$DIR/steps:$DIR/utils:$PATH
