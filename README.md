# bbc-speech-segmenter: Voice Activity Detection & Speaker Diarization

A complete speech segmentation system using Kaldi and x-vectors for voice
activity detection (VAD) and speaker diarisation.

The x-vector-vad system is described in the paper; **Ogura, M. & Haynes, M.
(2021) X-vector-vad for Multi-genre Broadcast Speech-to-text**. The paper has
been submitted to 2021 IEEE Automatic Speech Recognition and Understanding
Workshop (ASRU) and is currently under review as of June 2021.

* [ASRU 2021 website](https://asru2021.org/)

## Quickstart

```
$ docker pull bbcrd/bbc-speech-segmenter

# Test

$ docker run -w /wrk -v `pwd`:/wrk bbcrd/bbc-speech-segmenter ./test.sh

# Segmentation help

$ docker run bbcrd/bbc-speech-segmenter ./run-segmentation.sh --help
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

# Run segmentation (VAD + diarisation), results are in output-dir/diarize.stm

$ docker run -v `pwd`:/data bbcrd/bbc-speech-segmenter \
  ./run-segmentation.sh /data/audio.wav /data/audio.stm /data/output-dir

$ cat output-dir/diarize.stm
audio 0 audio_S00004 3.750 10.125 <speech>
audio 0 audio_S00003 10.125 13.687 <speech>
audio 0 audio_S00004 13.688 16.313 <speech>
...

# Train x-vector classifier

$ docker run -w /wrk/recipe -v `pwd`:/wrk bbcrd/bbc-speech-segmenter \
  local/xvector_utils.py data/bbc-vad-train/reference.stm            \
  data/bbc-vad-train/xvectors.ark new_model.pkl

# Evaluate x-vector classifier

$ docker run -w /wrk/recipe -v `pwd`:/wrk bbcrd/bbc-speech-segmenter \
  local/xvector_utils.py evaluate data/bbc-vad-eval/reference.stm    \
  data/bbc-vad-eval/xvectors.ark model/xvector-classifier.pkl
```

### Audio & STM file format

In order to run the segmentation script you need your audio in **16Khz Mono WAV**
format. You also need an STM file describing the segments you want to apply
voice activity detection and speaker diarization to.

For more information on the STM file format see [`XVECTOR_UTILS.md`](https://github.com/bbc/bbc-speech-segmenter/blob/main/XVECTOR_UTILS.md#labelsstm).

```
# Convert audio file to 16Khz mono wav

$ ffmpeg audio.mp3 -vn -ac 1 -ar 16000 audio.wav

# Create STM file for input

$ DURATION=$(ffprobe -i audio.wav -show_entries format=duration -v quiet -of csv="p=0")
$ DURATION=$(printf "%0.2f\n" $DURATION)

$ FILENAME=$(basename audio.wav)

$ echo "${FILENAME%.*} 0 ${FILENAME%.*} 0.00 $DURATION <label> _" > audio.stm

$ cat audio.stm
audio 0 audio 0.00 60.00 <label> _
```

## Use Docker image to run code in local checkout

```
# Bulid Docker image

$ docker build -t bbc-speech-segmenter .

# Spin up a Docker container in an interactive mode

$ docker run -it -v `pwd`:/wrk bbc-speech-segmenter /bin/bash

# Inside a Docker container

$ cd /wrk/

# Run test

$ ./test.sh
All checks passed
```

## Training and evaluation

### X-vector utility

`xvector_utils.py` can be used to train and evaluate x-vector classifier, as
well as o extract and visualize x-vectors. For more detailed information, see
[`XVECTOR_UTILS.md`](https://github.com/bbc/bbc-speech-segmenter/blob/main/XVECTOR_UTILS.md).

The documentation also gives [details on file formats](https://github.com/bbc/bbc-speech-segmenter/blob/main/XVECTOR_UTILS.md#input-files)
such as ARK, SCP or STM, which are required to use this tool.

### Run x-vector VAD training

Two files are required for x-vector-vad training:

* Reference STM file
* X-vectors ARK file

For example, from inside the Docker container:

```
$ cd /wrk/recipe

$ python3 local/xvector_utils.py train \
  data/bbc-vad-train/reference.stm     \
  data/bbc-vad-train/xvectors.ark      \
  new_model.pkl
```

The model will be saved as `new_model.pkl`.

### Run x-vector VAD evaluation

Three files are needed in order to run VAD evaluation:

* Reference STM file
* X-vectors ARK file
* x-vector-vad classifier model

For example, from inside the Docker container:

```
$ cd /wrk/recipe

$ python3 local/xvector_utils.py evaluate \
  data/bbc-vad-eval/reference.stm        \
  data/bbc-vad-eval/xvectors.ark         \
  model/xvector-classifier.pkl
```

### WebRTC baseline

The code for the baseline WebRTC system referenced in the paper is available in
the directory [`recipe/baselines/denoising_DIHARD18_webrtc`](https://github.com/bbc/bbc-speech-segmenter/tree/main/recipe/baselines/denoising_DIHARD18_webrtc).

## Request access to `bbc-vad-train` datasets

Due to size restriction, only [`bbc-vad-eval`](https://github.com/bbc/bbc-speech-segmenter/tree/main/recipe/data/bbc-vad-eval) is included in the repository. If you'd like access to `bbc-vad-train`, please contact [Matt Haynes](mailto:matt.haynes@bbc.co.uk?subject=[xvector-vad-for-stt]%20Request%20Access%20to%20Datasets).

## Authors

* Misa Ogura <misa.ogura01@gmail.com>
* Matt Haynes <matt.haynes@bbc.co.uk>
