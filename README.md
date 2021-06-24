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
$ docker run -it bbcrd/bbc-speech-segmenter /bin/bash
$ ./run-segmentation.sh --help
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
```

In order to run the segmentation script you need your audio in **16Khz Mono WAV**
format. You also need an STM file describing the segments you want to apply
voice activity detection and speaker diarization to.

```
# Convert audio file to 16Khz mono wav

$ ffmpeg audio.mp3 -vn -ac 1 -ar 16000 audio.wav

# Create STM file for input (here we assume audio is 60 seconds long), the
# format is "$fname 0 $fname $start_secs $end_secs <$label> _"

$ cat audio.stm
audio 0 audio 0.0 60.0 <label> _

# Run VAD and Diarization, results are in output-dir/diarization.stm

$ ./run-segmentation.sh audio.wav audio.stm output-dir
ls output-dir/diarization.stm
```

## Developers

### Build Docker image

```terminal
$ docker build -t bbc-speech-segmenter .
```

### Use Docker image to run code in local checkout

```terminal
$ docker run -it -v `pwd`:/wrk bbc-speech-segmenter /bin/bash
$ cd /wrk/
$ ./test.sh
All checks passed
```

### Run x-vector VAD training

Two files are required for x-vector-vad training:

* Reference STM file
* X-vectors ARK file

For example, from inside the Docker container:

```termiinal
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

```termiinal
$ cd /wrk/recipe

$ python3 local/xvector_utils.py evaluate \
  data/bbc-vad-eval/reference.stm        \
  data/bbc-vad-eval/xvectors.ark         \
  model/xvector-classifier.pkl
```

### X-vector utility

`xvector_utils.py` can be also used to extract and visualize x-vectors. For
more detailed information, see `XVECTOR_UTILS.md`.

### WebRTC baseline

The code for the baseline WebRTC system referenced in the paper is available in
the directory `recipe/baselines/denoising_DIHARD18_webrtc`.

## Request access to x-vector datasets

Due to size restriction, only `bbc-vad-eval` is included in the repository under `recipe/data`.

If you'd like access to `bbe-vad-train` or other datasets mentioned in the paper,
please contact [Matt Haynes](mailto:matt.haynes@bbc.co.uk?subject=[xvector-vad-for-stt]%20Request%20Access%20to%20Datasets).

## Authors

* Misa Ogura <misa.ogura01@gmail.com>
* Matt Haynes <matt.haynes@bbc.co.uk>
