# WebRTC baseline VAD

This code was originally published as part of the second [DIHARD Speech
Diarization Challenge](https://dihardchallenge.github.io/dihard2/). The
original repo is here: https://github.com/staplesinLA/denoising_DIHARD18 and
all original development was completed by @staplesinLA, @nryant and @mmmaat.

## Running the baseline system

### Setup docker image:
```
$ docker build -t baseline-webrtc-vad .
$ docker run -it baseline-webrtc-vad /bin/bash
```

### Run Webrtc VAD

```
$ python main_get_vad.py  \
    --wav_dir $wav_dir    \
    --output_dir $stm_dir \
    --speech_label speech \
    --med_filt_width 5    \
    --mode 0
```

### Merge close segments and format

```
for wav in $wav_dir/*.wav; do

  name=$(basename $wav .wav)

  stm=$stm_dir/$name.stm

  python merge.py $stm_dir/$name.sad                                        \
    | awk -v name=$name '{print name, 0, name "_spk", $1, $2, "<speech>" }' \
    > $stm

done
```

### Use WebRTC VAD STM file as VAD component in x-vector segmenter

```
for wav in $wav_dir/*.wav; do

  name=$(basename $wav .wav)

  stm=$stm_dir/$name.stm

  ./run-segmentation.sh --no-filter-unvoiced true $wav $stm $out_dir/$name \
    > $out_dir/$name.log 2>&1

done
```
