# X-vector utility

This is a quick guide for using x-vector utility tool (`recipe/local/xvector_utils.py`).

## System requirements

- Kaldi (for extracting x-vectors)
- Python 3.5+

For Python package dependencies, see `requirements.txt`.

## Usage

```terminal
$ python xvector_utils.py -h

Usage:
  xvector_utils.py [options] make-xvectors <wav.scp> <labels.stm> <xvectors.ark>
  xvector_utils.py [options] visualize <labels.stm> <xvectors.ark> <output_dir>
  xvector_utils.py [options] train <labels.stm> <xvectors.ark> <model.pkl>
  xvector_utils.py [options] tune <labels.stm> <xvectors.ark> <model.pkl>
  xvector_utils.py [options] evaluate <labels.stm> <xvectors.ark> <model.pkl>
  xvector_utils.py [options] predict <xvectors.ark> <model.pkl>
  xvector_utils.py [-h]

A utility tool to:

- extract x-vectors
- visualize x-vectors
- train, tune and evaluate an x-vector classifier

Visualize options:
  --interactive         If present, save plot as an interactive html file.
  --plot-all            If present, plot all data points. --sample-size option
                        will be ingored.
  --cmap STR            Specity matplotlib colormap for static plots, ignored
                        when generating interactive plots with --interactive.
                        Qualitative colormaps are recommended. [default: tab20]
  --num-components INT  Dimention of embedded space using t-SNE. [default: 2]
  --perc-variance FLOAT Fraction of the total variance that needs to be
                        explained by the generated components using PCA.
                        [default: 0.95]
  --sample-size INT     Sample size. [default: 10_000]
  --title STR           Custom title of the plot.

Train options:
  --params STR          String of LinearSVC parameters (see scikitlearn doc
                        for list of params) to train classifier on. e.g.,
                        '{"C": 0.001, "loss": "hinge"}'
                        [default: {"C": 0.01, "dual": False}]

Tune options:
  --param-grid STR      String of LinearSVC parameters (see scikitlearn doc
                        for list of params) to perform grid search on. e.g.,
                        '{"C": [0.1, 1], "loss": ["hinge", "squared_hinge"]}'
                        [default: {"C": [0.001, 0.01, 0.1, 1, 10, 100]}]
  --score STR           Scoring parameter (see scikitlearn documentation for
                        list of scoring params). [default: accuracy]
  --test-size FLOAT     Fraction of test sample size. [default: 0.20]

Evaluate options:
  --print-error-keys    Print xvector key names for classification errors.
  --save-roc PATH       Path to output CSV file containing ROC metrics.
                        Output CSV headers:
                            - fpr: False positive rate
                            - tpr: True positive rate
                            - thr: Probability thresholds
  --thr FLOAT           Probability threshold for binary classification.

Common options:
  -h --help             Show this screen.
  --version             Show version.
  --num-cpus INT        Max number of cpus to use. If not specified it uses all
                        processors.

Common arguments:
  <wav.scp> PATH        SCP file mapping from audio file IDs to absolute paths.
  <labels.stm> PATH     STM file containing class lablel(s) in the 6th column.
  <xvectors.ark> PATH   ARK file to read / write xvectors to.
  <model.pkl> PATH      Pickle file to read / write model to.
  <output_dir> PATH     Path to output directory.
```

## Input files

### Audio file format

The system expects audio files in 16KHz 16-bit mono wav format - use tools such as `ffmpeg` if your audio files don't conformt to this specification.

e.g.:

```terminal
$ ffmpeg -i input.mp3 -vn -ac 1 -ar 16000 output.wav
```

### `xvectors.ark`

It's a binary Kaldi ARK file containing 512-dimensional vectors (x-vectors) extracted from audio files. X-vectors can be extracted from audio files using the `xvector_utils.py make-xvectors` command and by providing appropriate `wav.scp` and `labels.stm`.

### `wav.scp`

A Kaldi style `wav.scp` file. It is a text file which provides mapping between file ids and the absolute path
of corresponding audio files. The file IDs should be the basename of the wav file minus the extension.

e.g.:

```text
audio_file_1 absolute/path/to/audio_file_1.wav
audio_file_2 absolute/path/to/audio_file_2.wav
audio_file_3 absolute/path/to/audio_file_3.wav
```

### `labels.stm`

The STM file is defined as part of the SCLITE toolkit and can be found [here](http://my.fit.edu/~vkepuska/ece5527/sctk-2.3-rc1/doc/infmts.htm).

Definition is as follows:

```text
STM :== <F> <C> <S> <BT> <ET> [ <LABEL> ] transcript . . .

    Where:

    <F>         The waveform filename. NOTE: no pathnames or extensions are expected.
    <C>         The waveform channel. This value is ignored.
    <S>         The speaker id, no restrictions apply to this name.
    <BT>        The begin time (seconds) of the segment.
    <ET>        The end time (seconds) of the segment.
    <LABEL>     A comma separated list of subset identifiers enclosed in angle brackets.
    transcript  (Optional) A whitespace separated list of words or _ to denote empty transcripts.
```

For this tool the transcript will be ignored.

Use 6th column to provide lables for each segment.

e.g.:

```text
audio_file_1 0 audio_file_1_Spk1 0.0 37.1779439252 <noise> _
audio_file_1 0 audio_file_1_Spk1 37.1779439252 39.5 <speech> EXSAMPLE TRANSCRIPT ONE
audio_file_1 0 audio_file_1_Spk1 39.5 41.31 <speech> EXSAMPLE TRANSCRIPT TWO
audio_file_1 0 audio_file_1_Spk1 41.31 47.38 <noise> _
audio_file_1 0 audio_file_1_Spk1 47.38 53.5610280374 <speech> _
```

If a segment belongs to more than one class, separate each class with comma.

e.g.:

```text
audio_file_1 0 audio_file_1_Spk1 0.0 37.1779439252 <noise,laughter> _
audio_file_1 0 audio_file_1_Spk1 37.1779439252 39.5 <speech> EXSAMPLE TRANSCRIPT ONE
audio_file_1 0 audio_file_1_Spk1 39.5 41.31 <speech,applause> EXSAMPLE TRANSCRIPT TWO
audio_file_1 0 audio_file_1_Spk1 41.31 47.38 <noise,music> _
audio_file_1 0 audio_file_1_Spk1 47.38 53.5610280374 <speech> _
```
