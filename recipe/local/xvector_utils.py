#!/usr/bin/env python3

# Copyright  2021 Matt Haynes <matt.haynes@bbc.co.uk>
#            2021 Misa Ogura <misa.ogura01@gmail.com>
# Apache 2.0

"""
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
  --sample-size INT     Sample size. [default: 10000]
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
"""  # noqa
import ast
import csv
import logging
import os
import pickle
import random
import shutil
import subprocess
import sys
import tempfile

from pathlib import Path
from pprint import pformat

import docopt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from kaldi_io import read_vec_flt_ark
from matplotlib.cm import get_cmap

from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import \
    classification_report, \
    confusion_matrix, \
    roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC

from stm import Stm


# -----------------------------------------------------------------------------
# XvectorClassifier class

class XvectorClassifier():

    @classmethod
    def load(klass, model_path):
        with open(model_path, 'rb') as model_pickle:
            model, class_names = pickle.load(model_pickle)

        return XvectorClassifier(model, class_names)

    def __init__(self, model, class_names):
        self._model = model
        self._class_names = class_names

    @property
    def model(self):
        return self._model

    @property
    def class_names(self):
        return self._class_names

    def save(self, model_path):
        with open(model_path, 'wb') as model_file:
            pickle.dump((self._model, self._class_names), model_file)

    def train(self, X_train, y_train):
        self._model.fit(X_train, y_train)

    def tune(self, X, y, test_size, split_random_state,
             param_grid, score, num_cpus):

        X_train, X_test, y_train, y_test = \
            train_test_split(X, y,
                             test_size=test_size,
                             random_state=split_random_state)

        print('Train samples: {}, test samples: {}'
              .format(len(y_train), len(y_test)))

        gs = GridSearchCV(init_model(),
                          param_grid=param_grid,
                          scoring=score,
                          cv=5,
                          n_jobs=num_cpus)

        gs.fit(X_train, y_train)
        best_params = gs.best_params_

        print('Best parameters set found on development set:')
        print(best_params)

        self._model = gs.best_estimator_

        self.evaluate(X_test, y_test)

    def evaluate(self, X_test, y_test, thr=None, roc_csv=None):
        if thr:
            print('\nEvaluating classifier with thr={}'.format(thr))

        else:
            print('\nNo threshold set - predicted class'
                  ' is the class with the highest probability')

        y_true, y_pred = y_test, self.predict(X_test, thr=thr)

        print('\n=== Classification report ===\n')
        print(classification_report(y_true,
                                    y_pred,
                                    labels=list(range(len(self._class_names))),
                                    target_names=self._class_names))

        # Calc confusion matrix only if there's more than one class in data

        if len(np.unique(y_true)) != 1:

            confusion_scores = confusion_matrix(y_test, y_pred)

            print('=== Confusion matrix ===\n')
            print(pd.DataFrame(
                confusion_scores,
                columns=['pred_neg', 'pred_pos'],
                index=['neg', 'pos']
            ))

            tn, fp, fn, tp = confusion_scores.ravel()

            fpr = fp / (tn + fp)
            tpr = tp / (tp + fn)
            ppv = tp / (tp + fp)
            acc = (tp + tn) / (tn + fp + fn + tp)

            print('\n=== Summary metrics ===\n')
            print('False positive rate: {:.4f}'.format(fpr))
            print('True positive rate (recall): {:.3f}'.format(tpr))
            print('Positive predictive value (precision): {:.3f}'.format(ppv))
            print('Accuracy: {:.3f}'.format(acc))

        if roc_csv:
            roc_csv = os.path.abspath(roc_csv)

            y_probs = self.predict_proba(X_test)[:, 1]
            fprs, tprs, thresholds = roc_curve(y_true, y_probs)

            with open(roc_csv, 'w') as f:
                header = ['fpr', 'tpr', 'thr']
                writer = csv.writer(f)
                writer.writerow(header)

                for (fpr, tpr, thr) in zip(fprs, tprs, thresholds):
                    writer.writerow([fpr, tpr, thr])

            print('\nSaved ROC metrics as {}'.format(roc_csv))

        return y_true, y_pred

    def predict_proba(self, data):
        return self._model.predict_proba(data)

    def predict(self, data, thr=None):
        if thr:
            y_scores = self.predict_proba(data)[:, 1]
            return np.where(y_scores > thr, 1, 0)

        else:
            return self._model.predict(data)


# -----------------------------------------------------------------------------
# Utility functions

def init_model(params={}):
    return CalibratedClassifierCV(
        Pipeline([
            ('scaler', Normalizer(norm='l1')),
            ('svc', LinearSVC(**params))
        ])
    )


def plot_static(data, labels, class_names, outputd_dir, dim=2, title=None,
                cmap='tab20'):

    if not title:
        title = 't-SNE visualization of x-vector embeddings'

    if len(class_names) <= 2:
        colors = ['k', 'silver']
    else:
        cmap = get_cmap(cmap)
        colors = [cmap(v) for v in np.linspace(0, 1, len(class_names))]

    if dim == 2:
        for i, (class_name, color) in enumerate(zip(class_names, colors)):
            data_by_class = data[labels == i]

            plt.scatter(list(data_by_class[:, 0]),
                        list(data_by_class[:, 1]),
                        label=class_name,
                        c=np.array([color]),
                        s=1)

            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')

    else:
        for i, class_name in enumerate(class_names):
            data_by_class = data[labels == i]

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            ax.scatter(list(data_by_class[:, 0]),
                       list(data_by_class[:, 1]),
                       list(data_by_class[:, 2]),
                       label=class_name,
                       s=1)

            ax.set_xlabel('Dimension 1')
            ax.set_ylabel('Dimension 2')
            ax.set_zlabel('Dimension 3')

    plt.legend()
    plt.title(title)

    plt.savefig(os.path.join(outputd_dir, 'visualization.png'),
                bbox_inches='tight',
                dpi=600)


def plot_interactive(data, labels, class_names, outputd_dir, dim=2,
                     annotations=None, title=None):

    if not title:
        title = 't-SNE visualization of x-vector embeddings'

    traces = []

    if dim == 2:
        for i, class_name in enumerate(class_names):
            data_by_class = data[labels == i]

            trace = go.Scatter(x=list(data_by_class[:, 0]),
                               y=list(data_by_class[:, 1]),
                               mode='markers',
                               name=class_name,
                               text=annotations,
                               marker={'size': 2, 'opacity': 0.8})

            traces.append(trace)
    else:
        for i, class_name in enumerate(class_names):
            data_by_class = data[labels == i]

            trace = go.Scatter3d(x=list(data_by_class[:, 0]),
                                 y=list(data_by_class[:, 1]),
                                 z=list(data_by_class[:, 2]),
                                 mode='markers',
                                 name=class_name,
                                 marker={'size': 2, 'opacity': 0.8})

            traces.append(trace)

    fig = go.Figure(data=traces)
    fig.update_layout(title=title, title_x=0.4)

    fig.write_html(os.path.join(outputd_dir, 'visualisation.html'))


def load_xvectors(logger, xvectors_ark):

    xvectors = []
    xvector_keys = []

    # Read ark

    for key, xvector in read_vec_flt_ark(xvectors_ark):
        xvectors.append(xvector)
        xvector_keys.append(key)

    xvectors = np.asarray(xvectors, dtype=np.float32)

    logger.info('Loaded %d xvectors' % len(xvectors))

    return xvectors, xvector_keys


def load_xvectors_and_labels(logger, labels_stm, xvectors_ark):

    labels = []

    # Get a look up from xvector.ark keys -> labels.stm <labels>

    xvector_key_to_labels = {}

    stm = Stm.load(labels_stm)

    for line in stm.lines():
        key = '%s_%010.03f_%010.03f' % (
            line.filename, line.begin_time, line.end_time
        )

        xvector_key_to_labels[key] = line.labels

    # Load xvectors

    xvectors, xvector_keys = load_xvectors(logger, xvectors_ark)

    for xvector, key in zip(xvectors, xvector_keys):

        parts = key.split('-')

        stm_key = '-'.join(parts[0:-2])

        for label in xvector_key_to_labels[stm_key]:
            labels.append(label)

    # Get label names -> index lookup

    labels_index = sorted(list(set(labels)))

    logger.info(
        'Found %d labels: %s' % (len(labels_index), ','.join(labels_index)))

    # Return data as np arrays

    labels = [labels_index.index(label) for label in labels]

    labels = np.asarray(labels, dtype=np.int)

    return xvectors, labels, labels_index, xvector_keys


def make_xvectors(logger, wav_scp, labels_stm, output_ark, num_cpus=-1):

    if os.path.exists(output_ark):
        logger.fatal('Output ark %s exists, not overwriting' % output_ark)
        exit(1)

    logger.info('Extracting x-vectors')

    xvector_opts = ''

    if num_cpus >= 1:
        xvector_opts += ' --nj {} '.format(num_cpus)

    # Run xvector extraction

    recipe_dir = Path(__file__).parents[1].absolute()

    tmp_dir = tempfile.TemporaryDirectory()

    cmd = '{} {} {} {} {}'.format('./make-xvectors.sh',
                                  xvector_opts,
                                  os.path.abspath(wav_scp),
                                  os.path.abspath(labels_stm),
                                  tmp_dir.name)

    process = subprocess.run('cd {} && {}'.format(recipe_dir, cmd),
                             shell=True)

    if process.returncode != 0:
        logger.fatal('Error extracting x-vectors, see ' + tmp_dir.name)
        sys.exit(1)

    src_ark = os.path.join(tmp_dir.name, 'xvectors', 'xvector.all.ark')

    # Load all xvectors and classes (just to test it looks good)

    xvectors, _, _, _ = load_xvectors_and_labels(logger, labels_stm, src_ark)

    logger.info('Generated %d xvectors' % len(xvectors))

    # Write ark to destination

    shutil.move(src_ark, output_ark)

    logger.info('X-vectors saved as {}'.format(output_ark))

    tmp_dir.cleanup()


def extract_timestamp(xvector_key):
    '''
    X-vector key format: fileid_segStart_segEnd-xvectorStart-xvectorEnd
        segStart & segEng in seconds
        xvectorStart & xvectorEnd in milliseconds
    '''

    # Split twice from right as fileid might contain underscore too
    fileid, seg_start, rest = xvector_key.rsplit('_', 2)

    # Extract X-vector timimng from the rest
    xvector_start_ms, xvector_end_ms = rest.split('-')[-2:]

    # Calculate X-vector duration in seconds
    xvector_start = int(xvector_start_ms) / 1000  # ms to sec
    xvector_end = int(xvector_end_ms) / 1000  # ms to sec
    xvector_duration = xvector_end - xvector_start  # duration in sec

    start = float(seg_start) + xvector_start
    end = start + xvector_duration

    return {'fileid': fileid,
            'start': '{:.2f}'.format(start),
            'end': '{:.2f}'.format(end)}


# -----------------------------------------------------------------------------
# Main

def main():

    # -------------------------------------------------------------------------
    # Set up

    # Create logger

    logger = logging.getLogger('xvector_utils.main')

    console = logging.StreamHandler()

    logging.basicConfig(level=logging.INFO, handlers=[console])

    # Parse args

    arguments = docopt.docopt(__doc__, version='0.0.1')

    if arguments['--num-cpus']:
        num_cpus = int(arguments['--num-cpus'])
        logger.info('Using {} cpus where parallelisation is possible'
                    .format(num_cpus))
    else:
        num_cpus = -1
        logger.info('Using all cpus where parallelisation is possible')

    # -------------------------------------------------------------------------
    # Validate STM

    if arguments['<labels.stm>'] is not None:

        logger.info('Validating stm file format')

        label_stm = os.path.abspath(arguments['<labels.stm>'])

        if not Stm.is_valid(label_stm):
            logger.fatal('Invalid stm file: {}'.format(label_stm))
            sys.exit(1)

    # -------------------------------------------------------------------------
    # Make xvectors API

    if arguments['make-xvectors']:

        make_xvectors(logger,
                      arguments['<wav.scp>'],
                      arguments['<labels.stm>'],
                      arguments['<xvectors.ark>'],
                      num_cpus)

    # -------------------------------------------------------------------------
    # Visualize API

    if arguments['visualize']:

        output_dir = os.path.abspath(arguments['<output_dir>'])

        if os.path.exists(output_dir) and os.path.isdir(output_dir):
            if os.listdir(output_dir):
                logger.fatal('Output directory not empty')
                sys.exit(1)
        else:
            os.makedirs(output_dir)

        X, y, class_names, xvector_keys = load_xvectors_and_labels(
            logger, arguments['<labels.stm>'], arguments['<xvectors.ark>'])

        if arguments['--plot-all']:
            logger.info('Plotting all data points')
        else:
            sample_size = min(len(y), int(arguments['--sample-size']))

            logger.info('Sampled {} data points'.format(sample_size))

            sample = random.sample(list(zip(X, y, xvector_keys)), sample_size)
            X, y, xvector_keys = zip(*sample)
            X, y = np.asarray(X), np.asarray(y)

        pca_params = {'n_components': float(arguments['--perc-variance']),
                      'svd_solver': 'full'}

        logger.info('Performing PCA with params: {}'
                    .format(pformat(pca_params)))

        pca = PCA(**pca_params)
        X_pca = pca.fit_transform(X)

        logger.info('Features reduced to {}'.format(X_pca.shape[1]))

        vis_params = {
            'n_components': int(arguments['--num-components']),
            'perplexity': 50.0
        }

        logger.info('Performing t-SNE with params: {}'
                    .format(pformat(vis_params)))

        reducer = TSNE(**vis_params)

        data = reducer.fit_transform(X_pca)

        timestamps = list(map(extract_timestamp, xvector_keys))

        logger.info('Saving visualization to {}'.format(output_dir))

        if arguments['--interactive']:
            plot_interactive(data,
                             y,
                             class_names,
                             output_dir,
                             dim=vis_params['n_components'],
                             annotations=timestamps,
                             title=arguments['--title'])

        else:
            plot_static(data,
                        y,
                        class_names,
                        output_dir,
                        dim=vis_params['n_components'],
                        title=arguments['--title'],
                        cmap=arguments['--cmap'])

        dataset_path = os.path.join(output_dir, 'dataset_reduced.pkl')

        logger.info('Saving the dataset with reduced features in {}'
                    .format(dataset_path))

        dataset = (X_pca, data, y)

        metadata = {'pca_params': pca_params,
                    'vis_params': vis_params,
                    'class_names': class_names}

        with open(dataset_path, 'wb') as dataset_file:
            pickle.dump((dataset, metadata), dataset_file)

        logger.info('Unpickle to retrieve: ((X_pca, X_tsne, y), metadata)')

    # -------------------------------------------------------------------------
    # Train API

    if arguments['train']:
        params = ast.literal_eval(arguments['--params'])

        logger.info('Training classifier with params {}'
                    .format(params))

        logger.info('Other parameters are set to default')

        X, y, class_names, _ = load_xvectors_and_labels(
            logger, arguments['<labels.stm>'], arguments['<xvectors.ark>'])

        clf = XvectorClassifier(init_model(params), class_names)

        clf.train(X, y)

        model_path = arguments['<model.pkl>']

        logger.info('Saving the model to {}'.format(model_path))

        clf.save(model_path)

    # -------------------------------------------------------------------------
    # Evaluate API

    if arguments['evaluate']:
        model_path = arguments['<model.pkl>']

        clf = XvectorClassifier.load(model_path)

        X, y, class_names, xvector_keys = load_xvectors_and_labels(
            logger, arguments['<labels.stm>'], arguments['<xvectors.ark>'])

        # Ensure y matches class name index in classifier

        for i, v in enumerate(y):
            y[i] = clf.class_names.index(class_names[v])

        logger.info('Evalating the model')

        if arguments['--thr']:
            thr = float(arguments['--thr'])
        else:
            thr = None

        roc_csv = arguments['--save-roc']

        y_true, y_pred = clf.evaluate(X, y, thr=thr, roc_csv=roc_csv)

        if arguments['--print-error-keys']:
            for i in range(len(y_true)):
                if y_true[i] != y_pred[i]:
                    logger.info(
                        'Incorrect classification for %s' % xvector_keys[i])

    # -------------------------------------------------------------------------
    # Tune API

    if arguments['tune']:
        test_size = float(arguments['--test-size'])
        logger.info('Splitting data with test size {}'.format(test_size))

        split_random_state = 42

        param_grid = ast.literal_eval(arguments['--param-grid'])
        score = arguments['--score']

        logger.info('Performing grid search with params {}, score {}'
                    .format(pformat(param_grid), score))
        logger.info('Other parameters are set to default')

        X, y, class_names, _ = load_xvectors_and_labels(
            logger, arguments['<labels.stm>'], arguments['<xvectors.ark>'])

        clf = XvectorClassifier(LinearSVC(), class_names)

        clf.tune(X, y, test_size, split_random_state,
                 param_grid, score, num_cpus)

        model_path = arguments['<model.pkl>']

        logger.info('Saving the model to {}'.format(model_path))

        clf.save(model_path)

    # -------------------------------------------------------------------------
    # Predict API

    if arguments['predict']:
        model_path = arguments['<model.pkl>']

        clf = XvectorClassifier.load(model_path)

        xvectors, xvector_keys = load_xvectors(
            logger, arguments['<xvectors.ark>'])

        # Print header

        print('xvector_key', ' '.join(clf.class_names))

        # Print data

        for (i, pred) in enumerate(clf.predict_proba(xvectors)):
            print(xvector_keys[i], ' '.join(map(str, pred)))


if __name__ == '__main__':
    main()
