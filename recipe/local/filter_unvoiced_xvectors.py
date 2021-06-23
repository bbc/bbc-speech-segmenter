#!/usr/bin/env python3

# Copyright  2021 Matt Haynes <matt.haynes@bbc.co.uk>
#            2021 Misa Ogura <misa.ogura01@gmail.com>
# Apache 2.0

# Takes a list of predictions for each xvector and a kaldi segments file.
# Filters xvectors based on their voiced probability.

import sys

threshold = float(sys.argv[1])
pred_file = sys.argv[2]
segm_file = sys.argv[3]

# Load predictions

xvector_predictions = {}

with open(pred_file) as pred_file:
    for line in pred_file:

        parts = line.strip().split(' ')

        # Ignore Header

        if parts[0] == 'xvector_key':
            continue

        # Parse prediction line, assume speech is col 3

        xvector_key = parts[0]
        speech_prob = float(parts[2])

        if speech_prob >= threshold:
            xvector_predictions[xvector_key] = True
        else:
            xvector_predictions[xvector_key] = False

# Print filtered segments

segments = []

with open(segm_file) as segm_file:
    for line in segm_file:
        parts = line.strip().split(' ')

        xvector_key = parts[0]

        if xvector_predictions[xvector_key] is True:
            print(line.strip())
