#!/usr/bin/env python3

# Copyright  2021 Matt Haynes <matt.haynes@bbc.co.uk>
#            2021 Misa Ogura <misa.ogura01@gmail.com>
# Apache 2.0
#
# Takes an RTTM file, xvectors.ark and classfier.pkl as input. Filters the RTTM
# segments based on the proportion of voiced frames.

import sys
import os
import re
import pickle

import numpy as np


def extract_segments(arktxt):
    segments = []

    with open(arktxt, encoding="utf-8") as ark:
        for line in ark:
            # Extract data fields
            _, time_info, data = list(
                re.match(
                    r"(.*)_(\d{6}.\d{3}_\d{6}.\d{3}-\d{8}-\d{8})  (.*)", line
                ).groups())

            # Get timestamps for vad segment & xvector segment
            vad_ts, seg_ts = time_info.split("-", 1)

            # Get start/end times in seconds
            vad_start, vad_end = map(float, vad_ts.split("_"))
            seg_start, seg_end = map(lambda x: float(x) / 100,
                                     seg_ts.split("-"))

            # Calculate absolute start/end time for the xvector segment
            start = (vad_start + seg_start)
            end = (vad_start + seg_end)

            # Extract xvector data
            data = re.sub(r"(\[|\])", "", data).strip()
            data = np.fromstring(data, dtype=float, sep=" ")

            segments.append((start, end, data))

    return segments


class SegmentsHandler:

    def __init__(self, xvectors, seg_every=0.75):
        self._xvectors = xvectors
        self._seg_every = seg_every

    def find_segments(self, start, end):
        segments = []

        for seg_start, seg_end, xvector in self._xvectors:
            if seg_start >= start and seg_end <= end:
                segments.append(xvector)

        return segments


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("usage: python3 filter-unvoiced-segments.py "
              "<threshold> <model-path> <rttm-dir> <arktxt>")
        sys.exit(1)

    thr = float(sys.argv[1])
    model_path = sys.argv[2]
    rttm_dir = sys.argv[3]
    arktxt = sys.argv[4]

    handler = SegmentsHandler(extract_segments(arktxt))

    with open(model_path, "rb") as f:
        clf, class_names = pickle.load(f)

    filtered_rttm = open(os.path.join(rttm_dir, "rttm.filt"), "w")

    with open(os.path.join(rttm_dir, "rttm")) as f:
        for line in f:
            parts = line.split()
            start = float(parts[3])
            end = start + float(parts[4])

            xvectors = handler.find_segments(start, end)

            if xvectors:
                decision = clf.predict(xvectors)

                if decision.mean() < thr:
                    continue
                else:
                    filtered_rttm.write(line)
            else:
                # Keep the line if there is no xvector found for the segment
                filtered_rttm.write(line)

    filtered_rttm.close()
