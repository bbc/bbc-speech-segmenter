#!/usr/bin/env python3

# Copyright  2021 Matt Haynes <matt.haynes@bbc.co.uk>
#            2021 Misa Ogura <misa.ogura01@gmail.com>
# Apache 2.0

from collections import defaultdict
from itertools import zip_longest

import filecmp
import math
import os
import random
import re
import subprocess
import sys
import tempfile

DEFAULT_RANDOM_SEED = 185670580184244466416740396116939082841


# -----------------------------------------------------------------------------
# Constants / Exceptions / Utility methods

STM_LINE_REGEX = \
    r'^(\S+)\s+(\S+)\s+(\S+)\s+([0-9\.]+)\s+([0-9\.]+)\s+(<\S+>)?[\s+]?(.*)$'


class StmLine():
    def __init__(self, filename, channel, speaker, begin_time, end_time,
                 labels, transcript):

        self.filename = filename
        self.channel = channel
        self.speaker = speaker
        self.begin_time = begin_time
        self.end_time = end_time
        self.labels = labels
        self.transcript = transcript
        self.duration = self.end_time - self.begin_time

    def __str__(self):

        parts = [
            self.filename,
            self.channel,
            self.speaker,
            '%0.2f' % self.begin_time,
            '%0.2f' % self.end_time
        ]

        if len(self.labels) > 0:
            parts.append('<' + ','.join(self.labels) + '>')

        parts.append(self.transcript)

        return ' '.join(parts)

    def __repr__(self):
        return str(self)


class InvalidStmError(Exception):
    pass


def parse_stm_line(stm_line, line_num='unknown'):

    match = re.search(STM_LINE_REGEX, stm_line.strip())

    if match is None:
        raise InvalidStmError(
            'stm is not valid, error on line: %s' % str(line_num))

    if match.group(6) is None:
        labels = []
    else:
        labels = match.group(6)
        labels = labels.strip().replace('<', '').replace('>', '').split(',')

    return StmLine(
        filename=match.group(1),
        channel=match.group(2),
        speaker=match.group(3),
        begin_time=float(match.group(4)),
        end_time=float(match.group(5)),
        labels=labels,
        transcript=match.group(7).strip()
    )


# -----------------------------------------------------------------------------
# Public interface

class Stm():

    # Class Methods

    @classmethod
    def is_valid(cls, stm_file):
        try:
            for _ in cls.read(stm_file):
                pass
        except InvalidStmError:
            return False

        return True

    @classmethod
    def is_sorted(cls, stm_file):

        with tempfile.NamedTemporaryFile(mode='w') as out:

            cmd = 'sort +0 -1 +1 -2 +3nb -4'
            cmd = "cat {} | {} > {}".format(stm_file, cmd, out.name)

            subprocess.run(cmd,
                           shell=True,
                           stdout=sys.stdout,
                           stderr=subprocess.PIPE)

            return filecmp.cmp(stm_file, out.name, shallow=False)

    @classmethod
    def read(cls, stm_file, exclude_ignore_segments=False):

        with open(stm_file, encoding='utf-8') as stm:

            for i, stm_line in enumerate(stm):

                # Ignore comment lines

                if stm_line.startswith(';;'):
                    continue

                # Parse each line

                stm_line = parse_stm_line(stm_line, line_num=i + 1)

                # Ignore if we should

                if stm_line.transcript == 'ignore_time_segment_in_scoring' \
                        and exclude_ignore_segments:
                    continue

                # Yield

                yield stm_line

    @classmethod
    def load(cls, stm_file, exclude_ignore_segments=False):
        stm_lines = list(cls.read(
            stm_file, exclude_ignore_segments=exclude_ignore_segments))

        return cls(stm_lines)

    # Instance methods

    def __init__(self, stm_lines):
        self._stm_lines = stm_lines

    def __len__(self):
        return len(self._stm_lines)

    def labels(self):
        labels = set([])

        for line in self.lines():
            for label in line.labels:
                labels.add(label)

        return list(labels)

    def write(self, stm_file):

        with tempfile.NamedTemporaryFile(mode='w') as tmp_stm:

            # Write main body

            for line in self.lines():
                tmp_stm.write(str(line) + '\n')

            tmp_stm.flush()

            cmd = 'sort +0 -1 +1 -2 +3nb -4'
            cmd = "cat {} | {} > {}".format(tmp_stm.name, cmd, stm_file)

            subprocess.run(cmd,
                           shell=True,
                           stdout=sys.stdout,
                           stderr=subprocess.PIPE)

            # Prepend labels header correctly

            labels = sorted(self.labels())

            if len(labels) > 0:
                with open(stm_file, encoding='utf-8') as f:
                    data = f.read()

                with open(stm_file, 'w', encoding='utf-8') as f:
                    f.write(';;\n')
                    f.write(';; CATEGORY "0" "" ""\n')
                    f.write(';; LABEL "O" "Overall" "Overall"\n')
                    f.write(';; CATEGORY "1" "Custom" ""\n')

                    for label in labels:
                        f.write(';; LABEL "%s" "%s" "%s"\n' %
                                (label, label, label))

                    f.write(data)

    def filter(self,
               min_segment_duration=-math.inf,
               max_segment_duration=math.inf):

        lines = self.lines()

        lines = [line for line in lines
                 if line.duration >= min_segment_duration]

        lines = [line for line in lines
                 if line.duration <= max_segment_duration]

        return Stm(lines)

    def total_duration(self):
        return sum([line.duration for line in self.lines()])

    def files(self):
        filenames = {}

        for line in self.lines():
            filenames[line.filename] = True

        return sorted(filenames.keys())

    def lines(self):
        return self._stm_lines

    def merge_consecutive_segments(self,
                                   max_segment_duration=math.inf,
                                   merge_boundary=0.0,
                                   merge_speakers=False):

        merged_lines = []

        for line in self.lines():

            if len(merged_lines) == 0:
                merged_lines.append(line)
                continue

            last_line = merged_lines[-1]

            same_file = line.filename == last_line.filename
            same_chan = line.channel == last_line.channel
            same_spkr = line.speaker == last_line.speaker

            bordering = line.begin_time - last_line.end_time <= merge_boundary

            good_size = \
                line.end_time - last_line.begin_time <= max_segment_duration

            can_merge = same_file and same_chan
            can_merge = can_merge and (same_spkr or merge_speakers)

            can_merge = can_merge and bordering
            can_merge = can_merge and good_size

            if can_merge:

                if last_line.speaker != line.speaker:

                    # Find common prefix, kaldi convention is that it's fname,
                    # try and keep it

                    prefix = os.path.commonprefix(
                        [last_line.speaker, line.speaker])

                    if len(prefix) > 3:
                        last_line.speaker = prefix + '_multiple_speakers'
                    else:
                        last_line.speaker = 'multiple_speakers'

                last_line.end_time = line.end_time

                last_line.labels = sorted(
                    set(last_line.labels).union(line.labels))

                last_line.transcript += ' ' + line.transcript
                last_line.transcript = last_line.transcript.strip()

            else:
                merged_lines.append(line)

        return Stm(merged_lines)

    def sample(self, total_duration=math.inf, random_seed=DEFAULT_RANDOM_SEED):
        lines = self.lines()

        random.seed(random_seed)

        random.shuffle(lines)

        fname_to_lines = defaultdict(list)

        for line in lines:
            fname_to_lines[line.filename].append(line)

        # Generate tuples containing one line from each programme, with
        # missing values filled with None if programmes run out of lines

        grouped_lines = zip_longest(*fname_to_lines.values())

        # Flatten and reverse so we can sample by popping

        grouped_lines = list(sum(grouped_lines, ()))
        grouped_lines = grouped_lines[::-1]

        sampled_lines = []

        sample_duration = 0

        while grouped_lines:

            sample = grouped_lines.pop()

            if sample:

                duration = sample.end_time - sample.begin_time
                good_size = (sample_duration + duration <= total_duration)

                if good_size:
                    sampled_lines.append(sample)
                    sample_duration += duration
                else:
                    break

        return Stm(sampled_lines)

    def add_labels(self, labels):
        if isinstance(labels, str):
            labels = [labels]

        lines = self.lines()

        for line in lines:
            for label in labels:
                line.labels.append(label)

        return Stm(lines)
