#!/usr/bin/env python3

# Copyright  2021 Matt Haynes <matt.haynes@bbc.co.uk>
#            2021 Misa Ogura <misa.ogura01@gmail.com>
# Apache 2.0
#
# Adds 1 xvector worth of padding to each segment.

import sys

# Read RTTM

utts = []

with open(sys.argv[1]) as f:
    for line in f:

        line = " ".join(line.split())

        parts = line.strip().split(' ')

        utt = {
            'type': parts[0],
            'file': parts[1],
            'chnl': parts[2],
            'tbeg': float(parts[3]),
            'tdur': float(parts[4]),
            'ortho': parts[5],
            'stype': parts[6],
            'name': parts[7],
            'conf': parts[8],
            'Slat': parts[9],
        }

        utts.append(utt)

# Pad each utt

pad_size = 0.75  # 1 xvector padding

for (idx, utt) in enumerate(utts):

    # Get last utt

    if idx == 0:
        last_utt = None
    else:
        last_utt = utts[idx - 1]

    # Get next utt

    if idx < len(utts) - 1:
        next_utt = utts[idx + 1]
    else:
        next_utt = None

    # Pad start ...

    if last_utt is None or last_utt['file'] != utt['file']:

        # First utt in programme - pad start to 0 or pad size

        utt['tbeg'] = max(0, utt['tbeg'] - pad_size)

    else:
        utt['tbeg'] = utt['tbeg'] - pad_size

    # Pad end ....

    if next_utt is None or next_utt['file'] != utt['file']:

        # Last utt in programme - ignore - TODO need to know duration of wav

        pass

    else:

        utt['tdur'] = utt['tdur'] + pad_size


# Find overlapping utts and fix

for (idx, utt) in enumerate(utts):

    # Get next utt

    if idx < len(utts) - 1:
        next_utt = utts[idx + 1]
    else:
        continue

    if next_utt['file'] == utt['file']:
        if utt['tbeg'] + utt['tdur'] > next_utt['tbeg']:
            avg = (utt['tbeg'] + utt['tdur'] + next_utt['tbeg']) / 2
            utt['tdur'] = avg - utt['tbeg']
            next_utt['tbeg'] = avg

# Merge adjoining utts of same speaker

merged_utts = []

for (idx, utt) in enumerate(utts):

    # Get next utt

    if idx < len(utts) - 1:
        next_utt = utts[idx + 1]
    else:
        merged_utts.append(utt)
        continue

    # Last utt in programme

    if next_utt['file'] != utt['file']:
        merged_utts.append(utt)
        continue

    # Different speaker

    if next_utt['name'] != utt['name']:
        merged_utts.append(utt)
        continue

    # Is / isn't adjoining?

    if utt['tbeg'] + utt['tdur'] == next_utt['tbeg']:
        next_utt['tbeg'] = utt['tbeg']
        next_utt['tdur'] += utt['tdur']
    else:
        merged_utts.append(utt)
        continue

# Print again

for utt in merged_utts:

    print('%s %s %s %0.3f %0.3f %s %s %s %s %s' % (
        utt['type'], utt['file'], utt['chnl'],
        utt['tbeg'], utt['tdur'], utt['ortho'],
        utt['stype'], utt['name'], utt['conf'],
        utt['Slat']
    ))
