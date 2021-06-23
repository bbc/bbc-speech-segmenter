#!/usr/bin/env python3

# Copyright  2021 Matt Haynes <matt.haynes@bbc.co.uk>
#            2021 Misa Ogura <misa.ogura01@gmail.com>
# Apache 2.0

# The xvector system seems to over segment in places, forming clusters of a
# single xvector in length. These are not that useful for STT and should be
# merged into the biggest adjoining segment.

import sys

from collections import deque

# -----------------------------------------------------------------------------
# Parse RTTM

clusters = deque([])

with open(sys.argv[1]) as rttm:
    for line in rttm:
        line = ' '.join(line.split())
        clusters.append(line.strip().split(' '))

# -----------------------------------------------------------------------------
# Fix over segmentation

new_clusters = []

while len(clusters) > 0:

    cluster = clusters.popleft()

    # Cluster bigger than 1 xvector, keep it

    if float(cluster[4]) > 0.75:
        new_clusters.append(cluster)
        continue

    # Single xvector cluster ...

    # First consume any following single xvector clsuters

    while len(clusters) > 0:

        if float(clusters[0][4]) > 0.75:
            break
        else:
            cluster[4] = float(cluster[4]) + float(clusters[0][4])
            clusters.popleft()

    # Now find which of the left / right cluster to merge into

    if len(new_clusters) == 0:
        clusters[0][3] = float(cluster[3])
        clusters[0][4] = float(clusters[0][4]) + float(cluster[4])

    elif len(clusters) == 0:
        new_clusters[-1][4] = float(new_clusters[-1][4]) + float(cluster[4])

    elif float(new_clusters[-1][4]) >= float(clusters[0][4]):
        new_clusters[-1][4] = float(new_clusters[-1][4]) + float(cluster[4])

    else:

        clusters[0][3] = float(cluster[3])
        clusters[0][4] = float(clusters[0][4]) + float(cluster[4])


# -----------------------------------------------------------------------------
# Write new RTTM

for cluster in new_clusters:
    print(' '.join(map(str, cluster)))
