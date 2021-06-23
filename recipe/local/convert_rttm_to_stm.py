#!/usr/bin/env python3

# Copyright  2021 Matt Haynes <matt.haynes@bbc.co.uk>
#            2021 Misa Ogura <misa.ogura01@gmail.com>
# Apache 2.0
#
# This file takes a file in RTTM format and writes to a new file in STM format

import sys


def error_print(s):
    sys.stderr.write(s)
    sys.stderr.write("\n")
    sys.stderr.flush()


def print_usage():
    error_print(
        "Converts an .rttm file as produced by the X-Vectors diarization")

    error_print("tool into a format readable by bbc-speech-identifier.")

    error_print("Usage: %s rttm_file_in stm_file_out" % sys.argv[0])


if __name__ == "__main__":
    try:
        filename_in = sys.argv[1]
        filename_out = sys.argv[2]
    except IndexError:
        print_usage()
        sys.exit(0)

    with open(filename_in, "r", encoding="utf-8") as fin:
        with open(filename_out, "w", encoding="utf-8") as fout:
            for line in fin:
                in_fields = line.strip().split()
                ts_start = float(in_fields[3])
                duration = float(in_fields[4])
                ts_end = ts_start + duration

                sortable_speaker_id = "%s_S%05d" % (
                    in_fields[1], int(in_fields[7]))

                out_fields = [
                    in_fields[1],
                    "0",
                    sortable_speaker_id,
                    "%.3f" % ts_start,
                    "%.3f" % ts_end,
                    "<speech>"
                ]

                fout.write(" ".join(out_fields) + "\n")
