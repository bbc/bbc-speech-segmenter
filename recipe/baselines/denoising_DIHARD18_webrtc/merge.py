import sys

# Read input segments

segments = []

with open(sys.argv[1]) as seg_file:
    for line in seg_file:
        line = line.strip().split(' ')
        tbeg = float(line[0])
        tend = float(line[1])

        segments.append([tbeg, tend])

# Merge any segments with XXXms

dist = 0.5

merged_segments = []

for segment in segments:

    if len(merged_segments) == 0:
        merged_segments.append(segment)
        continue

    last_tbeg, last_tend = merged_segments[-1]
    this_tbeg, this_tend = segment

    if this_tbeg - last_tend <= dist:
        merged_segments[-1][1] = this_tend
    else:
        merged_segments.append(segment)

# Print new segments

for (tbeg, tend) in merged_segments:
    print("%0.3f %0.3f speech" % (tbeg, tend))
