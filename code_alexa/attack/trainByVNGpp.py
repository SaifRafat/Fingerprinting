# =============================================================================
# trainByVNGpp.py — Feature extraction + training for VNG++ classifier
# 
# Computes time-series bursts and statistical features from traffic traces.
# Uses Gaussian Naive Bayes for classification.
# =============================================================================

import numpy as np
from sklearn.naive_bayes import GaussianNB

import fileUtils
import tools


def readfile(fpath):
    """Parse traffic CSV. Returns (up_pack_num, down_pack_num, up_total, down_total, time_list, tuple_list)."""
    up_stream_total   = 0
    down_stream_total = 0
    up_pack_num       = 0
    down_pack_num     = 0
    trace_time_list   = []
    tuple_list        = []

    for line in fileUtils.readTxtFile(fpath, ignore=','):
        parts = line.split(',')
        
        # Skip header row
        try:
            int(parts[0]) if parts[0].strip() else float(parts[1])
        except (ValueError, IndexError):
            continue

        # Extract packet info: [index, time, size, direction, ...]
        try:
            size = fileUtils.str2int(parts[2])
            direction = fileUtils.str2int(parts[3])
            timestamp = fileUtils.str2float(parts[1])
            pkt_tuple = (size, direction)
        except (IndexError, ValueError):
            # Skip malformed rows
            continue
        
        trace_time_list.append(timestamp)
        tuple_list.append(pkt_tuple)

        if direction == 1:
            up_stream_total += size
            up_pack_num     += 1
        elif direction == -1:
            down_stream_total += size
            down_pack_num     += 1
        else:
            raise ValueError(f'Unexpected direction flag: {direction}')

    return up_pack_num, down_pack_num, up_stream_total, down_stream_total, trace_time_list, tuple_list


def calculateBursts(tuple_list):
    """Group consecutive packets by direction; return burst sizes (summed bytes per direction flip)."""
    burst_list   = []
    current_dir  = None
    current_burst = 0

    for i, (size, direction) in enumerate(tuple_list):
        if i == 0:
            current_dir   = direction
            current_burst = size * direction
        elif direction != current_dir:
            burst_list.append(current_burst)
            current_dir   = direction
            current_burst = size * direction
        else:
            current_burst += size * direction

    burst_list.append(current_burst)   # flush last burst
    return burst_list


def computeFeature(fpath, range_start, range_end, interval):
    """Extract VNG++ features: total_time, up_total, down_total + burst histogram."""
    _, _, up_total, down_total, time_list, tuple_list = readfile(fpath)

    burst_list = calculateBursts(tuple_list)
    boundaries, counts = tools.getSectionList(range_start, range_end, interval)
    for burst in burst_list:
        idx        = tools.computeRange(boundaries, burst)
        counts[idx] += 1

    time_list.sort()
    total_trace_time = time_list[-1] - time_list[0]

    feature = [total_trace_time, up_total, down_total] + counts
    return feature


def train(train_data, train_labels):
    """Fit Gaussian Naive Bayes model. Returns trained model."""
    gnb = GaussianNB()
    gnb.fit(train_data, train_labels)
    return gnb
