# =============================================================================
# trainByBayes.py — Feature extraction + training for Gaussian Naive Bayes
# 
# Builds histogram features from packet size × direction products.
# Trains a Gaussian Naive Bayes classifier on these histograms.
# =============================================================================

import numpy as np
from sklearn.naive_bayes import GaussianNB

import fileUtils
import tools


def readfile(fpath):
    """Read traffic CSV and return list of (packet_size × direction) products."""
    products = []
    for line in fileUtils.readTxtFile(fpath, ignore=','):
        parts = line.split(',')
        
        # Skip header row
        try:
            int(parts[0]) if parts[0].strip() else float(parts[1])
        except (ValueError, IndexError):
            continue
        
        # Extract [index, time, size, direction, ...]
        try:
            size = fileUtils.str2int(parts[2])
            direction = fileUtils.str2int(parts[3])
            product = size * direction  # Encode both into product
            products.append(product)
        except (IndexError, ValueError):
            continue
    
    return products


def computeFeature(fpath, range_start, range_end, interval):
    """Build histogram feature vector from (size × direction) products binned by interval."""
    boundaries, counts = tools.getSectionList(range_start, range_end, interval)
    for value in readfile(fpath):
        idx        = tools.computeRange(boundaries, value)
        counts[idx] += 1
    return counts


def train(train_data, train_labels):
    """Fit Gaussian Naive Bayes model. Returns trained model."""
    gnb = GaussianNB()
    gnb.fit(train_data, train_labels)
    return gnb
