# =============================================================================
# trainByAdaBoost.py — Feature extraction + training for the AdaBoost classifier
# 
# Extends VNG++ features with packet ratio and burst statistics.
# Uses shallow decision trees as weak learners.
# =============================================================================

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np

import fileUtils
import tools
import trainByVNGpp   # Reuses readfile() and calculateBursts()


def computeFeature(fpath, range_start, range_end, interval):
    """Extract AdaBoost features: flow statistics + burst histograms."""
    up_pack_num, down_pack_num, up_total, down_total, _, tuple_list = \
        trainByVNGpp.readfile(fpath)

    burst_list = trainByVNGpp.calculateBursts(tuple_list)
    boundaries, counts = tools.getSectionList(range_start, range_end, interval)
    for burst in burst_list:
        idx        = tools.computeRange(boundaries, burst)
        counts[idx] += 1

    total_packets   = len(tuple_list)
    in_packet_ratio = down_pack_num / total_packets if total_packets else 0.0
    burst_count     = len(burst_list)

    feature = [up_total, down_total, in_packet_ratio, total_packets, burst_count] + counts
    return feature


def train(train_data, train_labels, n_estimators=50):
    """Fit AdaBoost classifier with decision stump weak learners."""
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    base_estimator = DecisionTreeClassifier(max_depth=1)  # decision stump
    
    clf = AdaBoostClassifier(
        estimator=base_estimator,
        n_estimators=n_estimators,
        learning_rate=1.0,
        algorithm='SAMME'
    )
    clf.fit(train_data, train_labels)
    return clf
