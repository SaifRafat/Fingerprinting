# =============================================================================
# tools.py — Histogram bucket helpers shared by Bayes, VNG++, and AdaBoost modules
# =============================================================================


def getSectionList(start, end, interval):
    """
    Create histogram buckets for a numeric range.

    Returns:
        range_boundaries : List of bucket edge values  
        section_counts   : List of int counters (all zeros), one per bucket
    """
    range_boundaries = list(range(start, end, interval))
    section_counts   = [0] * len(range_boundaries)
    return range_boundaries, section_counts


def computeRange(range_boundaries, value):
    """
    Find the bucket index for a value in a histogram.
    Returns the index of the rightmost boundary ≤ value (clamped to valid range).
    """
    index = 0
    for i, boundary in enumerate(range_boundaries):
        if value >= boundary:
            index = i
        else:
            break
    return index
