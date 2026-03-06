# =============================================================================
# testByJaccard.py — Classify traffic using set similarity (Jaccard index)
# 
# Compares test trace packet sets against trained class representatives
# and predicts the class with highest Jaccard similarity score.
# =============================================================================

import os
import re

import fileUtils
import trainByJaccard   # For readfile()
import parseWord2VecFile  # For normalize_label()


def getLabel(fpath):
    """Extract class label from filename. Expects format: <ClassName>_<number>..."""
    pattern = r'([a-zA-Z\'_-]+)_[0-9].*'
    fname   = os.path.basename(fpath)
    match   = re.match(pattern, fname)
    raw_label = match.group(1) if match else fname
    return parseWord2VecFile.normalize_label(raw_label)


def _jaccard_similarity(set_a, set_b):
    """Return Jaccard similarity: |A ∩ B| / |A ∪ B|."""
    intersection = set_a & set_b
    union        = set_a | set_b
    if not union:
        return 0.0
    return len(intersection) / len(union)


def _classify_one_file(test_fpath, class_file_paths):
    """
    Compare test_fpath against every class model file and return the
    label of the class with the highest Jaccard similarity.
    """
    test_set  = set(map(int, trainByJaccard.readfile(test_fpath)))
    best_score = -1
    best_label = ''

    for class_fpath in class_file_paths:
        class_set = set(map(int, trainByJaccard.readfile(class_fpath)))
        score     = _jaccard_similarity(test_set, class_set)
        if score > best_score:
            best_score = score
            best_label = getLabel(class_fpath)

    return best_label


def test(test_filepath_list, model_file_dir):
    """Classify test files using Jaccard model. Returns list of predicted labels."""
    class_files = fileUtils.genfilelist(model_file_dir)
    predictions = []
    for fpath in test_filepath_list:
        label = _classify_one_file(fpath, class_files)
        predictions.append(label)
    return predictions
