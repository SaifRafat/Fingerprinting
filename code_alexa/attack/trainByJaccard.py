# =============================================================================
# trainByJaccard.py — Train Jaccard set-based classifier
# 
# For each class, creates a representative packet set using majority vote
# across training samples. Unknown traffic is classified by highest set overlap.
# =============================================================================

import os
import math
from collections import defaultdict

import fileUtils
import testByJaccard   # For getLabel()


def readfile(fpath):
    """
    Read a traffic CSV and return a set of (packet_size × direction) products.
    Each product is an integer that uniquely encodes size + direction together.
    """
    packet_set = set()
    for line in fileUtils.readTxtFile(fpath, ignore=','):
        parts = line.split(',')
        if len(parts) == 4:
            elem = fileUtils.str2int(parts[-1]) * fileUtils.str2int(parts[-2])
        elif len(parts) == 1:
            elem = parts[0]
        else:
            elem = fileUtils.str2int(parts[4]) * fileUtils.str2int(parts[3])
        packet_set.add(elem)
    return packet_set


def _group_files_by_class(filepath_list):
    """Return a dict of {class_label: [filepath, ...]} grouped by label."""
    class_dict = defaultdict(list)
    for fpath in filepath_list:
        label = testByJaccard.getLabel(fpath)
        class_dict[label].append(fpath)
    return class_dict


def _train_from_file_list(file_list):
    """
    Given multiple files for one class, build the representative set by
    majority vote: keep only elements that appear in ≥ ceil(N/2) files.
    """
    sets_per_file = [readfile(f) for f in file_list]
    threshold     = math.ceil(len(sets_per_file) / 2)

    # union of all elements
    all_elements = set()
    for s in sets_per_file:
        all_elements |= s

    # keep only majority-present elements
    representative_set = set()
    for elem in all_elements:
        count = sum(1 for s in sets_per_file if elem in s)
        if count >= threshold:
            representative_set.add(elem)

    return representative_set


def _write_class_files(output_dir, class_sets):
    """Write each class's representative set to a file named after the class."""
    for label, elem_set in class_sets.items():
        fpath   = os.path.join(output_dir, label)
        content = '\n'.join(str(e) for e in elem_set)
        fileUtils.writeTxtFile(fpath, content)
    print(f'Wrote {len(class_sets)} class files to {output_dir}')


def train(filepath_list, class_file_dir):
    """Build class representative packet sets; write one file per class to output directory."""
    grouped    = _group_files_by_class(filepath_list)
    class_sets = {}

    for label, files in grouped.items():
        if len(files) == 1:
            class_sets[label] = readfile(files[0])
        else:
            class_sets[label] = _train_from_file_list(files)

    _write_class_files(class_file_dir, class_sets)
