# =============================================================================
# fileUtils.py — Core file I/O and string conversion utilities
# 
# Provides foundational I/O operations shared across all attack modules.
# =============================================================================

import os


def str2int(str_input):
    """Convert string to int, handling float strings like '3.0'."""
    if '.' in str_input:
        return int(float(str_input))
    return int(str_input)


def str2float(str_input):
    """Convert string to float."""
    return float(str_input)


def readTxtFile(fpath, ignore=''):
    """Yield lines from file, skipping those starting with 'ignore' prefix."""
    with open(fpath, 'r') as f:
        for line in f:
            line = line.strip()
            if ignore and line.startswith(ignore):
                continue
            yield line


def writeTxtFile(fpath, content):
    """Write a string to a file, overwriting if it exists."""
    with open(fpath, 'w') as f:
        f.write(content)


def genfilelist(pathDir):
    """Return a list of absolute file paths for every file in pathDir."""
    filenames = os.listdir(pathDir)
    return [os.path.join(pathDir, fname) for fname in filenames]
