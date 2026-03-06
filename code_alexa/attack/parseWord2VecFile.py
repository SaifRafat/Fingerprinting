# =============================================================================
# parseWord2VecFile.py — Load Word2Vec embeddings from CSV files
# 
# Converts custom Word2Vec CSV files (label + vector per entry) into a
# dictionary mapping labels to vector arrays. Handles multiple formats and
# normalizes labels for consistency with dataset filenames.
# =============================================================================

import re
from collections import defaultdict
import fileUtils


def normalize_label(label):
    """Normalize label: lowercase, remove quotes/apostrophes/hyphens, replace spaces with _."""
    label = label.lower()
    label = label.replace('\u0022', '').replace('\u0027', '').replace('\u2018', '')
    label = label.replace('\u2019', '').replace('\u201c', '').replace('\u201d', '')
    label = label.replace('-', '').replace('?', '').replace('.', '').replace(',', '')
    label = label.replace('(', '').replace(')', '').replace(' ', '_')
    while '__' in label:
        label = label.replace('__', '_')
    return label.strip('_')


def _is_entry_start(line):
    """True if the line begins with a letter — marks the start of a new entry."""
    return bool(re.match(r'^[A-Za-z]+.*', line))


def _is_entry_end(line):
    """True if the line starts with '#end' — marks the end of an entry."""
    return line.startswith('#end')


def _parse_number(item):
    """Strip brackets/quotes and return the numeric string inside."""
    m = re.match(r'[\"\[]*([-\.0-9e]*)[\]\"\,]*', item)
    if m:
        return m.group(1)
    raise ValueError(f'Could not parse number from: {item}')


def _parse_vector_line(line):
    """
    Split a line into tokens and convert each to float.
    Handles both space-separated and comma-separated formats.
    Skips any non-numeric tokens.
    """
    # Remove surrounding quotes if present
    line = line.strip()
    if line.startswith('"') and line.endswith('"'):
        line = line[1:-1]
    
    # Try comma-separated first (preferred format)
    if ',' in line:
        tokens = line.split(',')
    else:
        tokens = [t.strip() for t in line.split(' ') if t.strip()]
    
    result = []
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        
        # Try to convert to float directly
        try:
            result.append(fileUtils.str2float(token))
        except ValueError:
            # Skip tokens that can't be converted to float (non-numeric)
            continue
    return result


def _looks_like_number(token):
    """Check if a token looks like it could be a number."""
    try:
        float(token)
        return True
    except ValueError:
        return False


def loadData(fpath):
    """
    Parse a Word2Vec file into a dict of {label_string: [float, float, ...]}.

    Supports both formats:
        CommaFormat: LabelName, v1,v2,v3 ...
        SpaceFormat: LabelName v1 v2 v3 ...
    
    Handles multiple encodings (UTF-8, cp1252) to deal with smart quotes and special characters.
    """
    vec_dict  = defaultdict(list)
    vec_accum = []
    title     = ''
    expected_dim = None
    dimension_warnings = []

    # Try different encodings
    encodings = ['utf-8', 'cp1252', 'latin-1']
    file_content = None
    
    for encoding in encodings:
        try:
            with open(fpath, 'r', encoding=encoding) as f:
                file_content = f.readlines()
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    
    if file_content is None:
        raise ValueError(f'Could not decode file {fpath} with any supported encoding')

    for line_num, line in enumerate(file_content, 1):
        if line == '"\n':
            continue

        if _is_entry_start(line):
            # Handle comma-separated format: "Label, v1 v2 v3"
            if ',' in line:
                parts  = line.split(',', 1)  # Split only on first comma
                title  = normalize_label(parts[0].strip())
                vec_accum = _parse_vector_line(parts[1] if len(parts) > 1 else '')
            else:
                # Handle space-separated format: "Label v1 v2 v3"
                parts = line.split()
                title = normalize_label(parts[0].strip())
                if len(parts) > 1:
                    vec_accum = _parse_vector_line(' '.join(parts[1:]))

        elif _is_entry_end(line):
            # Validate vector dimension consistency
            if vec_accum:
                if expected_dim is None:
                    expected_dim = len(vec_accum)
                elif len(vec_accum) != expected_dim:
                    dimension_warnings.append(f'{title}: got {len(vec_accum)} dims, expected {expected_dim}')
            
            vec_dict[title] = vec_accum
            vec_accum = []
            title     = ''

        else:
            vec_accum.extend(_parse_vector_line(line))

    # Report dimension warnings
    if dimension_warnings:
        print(f'\nWarning: Vector Dimension Mismatches in {fpath}:')
        for warning in dimension_warnings[:5]:  # Show first 5 warnings
            print(f'   {warning}')
        if len(dimension_warnings) > 5:
            print(f'   ... and {len(dimension_warnings) - 5} more')
        print(f'Expected dimension: {expected_dim}\n')

    return vec_dict
