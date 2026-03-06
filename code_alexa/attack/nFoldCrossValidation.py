# =============================================================================
# nFoldCrossValidation.py — N-fold stratified cross-validation for all models
# =============================================================================

import os
import tempfile
import shutil
import numpy as np
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold

import fileUtils
import parseWord2VecFile
import testByJaccard

import trainByJaccard
import trainByBayes
import trainByVNGpp
import trainByAdaBoost

import testByBayes
import testByVNGpp
import testByAdaBoost


# ── Label helpers ─────────────────────────────────────────────────────────────

def _build_label_map(filepath_list):
    """
    Assign a unique integer (starting from 1) to every class label found
    in filepath_list.  Returns {class_name: int}.
    """
    label_map = {}
    counter   = 1
    for fpath in filepath_list:
        label = testByJaccard.getLabel(fpath)
        if label not in label_map:
            label_map[label] = counter
            counter += 1
    return label_map


def _label_for_file(fpath, label_map):
    """Return the integer label for a given file path."""
    return label_map[testByJaccard.getLabel(fpath)]


def _nums_to_strings(int_labels, label_map):
    """Convert a list of integer labels back to class name strings."""
    reverse_map = {v: k for k, v in label_map.items()}
    return [reverse_map[i] for i in int_labels]


def _strings_to_nums(str_labels, label_map):
    """Convert a list of class name strings to integer labels."""
    return [label_map[s] for s in str_labels]


# ── Feature extraction dispatcher ────────────────────────────────────────────

def _extract_feature(fpath, model_name, interval):
    """Call model-specific feature extractor; Jaccard returns path, others return feature vector."""
    if model_name == 'Jaccard':
        return fpath

    elif model_name == 'Bayes':
        return trainByBayes.computeFeature(fpath,
                                            range_start=-1500,
                                            range_end=1501,
                                            interval=interval)
    elif model_name == 'VNGpp':
        return trainByVNGpp.computeFeature(fpath,
                                            range_start=-400000,
                                            range_end=400001,
                                            interval=interval)
    elif model_name == 'AdaBoost':
        return trainByAdaBoost.computeFeature(fpath,
                                          range_start=-200000,
                                          range_end=200001,
                                          interval=interval)
    else:
        raise ValueError(f'Unknown model: {model_name}. Choose Jaccard/Bayes/VNGpp/AdaBoost')


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(data_dir, model_name, interval):
    """
    Load all traffic files from data_dir.

    Returns:
        all_data   : numpy array of feature vectors (or file paths for Jaccard)
        all_labels : numpy array of integer class labels
        label_map  : dict {class_name: int}
    """
    file_list  = fileUtils.genfilelist(data_dir)
    label_map  = _build_label_map(file_list)

    data_list  = []
    label_list = []

    for fpath in file_list:
        feature = _extract_feature(fpath, model_name, interval)
        data_list.append(feature)
        label_list.append(_label_for_file(fpath, label_map))

    return np.array(data_list), np.array(label_list), label_map


# ── Accuracy ──────────────────────────────────────────────────────────────────

def compute_accuracy(predictions, labels):
    """Return fraction of correct predictions."""
    assert len(predictions) == len(labels)
    return sum(p == l for p, l in zip(predictions, labels)) / len(labels)


# ── Word2vec scoring ──────────────────────────────────────────────────────────

_NUM_TOKEN_MAP = {
    '0': 'zero',
    '1': 'one',
    '2': 'two',
    '3': 'three',
    '4': 'four',
    '5': 'five',
    '6': 'six',
    '7': 'seven',
    '8': 'eight',
    '9': 'nine',
    '10': 'ten',
    '11': 'eleven',
    '12': 'twelve',
    '13': 'thirteen',
    '14': 'fourteen',
    '15': 'fifteen',
    '16': 'sixteen',
    '17': 'seventeen',
    '18': 'eighteen',
    '19': 'nineteen',
    '20': 'twenty',
    '30': 'thirty',
    '40': 'forty',
    '50': 'fifty',
    '60': 'sixty',
}
_STOP_TOKENS = {'a', 'an', 'the', 'is', 'are'}


def _canonical_intent_tokens(label):
    """
    Build a forgiving token signature so minor surface-form differences
    between dataset labels and query-list labels still map to one intent.
    """
    label = str(label).strip().lower()
    label = label.replace("'", '')
    label = label.replace('-', '_')
    label = label.replace('table_spoon', 'tablespoon')

    raw_tokens = [tok for tok in label.split('_') if tok]
    tokens = []
    for tok in raw_tokens:
        if tok == 'whats':
            # Treat "what's" and "what is" as the same intent stem.
            tokens.append('what')
            continue

        tok = _NUM_TOKEN_MAP.get(tok, tok)
        if tok in _STOP_TOKENS:
            continue

        # Merge singular/plural variants like election/elections, second/seconds.
        if tok.endswith('s') and len(tok) > 4:
            tok = tok[:-1]

        tokens.append(tok)
    return tuple(tokens)


def _build_word2vec_alias_map(eval_labels, word2vec_labels):
    """
    Return:
        alias_map     : dict {eval_label -> word2vec_key}
        unresolved    : list of eval labels not matched to any key
    """
    signature_to_keys = defaultdict(list)
    for key in word2vec_labels:
        signature_to_keys[_canonical_intent_tokens(key)].append(key)

    alias_map = {}
    unresolved = []

    for label in sorted(set(eval_labels)):
        if label in word2vec_labels:
            alias_map[label] = label
            continue

        signature = _canonical_intent_tokens(label)
        matches = signature_to_keys.get(signature, [])

        if not matches and signature:
            # Fallback for cases like "play_npr" vs "play_npr_917_wvxu".
            sig_set = set(signature)
            fuzzy_matches = []
            for key_sig, key_list in signature_to_keys.items():
                key_set = set(key_sig)
                overlap = len(sig_set & key_set)
                if overlap >= 2 and (sig_set.issubset(key_set) or key_set.issubset(sig_set)):
                    fuzzy_matches.extend(key_list)
            matches = sorted(set(fuzzy_matches))

        if not matches:
            unresolved.append(label)
            continue

        if len(matches) > 1:
            # Deterministic fallback if canonical signature is not unique.
            chosen = sorted(matches, key=len)[0]
            print(f'⚠️  Warning: Ambiguous alias for "{label}" -> "{chosen}" from {matches}')
            alias_map[label] = chosen
        else:
            alias_map[label] = matches[0]

    return alias_map, unresolved


def _cosine_similarity(vec_a, vec_b):
    """Compute cosine similarity between two vectors. Handle dimension mismatches."""
    a, b = np.array(vec_a), np.array(vec_b)
    
    # Check for dimension mismatch
    if len(a) != len(b):
        print(f'⚠️  Warning: Vector dimension mismatch - {len(a)} vs {len(b)}')
        # Pad shorter vector with zeros
        if len(a) < len(b):
            a = np.pad(a, (0, len(b) - len(a)), mode='constant')
        else:
            b = np.pad(b, (0, len(a) - len(b)), mode='constant')
    
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return np.dot(a, b) / denom if denom else 0.0


def _rank_score(word2vec_dict, predicted_label, true_label):
    """
    Find the rank of true_label when all class vectors are sorted by
    cosine similarity to the predicted_label vector.
    Lower rank = better (0 means the nearest neighbor is correct).
    Handles dimension mismatches gracefully.
    Returns None if either label is missing from word2vec_dict.
    """
    if predicted_label not in word2vec_dict:
        print(f'⚠️  Warning: Predicted label "{predicted_label}" not found in word2vec dictionary')
        return None
    if true_label not in word2vec_dict:
        print(f'⚠️  Warning: True label "{true_label}" not found in word2vec dictionary')
        return None
    
    pred_vec   = word2vec_dict[predicted_label]
    score_list = []
    for label, vec in word2vec_dict.items():
        if label == predicted_label:
            continue
        try:
            sim_score = _cosine_similarity(vec, pred_vec)
            score_list.append((label, sim_score))
        except Exception as e:
            print(f'⚠️  Error computing similarity for {label}: {e}')
            continue

    score_list.sort(key=lambda x: x[1])   # ascending: index 0 = most similar
    for rank, (label, _) in enumerate(score_list):
        if label == true_label:
            return rank
    return len(score_list)  # Return max rank if true_label not found in similarity list


def write_test_results(true_labels_str, predicted_labels_str,
                       acc_list, avg_accuracy,
                       word2vec_file, result_file, model_name):
    """
    Write a detailed result file including:
        - per-sample: true label, prediction, cosine distance, rank score
        - summary:    average/variance of rank score and accuracy
    """
    word2vec_dict = parseWord2VecFile.loadData(word2vec_file)
    assert len(true_labels_str) == len(predicted_labels_str)
    
    eval_labels = set(true_labels_str) | set(predicted_labels_str)
    label_alias_map, unresolved_labels = _build_word2vec_alias_map(
        eval_labels, set(word2vec_dict.keys())
    )
    resolved_alias_count = sum(
        1 for label, mapped in label_alias_map.items() if label != mapped
    )
    if resolved_alias_count:
        print(f'ℹ️  Resolved {resolved_alias_count} label aliases for word2vec lookup')
    if unresolved_labels:
        print(
            f'⚠️  {len(unresolved_labels)} labels still unresolved before scoring: '
            f'{unresolved_labels[:10]}'
        )

    lines  = ['label\tprediction\tcosine_distance\trank_score']
    scores = []
    ranks  = []
    missing_labels = set()

    for true, pred in zip(true_labels_str, predicted_labels_str):
        try:
            true_key = label_alias_map.get(true, true)
            pred_key = label_alias_map.get(pred, pred)

            # Check if labels exist in word2vec dictionary
            if true_key not in word2vec_dict:
                missing_labels.add(true)
                print(f'⚠️  Warning: True label "{true}" not found in word2vec dictionary')
                lines.append(f'{true}\t{pred}\tMISSING_LABEL\tN/A')
                continue
            if pred_key not in word2vec_dict:
                missing_labels.add(pred)
                print(f'⚠️  Warning: Predicted label "{pred}" not found in word2vec dictionary')
                lines.append(f'{true}\t{pred}\tMISSING_LABEL\tN/A')
                continue
            
            vec_true = word2vec_dict[true_key]
            vec_pred = word2vec_dict[pred_key]
            cos_dist = _cosine_similarity(vec_true, vec_pred)
            rank     = _rank_score(word2vec_dict, pred_key, true_key)
            
            if rank is not None:
                scores.append(cos_dist)
                ranks.append(rank)
                lines.append(f'{true}\t{pred}\t{cos_dist:.4f}\t{rank}')
            else:
                lines.append(f'{true}\t{pred}\t{cos_dist:.4f}\tN/A')
        except Exception as e:
            print(f'⚠️  Error processing ({true}, {pred}): {e}')
            lines.append(f'{true}\t{pred}\tERROR\tERROR')
    
    # Report missing labels summary
    if missing_labels:
        print(f'\n⚠️  Total unique labels missing from word2vec: {len(missing_labels)}')
        print(f'   Missing labels: {sorted(missing_labels)[:10]}')
        if len(missing_labels) > 10:
            print(f'   ... and {len(missing_labels) - 10} more')

    lines.append('\n=========== Summary ===========')
    if scores:
        lines.append(f'Average cosine distance : {np.mean(scores):.4f}')
    else:
        lines.append('Average cosine distance : N/A (no valid scores)')
    
    if ranks:
        avg_rank = np.mean(ranks)
        std_rank = np.std(ranks)
        lines.append(f'Average rank score      : {avg_rank:.4f}')
        lines.append(f'Rank score std          : {std_rank:.4f}')
        # Normalized Semantic Distance: convert rank to normalized distance (0-100 scale)
        # Higher rank = worse match, so normalized SD = rank value itself
        lines.append(f'Normalized Semantic Distance : {avg_rank:.4f}')
        lines.append(f'Norm SD std             : {std_rank:.4f}')
    else:
        lines.append('Average rank score      : N/A (no valid ranks)')
        lines.append('Rank score std          : N/A')
        lines.append('Normalized Semantic Distance : N/A')
        lines.append('Norm SD std             : N/A')
    
    lines.append(f'Accuracy per fold       : {acc_list}')
    lines.append(f'Average accuracy [{model_name}] : {avg_accuracy:.4f}')
    lines.append(f'Accuracy std            : {np.std(acc_list):.4f}')

    content = '\n'.join(lines)
    print(content)
    fileUtils.writeTxtFile(result_file, content)
    print(f'\nResults written to {result_file}')
    return float(np.mean(ranks)) if ranks else 0.0


# ── Main cross-validation function ───────────────────────────────────────────

def run_cross_validation(data_dir,
                          model_name,
                          n_folds,
                          interval,
                          result_file,
                          word2vec_file=None):
    """
    Run stratified N-fold cross-validation.

    Parameters
    ----------
    data_dir      : directory containing all labelled traffic files
    model_name    : 'Jaccard' | 'Bayes' | 'VNGpp' | 'AdaBoost'
    n_folds       : number of stratified folds (e.g. 5)
    interval      : histogram bucket size for Bayes/VNGpp/AdaBoost
    result_file   : path to write the result summary
    word2vec_file : (optional) path to word2vec file for semantic rank scoring

    Returns
    -------
    avg_rank_score : float or None   (only when word2vec_file is given)
    avg_accuracy   : float
    """
    print(f'\n[Cross-validation] model={model_name}  folds={n_folds}  interval={interval}')

    all_data, all_labels, label_map = load_data(data_dir, model_name, interval)
    skf      = StratifiedKFold(n_splits=n_folds)
    acc_list = []

    # track last fold's string predictions/labels for word2vec scoring
    last_true_str  = []
    last_pred_str  = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(all_data, all_labels)):
        X_train, X_test = all_data[train_idx],   all_data[test_idx]
        Y_train, Y_test = all_labels[train_idx], all_labels[test_idx]

        # ── train + predict ───────────────────────────────────────────────────
        if model_name == 'Jaccard':
            # Jaccard uses file paths, not numeric arrays
            tmp_dir = tempfile.mkdtemp()
            try:
                trainByJaccard.train(X_train.tolist(), tmp_dir)
                str_predictions = testByJaccard.test(X_test.tolist(), tmp_dir)
            finally:
                shutil.rmtree(tmp_dir)
            int_predictions = _strings_to_nums(str_predictions, label_map)

        elif model_name == 'Bayes':
            model           = trainByBayes.train(X_train, Y_train)
            int_predictions = testByBayes.test(model, X_test).tolist()
            str_predictions = _nums_to_strings(int_predictions, label_map)

        elif model_name == 'VNGpp':
            model           = trainByVNGpp.train(X_train, Y_train)
            int_predictions = testByVNGpp.test(model, X_test).tolist()
            str_predictions = _nums_to_strings(int_predictions, label_map)

        elif model_name == 'AdaBoost':
            model           = trainByAdaBoost.train(X_train, Y_train, n_estimators=50)
            int_predictions = testByAdaBoost.test(model, X_test).tolist()
            str_predictions = _nums_to_strings(int_predictions, label_map)

        else:
            raise ValueError(f'Unknown model: {model_name}')

        accuracy = compute_accuracy(int_predictions, Y_test.tolist())
        acc_list.append(accuracy)
        print(f'  Fold {fold_idx + 1}: accuracy = {accuracy:.4f}')

        # keep last fold for the report
        last_true_str  = _nums_to_strings(Y_test.tolist(), label_map)
        last_pred_str  = str_predictions

    avg_accuracy = sum(acc_list) / len(acc_list)
    print(f'\nAverage accuracy [{model_name}]: {avg_accuracy:.4f}')
    print(f'Per-fold results: {acc_list}')

    # Save basic results summary (always)
    basic_results = [
        f'Model: {model_name}',
        f'Interval: {interval}',
        f'Average accuracy: {avg_accuracy:.4f}',
        f'Accuracy std deviation: {np.std(acc_list):.4f}',
        f'Per-fold accuracies: {acc_list}',
    ]
    basic_content = '\n'.join(basic_results)
    fileUtils.writeTxtFile(result_file, basic_content)
    print(f'Results written to {result_file}')

    avg_rank_score = None
    if word2vec_file:
        avg_rank_score = write_test_results(
            last_true_str, last_pred_str,
            acc_list, avg_accuracy,
            word2vec_file, result_file, model_name
        )

    return avg_rank_score, avg_accuracy
