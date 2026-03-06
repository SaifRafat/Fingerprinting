# =============================================================================
# testByAdaBoost.py — Inference with AdaBoost ensemble classifier
# =============================================================================

import numpy as np


def test(model, test_data):
    """
    Run prediction using a pre-trained AdaBoost model.

    model     : fitted sklearn AdaBoostClassifier
    test_data : 2-D array-like, shape (n_samples, n_features)

    Returns a numpy array of predicted integer labels.
    """
    # Convert test_data to numpy array
    test_data = np.array(test_data)
    return model.predict(test_data)
