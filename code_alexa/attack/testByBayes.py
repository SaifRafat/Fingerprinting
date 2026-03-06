# =============================================================================
# testByBayes.py — Inference with Gaussian Naive Bayes classifier
# =============================================================================


def test(model, test_data):
    """
    Run prediction on test_data using a pre-trained sklearn model.

    model     : fitted GaussianNB (or any sklearn classifier)
    test_data : 2-D array-like, shape (n_samples, n_features)

    Returns a numpy array of predicted integer labels.
    """
    return model.predict(test_data)
