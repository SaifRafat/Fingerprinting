# =============================================================================
# testByVNGpp.py — Inference with VNG++ (Gaussian Naive Bayes) classifier
# =============================================================================


def test(model, test_data):
    """
    Run prediction using a pre-trained VNG++ (GaussianNB) model.

    model     : fitted sklearn model
    test_data : 2-D array-like, shape (n_samples, n_features)

    Returns a numpy array of predicted integer labels.
    """
    return model.predict(test_data)
