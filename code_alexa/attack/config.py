# =============================================================================
# config.py — All hardcoded paths and settings for the project
# Change these paths to match your local setup before running anything.
# =============================================================================

import os

# ── Root directories ─────────────────────────────────────────────────────────
DATA_DIR          = 'E:\\Research work\\wenggang\\paper\\code_alexa\\data\\trace_csv'          # Original traffic traces
BUFLO_DIR         = 'E:\\Research work\\wenggang\\paper\\code_alexa\\data\\buflo_padsets'     # BuFLO-obfuscated traces
RESULT_DIR        = 'E:\\Research work\\wenggang\\paper\\code_alexa\\result'                  # Result text files
RESULTS_DIR       = RESULT_DIR                                                                 # Alias for backward compatibility
FIGURES_DIR       = 'E:\\Research work\\wenggang\\paper\\code_alexa\\figures\\'               # Generated plots
MODEL_SAVE_DIR    = 'E:\\Research work\\wenggang\\paper\\code_alexa\\models\\'                # Trained model files
TEMP_DIR          = 'E:\\Research work\\wenggang\\paper\\code_alexa\\temp_files\\'            # Temporary files

# ── Word2Vec vector files ────────────────────────────────────────────────────
# Directory containing generated doc2vec CSV files
Quora_WORD2VEC     = r'E:\Research work\wenggang\paper\code_alexa\doc2vec_model\vector_csv\quora'
Yahoo_WORD2VEC     = r'E:\Research work\wenggang\paper\code_alexa\doc2vec_model\vector_csv\yahoo'

# Doc2Vec model paths
DOC2VEC_MODELS_QUORA_DIR = r'E:\Research work\wenggang\paper\code_alexa\doc2vec_model\quora_models'
DOC2VEC_MODELS_YAHOO_DIR = r'E:\Research work\wenggang\paper\code_alexa\doc2vec_model\yahoo_models'
DOC2VEC_QUORA_MODEL = os.path.join(DOC2VEC_MODELS_QUORA_DIR, 'quora_e100_v300.bin')  # Quora-trained
DOC2VEC_YAHOO_MODEL = os.path.join(DOC2VEC_MODELS_YAHOO_DIR, 'yahoo_e100_v300.bin')  # Yahoo-trained

# Quora vector files (different epochs/dimensions)
QUORA_E100_V300 = os.path.join(Quora_WORD2VEC, 'vectors_queries_quora_e100_v300.csv')
QUORA_E100_V325 = os.path.join(Quora_WORD2VEC, 'vectors_queries_quora_e100_v325.csv')
QUORA_E100_V350 = os.path.join(Quora_WORD2VEC, 'vectors_queries_quora_e100_v350.csv')
QUORA_E100_V375 = os.path.join(Quora_WORD2VEC, 'vectors_queries_quora_e100_v375.csv')
QUORA_E125_V300 = os.path.join(Quora_WORD2VEC, 'vectors_queries_quora_e125_v300.csv')
QUORA_E150_V300 = os.path.join(Quora_WORD2VEC, 'vectors_queries_quora_e150_v300.csv')
QUORA_E175_V300 = os.path.join(Quora_WORD2VEC, 'vectors_queries_quora_e175_v300.csv')

# Yahoo vector files (different epochs/dimensions)
YAHOO_E100_V300 = os.path.join(Yahoo_WORD2VEC, 'yahoo_e100_v300.csv')
YAHOO_E100_V325 = os.path.join(Yahoo_WORD2VEC, 'yahoo_e100_v325.csv')
YAHOO_E100_V350 = os.path.join(Yahoo_WORD2VEC, 'yahoo_e100_v350.csv')
YAHOO_E100_V375 = os.path.join(Yahoo_WORD2VEC, 'yahoo_e100_v375.csv')
YAHOO_E125_V300 = os.path.join(Yahoo_WORD2VEC, 'yahoo_e125_v300.csv')
YAHOO_E150_V300 = os.path.join(Yahoo_WORD2VEC, 'yahoo_e150_v300.csv')
YAHOO_E175_V300 = os.path.join(Yahoo_WORD2VEC, 'yahoo_e175_v300.csv')

# Backward compatibility - deprecated, use QUORA_* or YAHOO_* instead
# For semantic distance evaluation from paper: use both Yahoo and Quora for comparison
WORD2VEC_E100_V300 = QUORA_E100_V300  # Default model (Quora)
WORD2VEC_E100_V325 = QUORA_E100_V325
WORD2VEC_E100_V350 = QUORA_E100_V350
WORD2VEC_E100_V375 = QUORA_E100_V375
WORD2VEC_E125_V300 = QUORA_E125_V300
WORD2VEC_E150_V300 = QUORA_E150_V300
WORD2VEC_E175_V300 = QUORA_E175_V300

# Paper evaluation: use both Yahoo and Quora to compare semantic distance
# This matches the paper's approach (Figures 4-12) of showing both models

# ── Cross-validation settings ────────────────────────────────────────────────
N_FOLDS           = 5                  # Number of folds for StratifiedKFold

# ── Feature extraction intervals (histogram bucket sizes) ────────────────────
BAYES_INTERVAL    = 50                 # Bayes classifier histogram bucket
VNGPP_INTERVAL    = 5000               # VNG++ histogram bucket
ADABOOST_INTERVAL = 5000               # AdaBoost histogram bucket

# ── BuFLO defense parameters ──────────────────────────────────────────────────
BUFLO_F = 50     # Transmission frequency (packets per second)
BUFLO_T = 20     # Minimum transmission time (seconds)
BUFLO_SIZES = [1000, 1100, 1200, 1300, 1400, 1500]  # Padding sizes to test

# ── Auto-create directories if they don't exist ──────────────────────────────
for _dir in [RESULT_DIR, FIGURES_DIR, MODEL_SAVE_DIR, TEMP_DIR]:
    os.makedirs(_dir, exist_ok=True)
