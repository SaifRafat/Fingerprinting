# Code Alexa: Voice Command Fingerprinting on Smart Home Speakers

## Overview

This is a reproduction of the traffic fingerprinting attack research against Amazon Alexa based on the paper *"I Can Hear Your Alexa: Voice Command Fingerprinting on Smart Home Speakers"*. 

The project implements voice command fingerprinting attacks by analyzing network traffic patterns and develops a defense mechanism to mitigate these attacks.

## Methodology

### 1. Problem Reproduction

The goal was to reproduce the attack classification results from the original paper by implementing four different classifiers that can identify Alexa voice commands from network traffic alone.

### 2. Dataset Collection

The original paper used Yahoo and Quora datasets for semantic similarity analysis (Doc2Vec training), but these datasets were not available at the original URLs mentioned in the paper.

**Dataset Sources:**
- **Quora Dataset**: [Quora Question Pairs Dataset](https://www.kaggle.com/datasets/quora/question-pairs-dataset)
  - Used for training Doc2Vec models
  - Believed to be the exact dataset used in the original paper

- **Yahoo Dataset**: [Yahoo Answers Dataset](https://www.kaggle.com/datasets/soumikrakshit/yahoo-answers-dataset)
  - Alternative dataset used for semantic analysis
  - Filtered to include only questions starting with "how"
  - Provides semantic diversity similar to the original research

### 3. Doc2Vec Model Training

Trained Doc2Vec models on the collected datasets with varying configurations to analyze the impact on semantic similarity metrics:
- **Variable Epochs**: Multiple training iterations to test model convergence
- **Variable Vector Sizes**: Different embedding dimensions to optimize semantic representation
- These variations later enable analysis of how Doc2Vec parameters affect normalized semantic distance during attack evaluation

### 4. Attack Implementation

Implemented four classifiers to fingerprint voice commands:

**Classifiers:**
- **Jaccard** (`trainByJaccard.py`, `testByJaccard.py`) - Set-based packet signature similarity
- **Bayes** (`trainByBayes.py`, `testByBayes.py`) - Gaussian Naive Bayes on histogram features
- **VNG++** (`trainByVNGpp.py`, `testByVNGpp.py`) - Burst-based traffic analysis
- **AdaBoost** (`trainByAdaBoost.py`, `testByAdaBoost.py`) - Ensemble learning approach

**Parameters Tested:**
- **Rounding Parameter Impact**: Analyzed how rounding network packet sizes affects attack accuracy
- **Cross-validation**: 5-fold stratified cross-validation for robust evaluation

### 5. Defense Implementation

Implemented **BuFLO (Buffered Fixed-Length Obfuscation)** defense mechanism (`defense/buflo.py`):
- Pads all packets to fixed sizes (tested: 1000-1500 bytes)
- Enforces fixed transmission intervals
- Calculated average network overhead introduced by the defense

### 6. Evaluation & Analysis

Comprehensive evaluation conducted in `evaluate_attacks.py`:
- Classification accuracy across all four classifiers
- Semantic distance metrics using trained Doc2Vec models
- Trade-off analysis between attack success rate and defense overhead

## Project Structure

```
code_alexa/
├── attack/
│   ├── trainByJaccard.py, testByJaccard.py
│   ├── trainByBayes.py, testByBayes.py
│   ├── trainByVNGpp.py, testByVNGpp.py
│   ├── trainByAdaBoost.py, testByAdaBoost.py
│   ├── run.py                           # Orchestrate experiments
│   ├── evaluate_attacks.py              # Evaluate results
│   ├── nFoldCrossValidation.py         # 5-fold CV
│   ├── config.py                        # Configuration
│   └── tools.py                         # Utilities
├── defense/
│   └── buflo.py                         # BuFLO defense mechanism
├── doc2vec_model/                       # Trained Doc2Vec models
├── data/                                # Traffic traces (CSV format)
├── result/                              # Final results & accuracy metrics
└── temp_files/                          # Detailed output for each operation
```

## Results

- **Attack Classification Results**: Stored in `result/` folder
  - Accuracy metrics for all four classifiers
  - Cross-validation results
  - Semantic distance analysis

- **Detailed Operation Logs**: `temp_files/` contains complete output for each experiment
  - Individual classifier evaluations
  - Parameter sensitivity analysis
  - Intermediate results and metrics

- **Defense Analysis**: BuFLO overhead calculations
  - Network overhead percentages
  - Accuracy reduction vs. privacy gain trade-offs

## Dependencies

```
scikit-learn    # Machine learning classifiers
gensim          # Doc2Vec models
pandas          # Data manipulation
numpy           # Numerical computing
matplotlib      # Visualization
```

Install dependencies:
```bash
pip install scikit-learn gensim pandas numpy matplotlib
```

## Usage

### Quick Start

1. **Train a classifier**:
   ```bash
   cd attack
   python trainByAdaBoost.py    # Train AdaBoost classifier
   python testByAdaBoost.py     # Evaluate on test set
   ```

2. **Run complete pipeline** (all experiments):
   ```bash
   python run.py
   ```

3. **Apply BuFLO defense**:
   ```bash
   cd ../defense
   python buflo.py
   ```

4. **Evaluate attacks**:
   ```bash
   cd ../attack
   python evaluate_attacks.py
   ```

### Other Classifiers

Replace `AdaBoost` with `Jaccard`, `Bayes`, or `VNGpp` to test different classifiers:
```bash
python trainByJaccard.py
python testByJaccard.py
```

## Data Format

Traffic traces are stored as CSV files with the following columns:
- `index` - Packet sequence number
- `timestamp` - Packet arrival time (seconds)
- `packet_size` - Packet size in bytes
- `direction` - Traffic direction (+1: outgoing, -1: incoming)

Example:
```
index,timestamp,packet_size,direction
1,1234567.890,256,1
2,1234567.891,1500,-1
3,1234567.892,256,1
```

## Configuration

Edit `config.py` to customize:
- **Data paths**: Directory for traffic traces, models, and results
- **Doc2Vec settings**: Vector dimensions, epochs, window size
- **Cross-validation**: Number of folds, random seed
- **Rounding parameters**: Test different rounding values for sensitivity analysis
- **Buffer sizes**: BuFLO padding size options

## Key Findings

- Voice commands can be reliably identified from network traffic patterns
- Different classifiers show varying accuracy-complexity trade-offs
- Doc2Vec parameters (epoch, vector size) impact semantic similarity metrics
- BuFLO defense effectively protects against fingerprinting with quantifiable overhead
- Rounding parameter significantly affects attack performance



