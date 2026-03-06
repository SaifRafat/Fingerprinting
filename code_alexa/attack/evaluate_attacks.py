"""
Evaluate all attack algorithms with 5-fold cross-validation.

Generates a results table with:
- Accuracy
- Semantic Distance (SD) for Quora doc2vec model
- Normalized SD ranking

Based on paper: "I Can Hear Your Alexa: Voice Command Fingerprinting on Smart Home Speakers"
"""

import os
import tempfile
import shutil
import numpy as np
import pandas as pd
import gensim.models as g
from gensim.utils import simple_preprocess
from sklearn.preprocessing import normalize
from sklearn.model_selection import StratifiedKFold

import config
import nFoldCrossValidation
import testByJaccard
import testByBayes
import testByVNGpp
import testByAdaBoost
import trainByJaccard
import trainByBayes
import trainByVNGpp
import trainByAdaBoost


class SemanticEvaluator:
    """Calculate semantic distance metrics using doc2vec models."""
    
    def __init__(self, quora_model_path):
        """Load doc2vec model from path."""
        print(f"Loading Quora doc2vec model from {quora_model_path}...")
        self.model = g.Doc2Vec.load(quora_model_path)
        self.model.min_alpha = self.model.alpha
        
        self._vector_cache = {}  # Cache for precomputed vectors
        print("Doc2vec model loaded successfully!\n")
    
    def precompute_vectors(self, prior_set):
        """Precompute command vectors for faster semantic distance evaluation."""
        cache_key = "vectors"
        
        if cache_key not in self._vector_cache:
            print(f"    Precomputing vectors for {len(prior_set)} commands...", end=' ')
            vectors = {}
            for cmd in prior_set:
                vectors[cmd] = self._get_vector(cmd)
            self._vector_cache[cache_key] = vectors
            print("done")
        
        return self._vector_cache[cache_key]
    
    def _get_vector(self, text):
        """Get semantic vector for text using doc2vec model."""
        tokens = simple_preprocess(str(text))
        if not tokens:
            tokens = str(text).strip().split()
        return self.model.infer_vector(tokens, alpha=0.01, epochs=1000)


def get_all_labels(data_dir):
    """Extract all unique labels (command names) from data directory."""
    labels = set()
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            label = testByJaccard.getLabel(os.path.join(data_dir, filename))
            labels.add(label)
    return sorted(list(labels))


def evaluate_model(model_name, data_dir, interval, doc2vec_evaluator, n_folds=5):
    """Evaluate model accuracy and semantic distance with cross-validation."""
    print(f"\n{'='*70}")
    print(f"Running {model_name} evaluation ({n_folds}-fold cross-validation)")
    print(f"{'='*70}")
    
    try:
        # Load data
        all_data, all_labels, label_map = nFoldCrossValidation.load_data(
            data_dir, model_name, interval
        )
        
        prior_set = get_all_labels(data_dir)
        
        # Precompute vectors for all commands
        print("  Precomputing semantic vectors...")
        vectors = doc2vec_evaluator.precompute_vectors(prior_set)
        
        accuracies = []
        sd_list = []
        norm_sd_list = []
        
        # 5-fold cross-validation
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        fold_num = 1
        for train_idx, test_idx in skf.split(all_data, all_labels):
            print(f"  Fold {fold_num}/{n_folds}...", end=' ')
            
            # Get training and test data
            if model_name == 'Jaccard':
                train_data = [all_data[i] for i in train_idx]
                test_data = [all_data[i] for i in test_idx]
                train_labels = all_labels[train_idx]
                test_labels = all_labels[test_idx]
                
                print(f"  Training Jaccard on {len(train_data)} files...")
                # Train Jaccard model in a temporary directory
                tmp_dir = tempfile.mkdtemp()
                try:
                    trainByJaccard.train(train_data, tmp_dir)
                    print(f"  Testing Jaccard on {len(test_data)} files...")
                    predictions_str = testByJaccard.test(test_data, tmp_dir)
                    print(f"  Got {len(predictions_str)} predictions")
                finally:
                    shutil.rmtree(tmp_dir)
                
                # Convert string predictions to numeric for accuracy calculation
                predictions = nFoldCrossValidation._strings_to_nums(predictions_str, label_map)
                test_labels_str = nFoldCrossValidation._nums_to_strings(test_labels, label_map)
            
            elif model_name == 'Bayes':
                train_data = all_data[train_idx]
                test_data = all_data[test_idx]
                train_labels = all_labels[train_idx]
                test_labels = all_labels[test_idx]
                
                print(f"  Training Bayes on {len(train_data)} samples...")
                model = trainByBayes.train(train_data, train_labels)
                print(f"  Testing Bayes on {len(test_data)} samples...")
                predictions = testByBayes.test(model, test_data)
                print(f"  Got {len(predictions)} predictions")
                predictions_str = nFoldCrossValidation._nums_to_strings(predictions, label_map)
                test_labels_str = nFoldCrossValidation._nums_to_strings(test_labels, label_map)
            
            elif model_name == 'VNGpp':
                train_data = all_data[train_idx]
                test_data = all_data[test_idx]
                train_labels = all_labels[train_idx]
                test_labels = all_labels[test_idx]
                
                model = trainByVNGpp.train(train_data, train_labels)
                predictions = testByVNGpp.test(model, test_data)
                predictions_str = nFoldCrossValidation._nums_to_strings(predictions, label_map)
                test_labels_str = nFoldCrossValidation._nums_to_strings(test_labels, label_map)
            
            elif model_name == 'AdaBoost':
                train_data = all_data[train_idx]
                test_data = all_data[test_idx]
                train_labels = all_labels[train_idx]
                test_labels = all_labels[test_idx]
                
                model = trainByAdaBoost.train(train_data, train_labels)
                predictions = testByAdaBoost.test(model, test_data)
                predictions_str = nFoldCrossValidation._nums_to_strings(predictions, label_map)
                test_labels_str = nFoldCrossValidation._nums_to_strings(test_labels, label_map)
            
            else:
                raise ValueError(f'Unknown model: {model_name}')
            
            # Calculate accuracy
            accuracy_fold = np.mean(predictions == test_labels)
            accuracies.append(accuracy_fold)
            
            # Calculate semantic distances
            print(f"Computing SD...", end=' ', flush=True)
            
            # Prepare vectors for batch computation
            pred_vecs = np.array([vectors[p] if p in vectors else doc2vec_evaluator._get_vector(p) 
                                   for p in predictions_str])
            actual_vecs = np.array([vectors[a] if a in vectors else doc2vec_evaluator._get_vector(a) 
                                    for a in test_labels_str])
            prior_vecs = np.array([vectors[cmd] for cmd in prior_set])
            
            # Batch compute semantic distances (diagonal of cosine similarity)
            pred_vecs_norm = normalize(pred_vecs)
            actual_vecs_norm = normalize(actual_vecs)
            prior_vecs_norm = normalize(prior_vecs)
            
            # Semantic distance: cosine similarity between prediction and actual
            sd_values = np.sum(pred_vecs_norm * actual_vecs_norm, axis=1)
            sd_list.extend(sd_values.tolist())
            
            # Normalized semantic distance: rank of actual in similarity ranking
            # Compute similarities between all predictions and all prior commands
            similarities = np.dot(pred_vecs_norm, prior_vecs_norm.T)  # (n_pred, n_prior)
            
            for i, actual_label in enumerate(test_labels_str):
                actual_idx = prior_set.index(actual_label)
                sorted_indices = np.argsort(-similarities[i])  # Sort descending
                rank = np.where(sorted_indices == actual_idx)[0][0]
                norm_sd_list.append(int(rank))
            
            print(f"done; acc={accuracy_fold:.3f}")
            fold_num += 1
        
        # Average results across folds
        results = {
            'accuracy': np.mean(accuracies),
            'sd': np.mean(sd_list) if sd_list else 0.0,
            'norm_sd': np.mean(norm_sd_list) if norm_sd_list else 0.0,
        }
        
        print(f"\n✓ {model_name} Complete:")
        print(f"    Accuracy: {results['accuracy']:.1%}")
        print(f"    SD (Quora): {results['sd']:.3f}")
        print(f"    Norm SD (Quora): {results['norm_sd']:.2f}")
        
        return results
    
    except Exception as e:
        print(f"✗ Error evaluating {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run all attack algorithms and generate results table."""
    
    # Path for doc2vec model
    quora_model_path = config.DOC2VEC_QUORA_MODEL
    
    print("\n" + "="*70)
    print("VOICE COMMAND FINGERPRINTING ATTACK EVALUATION")
    print("="*70)
    print(f"Data directory: {config.DATA_DIR}")
    print(f"Quora doc2vec model: {quora_model_path}")
    
    # Check if doc2vec model exists
    if not os.path.exists(quora_model_path):
        print(f"✗ Error: Quora doc2vec model not found at {quora_model_path}")
        print("Please train doc2vec models using batch_train.py first")
        return
    
    # Initialize semantic evaluator
    print("\nInitializing doc2vec model...")
    doc2vec_evaluator = SemanticEvaluator(quora_model_path)
    
    # Evaluation parameters with correct model names
    attacks = [
        {'name': 'Jaccard', 'interval': None},  # Jaccard doesn't use interval
        {'name': 'Bayes', 'interval': config.BAYES_INTERVAL},
        {'name': 'VNGpp', 'interval': config.VNGPP_INTERVAL},
        {'name': 'AdaBoost', 'interval': config.ADABOOST_INTERVAL},
    ]
    
    # Run all attacks
    all_results = {}
    for attack in attacks:
        results = evaluate_model(
            attack['name'], config.DATA_DIR, attack['interval'],
            doc2vec_evaluator, n_folds=5
        )
        if results:
            all_results[attack['name']] = results
    
    # Create results DataFrame
    print("\n" + "="*90)
    print("RESULTS TABLE")
    print("="*90)
    print("\nFormatted as in paper Table I:")
    print("-"*90)
    
    table_data = []
    
    # Map algorithm names to paper names
    name_mapping = {
        'Jaccard': 'LL-Jaccard',
        'Bayes': 'LL-NB',
        'VNGpp': 'VNG++',
        'AdaBoost': 'P-SVM (AdaBoost)',
    }
    
    for model_name, paper_name in name_mapping.items():
        if model_name in all_results:
            row = {
                'Attack': paper_name,
                'Accuracy': f"{all_results[model_name]['accuracy']:.1%}",
                'SD': f"{all_results[model_name]['sd']:.3f}",
                'Norm SD': f"{all_results[model_name]['norm_sd']:.2f}",
            }
            table_data.append(row)
    
    # Add random guess baseline
    num_classes = len(get_all_labels(config.DATA_DIR))
    expected_norm_sd = (num_classes - 1) / 2
    table_data.append({
        'Attack': 'Random Guess',
        'Accuracy': f"{1/num_classes:.1%}",
        'SD': 'N/A',
        'Norm SD': f"{expected_norm_sd:.1f}",
    })
    
    results_table = pd.DataFrame(table_data)
    print(results_table.to_string(index=False))
    
    # Save to CSV and text file
    output_csv = os.path.join(config.RESULT_DIR, 'attack_evaluation_results.csv')
    output_txt = os.path.join(config.RESULT_DIR, 'attack_evaluation_results.txt')
    
    results_table.to_csv(output_csv, index=False)
    print(f"\n✓ CSV results saved to: {output_csv}")
    
    with open(output_txt, 'w') as f:
        f.write("VOICE COMMAND FINGERPRINTING ATTACK EVALUATION RESULTS\n")
        f.write("="*90 + "\n\n")
        f.write(results_table.to_string(index=False))
        f.write(f"\n\nNumber of voice commands (classes): {num_classes}\n")
        f.write("Based on 5-fold cross-validation\n")
        f.write(f"Doc2Vec Model: Quora ({quora_model_path})\n")
        f.write("\nAttack Algorithms:\n")
        f.write("- LL-Jaccard: Jaccard similarity on packet sets\n")
        f.write("- LL-NB: Naive Bayes on packet histogram features\n")
        f.write("- VNG++: Variable n-gram bursts with Naive Bayes\n")
        f.write("- P-SVM (AdaBoost): AdaBoost classifier on traffic features\n")
        f.write("\nMetrics:\n")
        f.write("- Accuracy: Percentage of correct predictions\n")
        f.write("- SD (Semantic Distance): Cosine similarity between predicted and actual command vectors\n")
        f.write("- Norm SD (Normalized SD): Rank of actual command in similarity-sorted results (0=perfect)\n")
    
    print(f"✓ Text results saved to: {output_txt}\n")


if __name__ == '__main__':
    main()
