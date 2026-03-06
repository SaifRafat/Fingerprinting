"""
Batch inference script for Doc2Vec models
Generate vectors for all trained models in one run
"""
import gensim.models as g
import pandas as pd
import os
from gensim.utils import simple_preprocess
# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def write2file(filename, label, vector_values):
    """Write label and vector to output file."""
    with open(filename, 'a', encoding='utf-8') as f:
        # Write label and vector on first line
        f.write(f'{label},{vector_values}\n')
        # Write #end marker on separate line
        f.write('#end\n')


def refresh_csv(filename):
    """Delete existing output file if it exists."""
    if os.path.exists(filename):
        os.remove(filename)
        print(f'Refreshed {filename}')


def read_queries_csv(filename):
    """Read queries from CSV file. Returns list of query strings."""
    try:
        data = pd.read_csv(filename)
        # Try different column names
        if 'query' in data.columns:
            queries = data['query'].tolist()
        elif len(data.columns) == 1:
            queries = data.iloc[:, 0].tolist()
        else:
            queries = data.iloc[:, 0].tolist()
        return queries
    except Exception as e:
        print(f'Error reading CSV: {e}')
        return []


def read_queries_txt(filename):
    """Read queries from text file (one per line). Returns list of query strings."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            queries = [line.strip() for line in f if line.strip()]
        return queries
    except Exception as e:
        print(f'Error reading text file: {e}')
        return []


def infer_vectors(model_path, 
                  queries, 
                  output_file,
                  start_alpha=0.01,
                  infer_epoch=1000):
    """Infer vectors for queries using trained Doc2Vec model. Returns list of vectors."""
    
    print('\nInferring Document Vectors')
    print(f'Model: {model_path} | Queries: {len(queries)} | Output: {output_file}')
    
    # Load model
    print('Loading model...')
    model = g.Doc2Vec.load(model_path)
    print(f'Model loaded: vector_size={model.vector_size}, vocab_size={len(model.wv)}')
    
    # Set model to inference mode (prevent training updates)
    model.min_alpha = model.alpha
    model.train = lambda *args, **kwargs: None
    
    # Refresh output file (delete if exists)
    refresh_csv(output_file)
    
    # Infer vectors for each query
    vectors = []
    print(f'Inferring vectors for {len(queries)} queries...')
    
    for i, query in enumerate(queries):
        if (i + 1) % 10 == 0:
            print(f'  Processed {i + 1}/{len(queries)} queries...')
        
        # Tokenize query same way as training: using simple_preprocess
        tokens = simple_preprocess(str(query))
        if not tokens:
            # If preprocessing removes all tokens, use original split
            tokens = query.strip().split()
        vector = model.infer_vector(tokens, alpha=start_alpha, epochs=infer_epoch)
        vectors.append(vector)
        
        # Format vector as comma-separated string
        vector_str = ','.join([str(v) for v in vector])
        
        # Write to file: label and vector on one line, #end on next line
        write2file(output_file, query, vector_str)
    
    print(f'Complete! Wrote {len(vectors)} vectors to {output_file}')
    
    return vectors


def simple_infer(model_path, query_text, start_alpha=0.01, infer_epoch=1000):
    """
    Simple inference for a single query or list of queries
    
    Args:
        model_path: Path to trained Doc2Vec model (.bin file)
        query_text: Single query string or list of query strings
        start_alpha: Learning rate for inference (default 0.01)
        infer_epoch: Number of inference iterations (default 1000)
        
    Returns:
        Single vector or list of vectors
    """
    if not os.path.exists(model_path):
        print(f'ERROR: Model not found: {model_path}')
        return None
    
    # Load model
    print(f'Loading model: {model_path}')
    model = g.Doc2Vec.load(model_path)
    
    # Set model to inference mode (prevent training updates)
    model.min_alpha = model.alpha
    model.train = lambda *args, **kwargs: None
    
    # Handle single string or list
    if isinstance(query_text, str):
        tokens = simple_preprocess(str(query_text))
        if not tokens:
            tokens = query_text.strip().split()
        vector = model.infer_vector(tokens, alpha=start_alpha, epochs=infer_epoch)
        return vector
    else:
        vectors = []
        for query in query_text:
            tokens = simple_preprocess(str(query))
            if not tokens:
                tokens = query.strip().split()
            vector = model.infer_vector(tokens, alpha=start_alpha, epochs=infer_epoch)
            vectors.append(vector)
        return vectors


def batch_infer_experiments():
    """
    Generate vectors for all trained Quora models found in quora_models/ directory
    """
    
    # Input queries - from your query list file
    # You can use: queries_list.txt, Quora_dataset.csv, or mannner_train.csv
    input_file = os.path.join(SCRIPT_DIR, 'toy_data/queries_list.txt')  # Update this path to your query file
    model_dir = os.path.join(SCRIPT_DIR, 'yahoo_models')  # Directory where your trained models are saved
    output_dir = os.path.join(SCRIPT_DIR, 'vector_csv/yahoo')
    
    # Inference parameters
    start_alpha = 0.01
    infer_epoch = 1000
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f'ERROR: Input file not found: {input_file}')
        print(f'Please create a query file at: {input_file}')
        print('Format: One query per line (txt) or CSV with query column')
        return
    
    # Check if model directory exists
    if not os.path.exists(model_dir):
        print(f'ERROR: Model directory not found: {model_dir}')
        print('Please run batch_train.py first to train the models')
        return
    
    # Read queries once
    if input_file.endswith('.txt'):
        queries = read_queries_txt(input_file)
    else:
        queries = read_queries_csv(input_file)
    
    print(f'Loaded {len(queries)} queries from {input_file}\n')
    
    if not queries:
        print('ERROR: No queries found in input file')
        return
    
    # Find all .bin model files
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.bin')]
    
    if not model_files:
        print(f'ERROR: No .bin model files found in {model_dir}')
        return
    
    model_files.sort()
    
    print('='*70)
    print(f'Found {len(model_files)} trained models')
    print('='*70)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Infer vectors for each model
    for model_filename in model_files:
        model_path = os.path.join(model_dir, model_filename)
        
        # Generate output filename from model filename
        # e.g., quora_e100_v300.bin -> vectors_queries_quora_e100_v300.csv
        output_filename = model_filename.replace('quora_', 'vectors_queries_quora_').replace('.bin', '.csv')
        output_file = os.path.join(output_dir, output_filename)
        
        print(f'\n>>> Inferring: {model_filename}')
        
        infer_vectors(
            model_path=model_path,
            queries=queries,
            output_file=output_file,
            start_alpha=start_alpha,
            infer_epoch=infer_epoch
        )
    
    print('\n' + '='*70)
    print('ALL INFERENCE COMPLETE!')
    print('='*70)
    print(f'\nGenerated {len(model_files)} vector CSV files in {output_dir}/')
    for model_file in model_files:
        output_filename = model_file.replace('quora_', 'vectors_queries_quora_').replace('.bin', '.csv')
        print(f'  ✓ {output_filename}')


if __name__ == '__main__':
    batch_infer_experiments()
