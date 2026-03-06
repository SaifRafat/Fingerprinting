"""
Batch training script for Doc2Vec models
Trains multiple models with different parameters in one run
"""
import gensim.models as g
import logging
import os
import pandas as pd
from gensim.utils import simple_preprocess

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Configure logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class TaggedLineDocument(object):
    """
    Iterator that yields tagged documents from a text file
    Each line in the file becomes a tagged document
    """
    def __init__(self, filename):
        self.filename = filename
    
    def __iter__(self):
        with open(self.filename, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                # Skip empty lines
                line = line.strip()
                if line:
                    # Split line into words and create TaggedDocument
                    yield g.doc2vec.TaggedDocument(words=line.split(), tags=[i])


class TaggedCSVDocument(object):
    """Iterator yielding tagged documents from CSV file (Yahoo format)."""
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
    
    def __iter__(self):
        doc_id = 0
          
        # Check for Yahoo format (question_title, question_content)
        if 'question_title' in self.df.columns:
            for text in self.df['question_title'].dropna():
                words = simple_preprocess(str(text))
                if words:
                    yield g.doc2vec.TaggedDocument(words=words, tags=[str(doc_id)])
                    doc_id += 1
        


def train_doc2vec(train_corpus, 
                  vector_size=300, 
                  window_size=15, 
                  min_count=1,
                  sampling_threshold=1e-5,
                  negative_size=5,
                  train_epoch=100,
                  dm=0,  # 0=DBOW, 1=DMPV
                  worker_count=1,
                  pretrained_emb=None,
                  saved_path='model.bin'):
    """Train Doc2Vec model with given parameters. Returns trained model."""
    
    print('\nTraining Doc2Vec Model (Yahoo)')
    print(f'Config: vector_size={vector_size}, epochs={train_epoch}, window={window_size}')
    print(f'  Window size: {window_size}')
    print(f'  Training epochs: {train_epoch}')
    print(f'  Model type: {"DBOW" if dm == 0 else "DMPV"}')
    print(f'  Negative sampling size: {negative_size}')
    print(f'  Pretrained embeddings: {pretrained_emb if pretrained_emb else "None"}')
    print('=' * 60)
    
    # Initialize model
    model = g.Doc2Vec(
        vector_size=vector_size,
        window=window_size,
        min_count=min_count,
        sample=sampling_threshold,
        negative=negative_size,
        dm=dm,
        workers=worker_count,
        epochs=train_epoch
    )
    
    # Build vocabulary
    print('Building vocabulary...')
    model.build_vocab(train_corpus)
    print(f'Vocabulary size: {len(model.wv)}')
    
    # Load pretrained embeddings if provided
    if pretrained_emb:
        print(f'Loading pretrained embeddings from {pretrained_emb}...')
        model.wv.intersect_word2vec_format(pretrained_emb, binary=True, lockf=1.0)
    
    # Train model
    print(f'Training model for {train_epoch} epochs...')
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
    
    # Save model
    print(f'Saving model to {saved_path}...')
    model.save(saved_path)
    print('Training complete!')
    
    return model
    


def batch_train_experiments():
    """
    Train multiple Doc2Vec models with different parameter combinations
    using Yahoo dataset
    """
    
    # Common parameters
    window_size = 15
    min_count = 1
    sampling_threshold = 1e-5
    negative_size = 5
    dm = 0  # DBOW mode
    worker_count = 1
    pretrained_emb = None
    
    # Training corpus from CSV file
    csv_file = r'E:\Research work\wenggang\paper\code_alexa\doc2vec_model\toy_data\yahoo\manner_train.csv'
    
    # Create models folder
    models_folder = os.path.join(SCRIPT_DIR, 'yahoo_models')
    os.makedirs(models_folder, exist_ok=True)
    
    print(f'\nUsing CSV file: {csv_file}')
    print(f'Models will be saved to: {models_folder}\n')
    
    # Experiment 1: Vary epochs with fixed vector_size=300
    print('\n' + '='*70)
    print('EXPERIMENT 1: Varying Epochs (vector_size=300)')
    print('='*70)
    
    vector_size = 300
    epochs_list = [100, 125, 150, 175]
    
    for train_epoch in epochs_list:
        saved_path = os.path.join(models_folder, f'yahoo_e{train_epoch}_v{vector_size}.bin')
        
        print(f'\n>>> Training: e{train_epoch}_v{vector_size}')
        train_corpus = TaggedCSVDocument(csv_file)
        
        train_doc2vec(
            train_corpus=train_corpus,
            vector_size=vector_size,
            window_size=window_size,
            min_count=min_count,
            sampling_threshold=sampling_threshold,
            negative_size=negative_size,
            train_epoch=train_epoch,
            dm=dm,
            worker_count=worker_count,
            pretrained_emb=pretrained_emb,
            saved_path=saved_path
        )
    
    # Experiment 2: Vary vector_size with fixed epoch=100
    print('\n' + '='*70)
    print('EXPERIMENT 2: Varying Vector Size (epoch=100)')
    print('='*70)
    
    train_epoch = 100
    vector_sizes = [300, 325, 350, 375]
    
    for vector_size in vector_sizes:
        saved_path = os.path.join(models_folder, f'yahoo_e{train_epoch}_v{vector_size}.bin')
        
        # Skip e100_v300 since it was already trained in Experiment 1
        if train_epoch == 100 and vector_size == 300:
            print(f'\n>>> Skipping: e{train_epoch}_v{vector_size} (already trained)')
            continue
        
        print(f'\n>>> Training: e{train_epoch}_v{vector_size}')
        train_corpus = TaggedCSVDocument(csv_file)
        
        train_doc2vec(
            train_corpus=train_corpus,
            vector_size=vector_size,
            window_size=window_size,
            min_count=min_count,
            sampling_threshold=sampling_threshold,
            negative_size=negative_size,
            train_epoch=train_epoch,
            dm=dm,
            worker_count=worker_count,
            pretrained_emb=pretrained_emb,
            saved_path=saved_path
        )
    
    print('\n' + '='*70)
    print('ALL TRAINING COMPLETE!')
    print('='*70)
    print(f'\nAll models saved to: {models_folder}')
    print('\nTrained models:')
    print('Experiment 1 (varying epochs):')
    for ep in epochs_list:
        print(f'  - yahoo_e{ep}_v300.bin')
    print('Experiment 2 (varying vector_size):')
    for vs in vector_sizes:
        if not (train_epoch == 100 and vs == 300):
            print(f'  - yahoo_e100_v{vs}.bin')


if __name__ == '__main__':
    batch_train_experiments()
