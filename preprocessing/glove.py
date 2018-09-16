import os
import zipfile
import sqlite3
import numpy as np
from tqdm import tqdm

from util import maybe_download
from vocab import START_VOCAB
import config

def preprocess_glove():
    get_glove()
    create_db()
    save_embeddings()

def get_glove():
    prefix = config.GLOVE_DIR

    print("Storing datasets in {}".format(prefix))

    if not os.path.exists(prefix):
        os.makedirs(prefix)

    glove_zip = maybe_download(config.GLOVE_BASE_URL, config.GLOVE_FILENAME, config.GLOVE_DIR,
            862182613)

    if os.path.exists(os.path.join(prefix, 'glove.6B.{}d.txt'.format(config.GLOVE_DIM))):
        return

    print('Unzipping GloVe data')
    glove_zip_ref = zipfile.ZipFile(os.path.join(config.GLOVE_DIR, config.GLOVE_FILENAME), 'r')

    glove_zip_ref.extractall(config.GLOVE_DIR)
    glove_zip_ref.close()

def create_db():
    print('Creating vocab database at {}...'.format(config.VOCAB_PATH))
    if os.path.exists(config.VOCAB_PATH):
        print('Vocab database already exists at {}, skipping'.format(config.VOCAB_PATH))
        return

    with sqlite3.connect(config.VOCAB_PATH) as conn:
        with open(config.VOCAB_SCHEMA, 'r') as f:
            for line in f:
                conn.execute(line)

    print('Populating vocab database at {}...'.format(config.VOCAB_PATH))

    with sqlite3.connect(config.VOCAB_PATH) as conn:
        for i, (word, vector) in enumerate(_glove_lines()):
            conn.execute('INSERT INTO {} (id,word,vector) VALUES (?,?,?)'.format(config.VOCAB_TABLE),
                    (i, word, vector))

def save_embeddings():
    print('Creating embeddings file at {}...'.format(config.EMBEDDINGS_PATH))

    embeddings = np.zeros((400000 + len(START_VOCAB), config.GLOVE_DIM))
    for i, (word, vector) in enumerate(_glove_lines()):
        if vector:
            vector = [float(c) for c in vector.split(' ')]
            assert(len(vector) == config.GLOVE_DIM)
            embeddings[i, :] = vector
    np.savez_compressed(config.EMBEDDINGS_PATH, glove=embeddings)

def _glove_lines(progress=True):
    vocab_size = 400000

    for token in START_VOCAB:
        yield token, None

    with open(os.path.join(
        config.GLOVE_DIR, 'glove.6B.{}d.txt'.format(config.GLOVE_DIM)), 'r') as f:
        it = tqdm(f, total=vocab_size) if progress else f
        for line in it:
            i = line.find(' ')
            word = line[:i]
            vector = line[i+1:]
            yield word, vector
