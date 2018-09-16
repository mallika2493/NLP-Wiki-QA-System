import sqlite3
import config

_PAD = '<pad>'
_SOS = '<sos>'
_UNK = '<unk>'
START_VOCAB = [_PAD, _SOS, _UNK]

PAD_ID = 0
SOS_ID = 1
UNK_ID = 2

def index_word(word):
    return index_words([word])

def index_words(words):
    words = [word.lower() for word in words]
    with sqlite3.connect(config.VOCAB_PATH) as conn:
        c = conn.execute('SELECT word, id FROM {table} WHERE word IN ({words})'
                .format(table=config.VOCAB_TABLE, words=','.join('?'*len(words))),
                words)
        d = dict(c.fetchall())
    return [d.get(word) or 0 for word in words]

def word_embeddings(word_ids):
    with sqlite3.connect(config.VOCAB_PATH) as conn:
        c = conn.execute('SELECT id, vector FROM {table} WHERE id IN ({ids})'
                .format(table=config.VOCAB_TABLE, ids=','.join('?'*len(word_ids))),
                word_ids)
        d = dict(c.fetchall())
    return [parse_vector(d.get(word_id)) for word_id in word_ids]

def parse_vector(vec_string):
    if vec_string:
        return [float(s) for s in vec_string.split(' ')]
    else:
        return [0.0] * config.GLOVE_DIM
