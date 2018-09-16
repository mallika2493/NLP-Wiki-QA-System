GLOVE_BASE_URL = "http://nlp.stanford.edu/data/"
GLOVE_FILENAME = "glove.6B.zip"
GLOVE_DIR = 'download/glove/'
GLOVE_DIM = 300

VOCAB_PATH = 'data/vocab.db'
VOCAB_SCHEMA = 'data/vocab_schema.sql'
VOCAB_TABLE = 'embed_300d'

EMBEDDINGS_PATH = 'data/glove.{}.npz'.format(GLOVE_DIM)

CORENLP_JAR = 'data/corenlp/stanford-corenlp-3.8.0.jar'
CORENLP_MODELS_JAR = 'data/corenlp/stanford-corenlp-3.8.0-models.jar'
CORENLP_URL = 'http://localhost:9000?properties={{"annotators":"{annotators}","outputFormat":"json","tokenize.options":"splitHyphenated=true"}}'

SQUAD_TRAIN_FILENAME = 'train-v1.1.json'
SQUAD_DEV_FILENAME = 'dev-v1.1.json'
SQUAD_TRAIN_PREFIX = 'data/squad/train'
SQUAD_DEV_PREFIX = 'data/squad/dev'

TREC_LOCATION = 'https://raw.githubusercontent.com/brmson/dataset-factoid-curated/master/trec/trecnew-curated.tsv'
TREC_PATH = 'data/trecnew-curated.tsv'

MODELS_DIR = 'models/'
LOGS_DIR = 'logs/'


BATCH_SIZE = 100
DROPOUT_VAL = 1.0
HIDDEN_STATE_SIZE = 150
MAX_GRADIENT_NORM = -1
TRAIN_EMBEDDINGS = False
USE_MATCH = 1
NUM_EPOCHS = 30
DATA_DIR = "data/squad"

TRAIN_SIZE = None
EVAL_SIZE = None

FEATURES = ['match_exact', 'match_iexact', 'match_lemma', 'pos_WP', 'pos_SYM', 'pos_JJR', 'pos_RP',
        'pos_RB', 'pos_$', 'pos_PRP', 'pos_:', "pos_''", 'pos_TO', 'pos_NNP', 'pos_NNS', 'pos_CC',
        'pos_WP$', 'pos_VBG', 'pos_-LRB-', 'pos_VBP', 'pos_-RRB-', 'pos_WDT', 'pos_FW', 'pos_VBD',
        'pos_WRB', 'pos_LS', 'pos_VBZ', 'pos_,', 'pos_PRP$', 'pos_POS', 'pos_#', 'pos_RBR', 'pos_.',
        'pos_NNPS', 'pos_VB', 'pos_MD', 'pos_VBN', 'pos_CD', 'pos_JJ', 'pos_JJS', 'pos_IN',
        'pos_RBS', 'pos_NN', 'pos_``', 'pos_DT', 'pos_PDT', 'pos_EX', 'pos_UH', 'ner_ORGANIZATION',
        'ner_DURATION', 'ner_PERCENT', 'ner_LOCATION', 'ner_PERSON', 'ner_MONEY', 'ner_MISC',
        'ner_O', 'ner_NUMBER', 'ner_TIME', 'ner_DATE', 'ner_ORDINAL', 'tf']

WIKI_ARTICLE_COUNT = 5
WIKI_THREADS = 8
