import sys
import requests
import nltk
from nltk.parse.corenlp import CoreNLPServer, CoreNLPParser, GenericCoreNLPParser
import config

class CoreNLPProcessor:
    def __init__(self, url=None, annotators=['ssplit','tokenize']):
        self.url = url or config.CORENLP_URL
        self.url = self.url.format(annotators=','.join(annotators))
        self.extract_lemmas = 'lemma' in annotators
        self.extract_pos = 'pos' in annotators
        self.extract_ner = 'ner' in annotators

    def process(self, sentence):
        tokens = list(self._processed_tokens(sentence))
        processed_data = {}
        processed_data['tokens'] = [token['word'] for token in tokens]
        processed_data['indices'] = [token['characterOffsetBegin'] for token in tokens]
        if self.extract_lemmas:
            processed_data['lemmas'] = [token['lemma'] for token in tokens]
        if self.extract_pos:
            processed_data['pos'] = [token['pos'] for token in tokens]
        if self.extract_ner:
            processed_data['ner'] = [token['ner'] for token in tokens]
        return processed_data


    def _processed_tokens(self, sentence):
        response = requests.post(self.url, data=sentence.encode('utf8'))
        data = response.json()
        for sentence in data['sentences']:
            for token in sentence['tokens']:
                yield token

question_processor = CoreNLPProcessor(annotators=['ssplit','tokenize','lemma','pos','ner'])
context_processor = CoreNLPProcessor(annotators=['ssplit','tokenize','lemma','pos','ner'])
tokenizer = CoreNLPProcessor()

def tokenize(sentence):
    return tokenizer.process(sentence)['tokens']

def server():
    print('Starting CoreNLP server...')
    serv = CoreNLPServer(path_to_jar=config.CORENLP_JAR, path_to_models_jar=config.CORENLP_MODELS_JAR)
    try:
        serv.start()
        print('Server started.')
        while True:
            pass
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(e)
    finally:
        print('Stopping server...')
        serv.stop()

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'server':
        server()
    elif len(sys.argv) > 1 and sys.argv[1] == 'tokenize':
        sentence = ' '.join(sys.argv[2:])
        print(tokenize(sentence))
