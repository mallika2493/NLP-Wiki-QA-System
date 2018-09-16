import uuid
from vocab import index_words, word_embeddings
import corenlp

class Context:
    def __init__(self, context, token_ids=None, lemmas=None, pos=None, ner=None, answer_spans=None,
            context_id=None, doc_name=None):
        if isinstance(context, str):
            original_context = context
            self.original_context = context
            data = corenlp.context_processor.process(context)
            context = data['tokens']
            lemmas = data['lemmas']
            pos = data['pos']
            ner = data['ner']
            self.token_indices = data['indices']
            #self._build_token_indices(original_context, context)

        self.tokens = context
        self.token_ids = token_ids or index_words(context)
        self.lemmas = lemmas
        self.pos = pos
        self.ner = ner
        self.context_id = context_id or str(uuid.uuid4())
        self._embeddings = None
        self.doc_name = doc_name

    def answer_span(self, text, answer_start):
        answer_tokens = corenlp.tokenize(text)
        for i, (token, token_index) in enumerate(zip(self.tokens, self.token_indices)):
            if token_index >= answer_start and token == answer_tokens[0]:
                if self.tokens[i:i+len(answer_tokens)] == answer_tokens:
                    return (i, i + len(answer_tokens) - 1)
        raise ValueError('Answer not found in tokenized context')

    def _build_token_indices(self, context, tokens):
        index = 0
        self.token_indices = []
        for token in tokens:
            try:
                index = context.index(token, index)
            except ValueError:
                pass
            self.token_indices.append(index)
            index += len(token)

    @property
    def embeddings(self):
        if not self._embeddings:
            self._embeddings = word_embeddings(self.token_ids)
        return self._embeddings

    def serialize(self):
        return {
                'id': self.context_id,
                'tokens': self.tokens,
                'token_ids': self.token_ids,
                'lemmas': self.lemmas,
                'pos': self.pos,
                'ner': self.ner,
                }

    @staticmethod
    def deserialize(context_dict):
        return Context(context_dict['tokens'],
                token_ids=context_dict['token_ids'],
                context_id=context_dict['id'],
                lemmas=context_dict['lemmas'],
                pos=context_dict['pos'],
                ner=context_dict['ner'])
