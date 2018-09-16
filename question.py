from collections import defaultdict
import numpy as np
from vocab import index_words, word_embeddings
import corenlp

class Question:
    def __init__(self, sentence, context, token_ids=None, lemmas=None, pos=None, ner=None,
            answer_spans=[(-1,-1)]):
        if isinstance(sentence, str):
            data = corenlp.question_processor.process(sentence)
            sentence = data['tokens']
            lemmas = data['lemmas']
            pos = data['pos']
            ner = data['ner']
        if not token_ids:
            token_ids = index_words(sentence)

        self.tokens = sentence
        self.context = context
        self.token_ids = token_ids
        self.lemmas = lemmas
        self.pos = pos
        self.ner = ner
        self.answer_spans = answer_spans
        self._embeddings = None
        self._features = None

    @property
    def embeddings(self):
        if not self._embeddings:
            self._embeddings = word_embeddings(self.token_ids)
        return self._embeddings

    def serialize(self):
        return {
                'tokens': self.tokens,
                'token_ids': self.token_ids,
                'lemmas': self.lemmas,
                'pos': self.pos,
                'ner': self.ner,
                'answers': self.answer_spans,
                'context': self.context.context_id,
                }

    @staticmethod
    def deserialize(question_dict, context):
        return Question(question_dict['tokens'], context,
                token_ids=question_dict['token_ids'],
                lemmas=question_dict['lemmas'],
                pos=question_dict['pos'],
                ner=question_dict['ner'],
                answer_spans=question_dict['answers'])

    @staticmethod
    def distribute(question_str, contexts):
        question = Question(question_str, contexts[0])
        return [Question(question.tokens, context,
            token_ids=question.token_ids,
            lemmas=question.lemmas,
            pos=question.pos,
            ner=question.ner) for context in contexts]

    def vectorize(self, feature_dict):
        if self._features is not None:
            return self._features

        features = np.zeros((len(self.context.tokens), len(feature_dict)))

        lower_tokens = { t.lower() for t in self.tokens }

        context_length = float(len(self.context.lemmas))
        term_counts = defaultdict(lambda: 0)
        for lemma in self.context.lemmas:
            term_counts[lemma] += 1
        for lemma in term_counts:
            term_counts[lemma] /= context_length

        for i, (c_token, c_id, c_lemma, c_pos, c_ner) in enumerate(
                zip(self.context.tokens, self.context.token_ids, self.context.lemmas,
                    self.context.pos, self.context.ner)):
            if c_token in self.tokens:
                features[i][feature_dict['match_exact']] = 1.0
            if c_token.lower() in lower_tokens:
                features[i][feature_dict['match_iexact']] = 1.0
            if c_lemma in self.lemmas:
                features[i][feature_dict['match_lemma']] = 1.0

            pos_feature = 'pos_{}'.format(c_pos)
            if pos_feature in feature_dict:
                features[i][feature_dict[pos_feature]] = 1.0

            ner_feature = 'ner_{}'.format(c_ner)
            if ner_feature in feature_dict:
                features[i][feature_dict[ner_feature]] = 1.0

            features[i][feature_dict['tf']] = term_counts[c_lemma]

        self._features = features
        return features
