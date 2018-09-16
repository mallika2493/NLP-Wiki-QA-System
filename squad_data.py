import json
import random
from tqdm import tqdm
from context import Context
from question import Question

class SquadData:
    def __init__(self, questions=[], contexts={}):
        self.questions = questions
        self.skipped_questions = 0
        self.contexts = contexts or { question.context.context_id: question.context
                for question in self.questions }
        #self._build_feature_dict()

    def __len__(self):
        return len(self.questions)

    def __iter__(self):
        return iter(self.questions)

    def _load_json(self, filename):
        print('Preprocessing dataset from {}'.format(filename))
        with open(filename, 'r') as f:
            data = json.load(f)
        for dataset in tqdm(data['data']):
            for paragraph in tqdm(dataset['paragraphs']):
                self._add_paragraph(paragraph)
        print('\nSkipped {} questions'.format(self.skipped_questions))

    def _add_paragraph(self, paragraph):
        context = Context(paragraph['context'])
        self.contexts[context.context_id] = context
        for qa in paragraph['qas']:
            try:
                self._create_question(qa, context)
            except ValueError:
                self.skipped_questions += 1

    def _create_question(self, question_dict, context):
        question_text = question_dict['question']
        answer_spans = [context.answer_span(**answer_dict)
                for answer_dict in question_dict['answers']]
        self.questions.append(Question(question_text, context, answer_spans=answer_spans))

    def shuffle(self):
        random.shuffle(self.questions)
        return self

    def split(self, ratio):
        split_at = int(len(self.questions) * ratio)
        return (SquadData(self.questions[:split_at]), SquadData(self.questions[split_at:]))

    def save(self, prefix):
        with open(prefix + '-questions.json', 'w') as f:
            f.write('[')
            num_questions = len(self.questions)
            # Write on separate lines for readability
            for i, question in enumerate(self.questions):
                json.dump(question.serialize(), f)
                if i < num_questions - 1:
                    f.write(',\n')
            f.write(']')

        with open(prefix + '-contexts.json', 'w') as f:
            f.write('{ ')
            num_contexts = len(self.contexts)
            for i, (context_id, context) in enumerate(self.contexts.items()):
                f.write('"{cid}": '.format(cid=context_id))
                json.dump(context.serialize(), f)
                if i < num_contexts - 1:
                    f.write(',\n')
            f.write('}')

    def _build_feature_dict(self):
        features = ['match_exact', 'match_iexact', 'match_lemma']
        pos_tags = set()
        ner_tags = set()
        for context in self.contexts.values():
            for pos_tag in context.pos:
                pos_tags.add('pos_{}'.format(pos_tag))
            for ner_tag in context.ner:
                ner_tags.add('ner_{}'.format(ner_tag))

        features.extend(pos_tags)
        features.extend(ner_tags)

        features.append('tf')

        self.feature_dict = {}
        for i, feature in enumerate(features):
            self.feature_dict[feature] = i
        return self.feature_dict

    @staticmethod
    def load_raw(filename):
        data = SquadData()
        data._load_json(filename)
        return data

    @staticmethod
    def load(prefix, size=None):
        with open(prefix + '-contexts.json', 'r') as f:
            contexts = { context_id: Context.deserialize(context_dict)
                    for context_id, context_dict in json.load(f).items() }
        with open(prefix + '-questions.json', 'r') as f:
            questions = [Question.deserialize(q, context=contexts[q['context']])
                    for q in json.load(f)]
        if size:
            questions = questions[:size]
        return SquadData(questions, contexts)
