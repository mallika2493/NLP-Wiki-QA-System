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

    def _load_json(self, filename):
        print('Preprocessing dataset from {}'.format(filename))
        with open(filename, 'r') as f:
            data = json.load(f)
        for dataset in tqdm(data['data'])[:1]:
            for paragraph in tqdm(dataset['paragraphs']):
                self._add_paragraph(paragraph)

    def _add_paragraph(self, paragraph):
        context = Context(paragraph['context'])
        self.contexts[context.context_id] = context
        for qa in paragraph['qas']:
            self._create_question(qa, context)

    def _create_question(self, question_dict, context):
        question_text = question_dict['question']
        answer_spans = [context.answer_span(**answer_dict)
                for answer_dict in question_dict['answers']]
        self.questions.append(Question(question_text, context, answer_spans=answer_spans))
        #try:
        #    answer_spans = [context.answer_span(**answer_dict)
        #            for answer_dict in question_dict['answers']]
        #    self.questions.append(Question(question_text, context, answer_spans=answer_spans))
        #except Exception as e:
        #    self.skipped_questions += 1

    def shuffle(self):
        random.shuffle(self.questions)
        return self

    def split(self, ratio):
        split_at = int(len(self.questions) * ratio)
        return (SquadData(self.questions[:split_at]), SquadData(self.questions[split_at:]))

    def save(self, prefix):
        with open(prefix + '-questions.json', 'w') as f:
            json.dump([question.serialize() for question in self.questions], f)

        with open(prefix + '-contexts.json', 'w') as f:
            json.dump({ context_id: context.serialize()
                for context_id, context in self.contexts.items() }, f)

    @staticmethod
    def load_raw(filename):
        data = SquadData()
        data._load_json(filename)
        return data

    @staticmethod
    def load(prefix):
        with open(prefix + '-contexts.json', 'r') as f:
            contexts = { context_id: Context.deserialize(context_dict)
                    for context_id, context_dict in json.load(f).items() }
        with open(prefix + '-questions.json', 'r') as f:
            questions = [Question.deserialize(q, context=contexts[q['context']])
                    for q in json.load(f)]
        return SquadData(questions, contexts)
