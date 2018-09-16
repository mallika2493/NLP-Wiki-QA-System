import os
import sys
import csv
import re

import tensorflow as tf
from tqdm import tqdm

import config
from document_retriever import DocumentRetriever
from qa_model import QASystem

def evaluate(model_name, n=None):
    data = []
    with open(config.TREC_PATH, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in list(reader)[199:]:
            data.append((row[2].strip(), row[3].strip()))

    if not n:
        n = len(data)

    qa = QASystem(model_name)

    top_count = 0
    top_5_count = 0
    top_10_count = 0
    with tf.Session() as sess:
        qa.initialize_model(sess)

        with open(os.path.join(config.MODELS_DIR, model_name, 'trec.csv'), 'w') as f:
            writer = csv.writer(f)
            i = 0
            for question, answer_pattern in tqdm(data[:n]):
                answers = [answer for answer, confidence, doc in answer_question(qa, sess, question, 10)]
                writer.writerow(answers)
                correct = [bool(re.search(answer_pattern, answer)) for answer in answers]
                if True in correct[:1]:
                    top_count += 1
                if True in correct[:5]:
                    top_5_count += 1
                if True in correct[:10]:
                    top_10_count += 1
                i += 1
                print('{}: {}, {}, {}'.format(i, float(top_count) / i, float(top_5_count) / i,
                    float(top_10_count) / i))
    print('Top match: {}'.format(float(top_count) / n))
    print('Top 5 match: {}'.format(float(top_5_count) / n))
    print('Top 10 match: {}'.format(float(top_10_count) / n))

def answer_question(qa, sess, question, best_n):
    retriever = DocumentRetriever()
    questions = retriever.retrieve_questions(question)
    return qa.answer_many_contexts(sess, questions, best_n)

if __name__ == '__main__':
    model_name = 'default'
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    evaluate(model_name)
