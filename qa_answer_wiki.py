import os
import sys

import tensorflow as tf

from document_retriever import DocumentRetriever
from qa_model import QASystem


def predict(model_name):
    qa = QASystem(model_name)

    with tf.Session() as sess:
        qa.initialize_model(sess)
        while True:
            question = input("Ask a question: ")
            for answer, confidence, doc in answer_question(qa, sess, question, best_n=10):
                print('{:.2f}:\t{}    ({})'.format(confidence, answer, doc))

def answer_question(qa, sess, question, best_n=5):
    retriever = DocumentRetriever()
    questions = retriever.retrieve_questions(question)
    return qa.answer_many_contexts(sess, questions, best_n)


if __name__ == "__main__":
    model_name = 'default'
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    predict(model_name)
