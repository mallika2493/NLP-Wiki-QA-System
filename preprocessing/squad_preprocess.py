import argparse
import json
import linecache
import nltk
import numpy as np
import os
from tqdm import tqdm
import random

from collections import Counter
from squad_data import SquadData
from util import maybe_download

import config

random.seed(42)
np.random.seed(42)

squad_base_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/"

# Size train: 30288272
# size dev: 4854279


def data_from_json(filename):
    with open(filename) as data_file:
        data = json.load(data_file)
    return data


def list_topics(data):
    list_topics = [data['data'][idx]['title'] for idx in range(0,len(data['data']))]
    return list_topics


def tokenize(sequence):
    tokens = [token.replace("``", '"').replace("''", '"') for token in nltk.word_tokenize(sequence)]
    return tokens


def token_idx_map(context, context_tokens):
    acc = ''
    current_token_idx = 0
    token_map = dict()

    for char_idx, char in enumerate(context):
        if char != u' ':
            acc += char
            context_token = context_tokens[current_token_idx]
            if acc == context_token:
                syn_start = char_idx - len(acc) + 1
                token_map[syn_start] = [acc, current_token_idx]
                acc = ''
                current_token_idx += 1
    return token_map


def invert_map(answer_map):
    return {v[1]: [v[0], k] for k, v in answer_map.iteritems()}


def read_write_dataset(dataset, tier, prefix):
    """Reads the dataset, extracts context, question, answer,
    and answer pointer in their own file. Returns the number
    of questions and answers processed for the dataset"""
    qn, an = 0, 0
    skipped = 0

    with open(os.path.join(prefix, tier +'.context'), 'w') as context_file,  \
         open(os.path.join(prefix, tier +'.question'), 'w') as question_file,\
         open(os.path.join(prefix, tier +'.answer'), 'w') as text_file, \
         open(os.path.join(prefix, tier +'.span'), 'w') as span_file:

        for articles_id in tqdm(range(len(dataset['data'])), desc="Preprocessing {}".format(tier)):
            article_paragraphs = dataset['data'][articles_id]['paragraphs']
            for pid in range(len(article_paragraphs)):
                context = article_paragraphs[pid]['context']
                # The following replacements are suggested in the paper
                # BidAF (Seo et al., 2016)
                context = context.replace("''", '" ')
                context = context.replace("``", '" ')

                context_tokens = tokenize(context)
                answer_map = token_idx_map(context, context_tokens)

                qas = article_paragraphs[pid]['qas']
                for qid in range(len(qas)):
                    question = qas[qid]['question']
                    question_tokens = tokenize(question)

                    answers = qas[qid]['answers']
                    qn += 1

                    num_answers = range(1)

                    for ans_id in num_answers:
                        # it contains answer_start, text
                        text = qas[qid]['answers'][ans_id]['text']
                        a_s = qas[qid]['answers'][ans_id]['answer_start']

                        text_tokens = tokenize(text)

                        answer_start = qas[qid]['answers'][ans_id]['answer_start']

                        answer_end = answer_start + len(text)

                        last_word_answer = len(text_tokens[-1]) # add one to get the first char

                        try:
                            a_start_idx = answer_map[answer_start][1]

                            a_end_idx = answer_map[answer_end - last_word_answer][1]

                            # remove length restraint since we deal with it later
                            context_file.write(' '.join(context_tokens) + '\n')
                            question_file.write(' '.join(question_tokens) + '\n')
                            text_file.write(' '.join(text_tokens) + '\n')
                            span_file.write(' '.join([str(a_start_idx), str(a_end_idx)]) + '\n')

                        except Exception as e:
                            skipped += 1

                        an += 1

    print("Skipped {} question/answer pairs in {}".format(skipped, tier))
    return qn,an


def preprocess_squad():

    download_prefix = os.path.join("download", "squad")
    data_prefix = os.path.join("data", "squad")

    print("Downloading datasets into {}".format(download_prefix))
    print("Preprocessing datasets into {}".format(data_prefix))

    if not os.path.exists(download_prefix):
        os.makedirs(download_prefix)
    if not os.path.exists(data_prefix):
        os.makedirs(data_prefix)

    train_filename = maybe_download(squad_base_url, config.SQUAD_TRAIN_FILENAME, download_prefix, 30288272)
    train_data = SquadData.load_raw(train_filename)
    train_data.shuffle()
    train_data.save(config.SQUAD_TRAIN_PREFIX)

    dev_filename = maybe_download(squad_base_url, config.SQUAD_DEV_FILENAME, download_prefix, 4854279)
    dev_data = SquadData.load_raw(dev_filename)
    dev_data.shuffle()
    dev_data.save(config.SQUAD_DEV_PREFIX)
