import os
import json
import sys

import tensorflow as tf

import config
from data_utils import get_trimmed_glove_vectors
from squad_data import SquadData
from qa_model import QASystem


def run_func(model_name):
    train = SquadData.load(config.SQUAD_TRAIN_PREFIX, size=config.TRAIN_SIZE)
    dev = SquadData.load(config.SQUAD_DEV_PREFIX, size=config.EVAL_SIZE)

    qa = QASystem(model_name)
    
    with tf.Session() as sess:
        # ====== Load a pretrained model if it exists or create a new one if no pretrained available ======
        qa.initialize_model(sess)
        qa.train(sess, [train, dev])


if __name__ == "__main__":
    model_name = 'default'
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    run_func(model_name)
