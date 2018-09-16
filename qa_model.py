import time
import logging
import os

import numpy as np
import sys
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from general_utils import Progbar
from data_utils import *
from collections import defaultdict

from attention_wrapper import _maybe_mask_score
from attention_wrapper import *
from tensorflow.python import debug as tf_debug
from tensorflow.python.ops import array_ops


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config


logging.basicConfig(stream = sys.stdout, level=logging.INFO)

# -- A helper function to reverse a tensor along seq_dim
def _reverse(input_, seq_lengths, seq_dim, batch_dim):
  if seq_lengths is not None:
    return array_ops.reverse_sequence(
        input=input_, seq_lengths=seq_lengths,
        seq_dim=seq_dim, batch_dim=batch_dim)
  else:
    return array_ops.reverse(input_, axis=[seq_dim])



class Encoder(object):
    def __init__(self, hidden_size=None, initializer = lambda : None):#tf.contrib.layers.xavier_initializer):
        self.hidden_size = hidden_size or config.HIDDEN_STATE_SIZE
        self.init_weights = initializer


    def encode(self, inputs, masks, encoder_state_input = None):
        """
        :param inputs: vector representations of question and passage (a tuple) 
        :param masks: masking sequences for both question and passage (a tuple)

        :param encoder_state_input: (Optional) pass this as initial hidden state
                                    to tf.nn.dynamic_rnn to build conditional representations
        :return: an encoded representation of the question and passage.
        """


        question, passage, passage_features = inputs
        masks_question, masks_passage = masks    


        # read passage conditioned upon the question
        with tf.variable_scope("encoded_question"):
            lstm_cell_question = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, state_is_tuple = True)
            encoded_question, (q_rep, _) = tf.nn.dynamic_rnn(lstm_cell_question, question, masks_question, dtype=tf.float32) # (-1, Q, H)

        with tf.variable_scope("encoded_passage"):
            lstm_cell_passage  = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, state_is_tuple = True)
            passage_extra = tf.concat((passage, passage_features), 2)
            encoded_passage, (p_rep, _) =  tf.nn.dynamic_rnn(lstm_cell_passage, passage_extra, masks_passage, dtype=tf.float32) # (-1, P, H)


        # outputs beyond sequence lengths are masked with 0s
        return encoded_question, encoded_passage , q_rep, p_rep


class BaselineDecoder(object):
    def __init__(self):
        return

    def decode(self, encoded_passage , q_rep, mask, labels):

        # (batch_size, 1, D),  (batch_size, Q, D)
        input_size = q_rep.get_shape()[-1]
        q_rep1 = tf.layers.dense(q_rep, input_size, name="W1")
        q_rep2 = tf.layers.dense(q_rep, input_size, name="W2")


        q_rep1 = tf.expand_dims(q_rep1, 1)
        logit_1 = tf.reduce_sum(q_rep1*encoded_passage, [2])

        q_rep2 = tf.expand_dims(q_rep2, 1)
        logit_2 = tf.reduce_sum(q_rep2*encoded_passage, [2])

        func = lambda score: _maybe_mask_score(score, mask, float("-inf"))

        return [func(logit_1),func(logit_2)]


   
class Decoder(object):
    def __init__(self, hidden_size=None, initializer= lambda : None):
        self.hidden_size = hidden_size or config.HIDDEN_STATE_SIZE
        self.init_weights = initializer



    def run_lstm(self, encoded_rep, q_rep, masks):
        encoded_question, encoded_passage = encoded_rep
        masks_question, masks_passage = masks

        q_rep = tf.expand_dims(q_rep, 1) # (batch_size, 1, D)
        encoded_passage_shape = tf.shape(encoded_passage)[1]
        q_rep = tf.tile(q_rep, [1, encoded_passage_shape, 1])

        mixed_question_passage_rep = tf.concat([encoded_passage, q_rep], axis=-1)

        with tf.variable_scope("lstm_"):
            cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, state_is_tuple = True)
            reverse_mixed_question_passage_rep = _reverse(mixed_question_passage_rep, masks_passage, 1, 0)

            output_attender_fw, _ = tf.nn.dynamic_rnn(cell, mixed_question_passage_rep, dtype=tf.float32, scope ="rnn")    
            output_attender_bw, _ = tf.nn.dynamic_rnn(cell, reverse_mixed_question_passage_rep, dtype=tf.float32, scope = "rnn")

            output_attender_bw = _reverse(output_attender_bw, masks_passage, 1, 0)

            
        output_attender = tf.concat([output_attender_fw, output_attender_bw], axis = -1) # (-1, P, 2*H)
        return output_attender




    def run_match_lstm(self, encoded_rep, masks):
        encoded_question, encoded_passage = encoded_rep
        masks_question, masks_passage = masks

        match_lstm_cell_attention_fn = lambda curr_input, state : tf.concat([curr_input, state], axis = -1)
        query_depth = encoded_question.get_shape()[-1]


        # output attention is false because we want to output the cell output and not the attention values
        with tf.variable_scope("match_lstm_attender"):
            attention_mechanism_match_lstm = BahdanauAttention(query_depth, encoded_question, memory_sequence_length = masks_question)
            cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, state_is_tuple = True)
            lstm_attender  = AttentionWrapper(cell, attention_mechanism_match_lstm, output_attention = False, attention_input_fn = match_lstm_cell_attention_fn)

            # we don't mask the passage because masking the memories will be handled by the pointerNet
            reverse_encoded_passage = _reverse(encoded_passage, masks_passage, 1, 0)

            output_attender_fw, _ = tf.nn.dynamic_rnn(lstm_attender, encoded_passage, dtype=tf.float32, scope ="rnn")    
            output_attender_bw, _ = tf.nn.dynamic_rnn(lstm_attender, reverse_encoded_passage, dtype=tf.float32, scope = "rnn")

            output_attender_bw = _reverse(output_attender_bw, masks_passage, 1, 0)

        
        output_attender = tf.concat([output_attender_fw, output_attender_bw], axis = -1) # (-1, P, 2*H)
        return output_attender


    def run_answer_ptr(self, output_attender, masks, labels):
        batch_size = tf.shape(output_attender)[0]
        masks_question, masks_passage = masks
        labels = tf.unstack(labels, axis=1) 
        #labels = tf.ones([batch_size, 2, 1])


        answer_ptr_cell_input_fn = lambda curr_input, context : context # independent of question
        query_depth_answer_ptr = output_attender.get_shape()[-1]

        with tf.variable_scope("answer_ptr_attender"):
            attention_mechanism_answer_ptr = BahdanauAttention(query_depth_answer_ptr , output_attender, memory_sequence_length = masks_passage)
            # output attention is true because we want to output the attention values
            cell_answer_ptr = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, state_is_tuple = True )
            answer_ptr_attender = AttentionWrapper(cell_answer_ptr, attention_mechanism_answer_ptr, cell_input_fn = answer_ptr_cell_input_fn)
            logits, _ = tf.nn.static_rnn(answer_ptr_attender, labels, dtype = tf.float32)

        return logits 



    def decode_lstm(self, encoded_rep, q_rep, masks, labels):
        """ 
            Ablation study on match-LSTM (replace match-LSTM with a simple LSTM)
        """
        output_lstm = self.run_lstm(encoded_rep, q_rep, masks)
        logits = self.run_answer_ptr(output_lstm, masks, labels)
        
        return logits



    def decode(self, encoded_rep, q_rep, masks, labels):
        """
        takes in encoded_rep
        and output a probability estimation over
        all paragraph tokens on which token should be
        the start of the answer span, and which should be
        the end of the answer span.

        :param encoded_rep: 
        :param masks
        :param labels


        :return: logits: for each word in passage the probability that it is the start word and end word.
        """

        output_attender = self.run_match_lstm(encoded_rep, masks)
        logits = self.run_answer_ptr(output_attender, masks, labels)
    
        return logits
    




class QASystem(object):    
    def __init__(self, model_name, features=None, encoder=None, decoder=None, pretrained_embeddings=None):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """

        # ==== set up logging ======


        logger = logging.getLogger("QASystemLogger")
        logger.addHandler(logging.FileHandler(os.path.join(config.LOGS_DIR, model_name + '.log')))
        self.model_name = model_name
        self.logger = logger


        # ==== set up placeholder tokens ========
        self.embeddings = pretrained_embeddings or get_trimmed_glove_vectors()
        self.encoder = encoder or Encoder()
        self.decoder = decoder or Decoder()

        if not features:
            features = config.FEATURES
        self.feature_dict = { feature: i for i, feature in enumerate(features) }


        self.setup_placeholders()
        


        # ==== assemble pieces ====
        with tf.variable_scope("qa"):
            self.setup_word_embeddings()
            self.setup_system()
            self.setup_loss()
            self.setup_train_op()
            self.saver = tf.train.Saver()

        



    def setup_train_op(self):
        """
        Add train_op to self
        """
        with tf.variable_scope("train_step"):
            adam_optimizer = tf.train.AdamOptimizer()
            grads, vars = zip(*adam_optimizer.compute_gradients(self.loss))

            clip_val = config.MAX_GRADIENT_NORM
            # if -1 then do not perform gradient clipping
            if clip_val != -1:
                clipped_grads, _ = tf.clip_by_global_norm(grads, config.MAX_GRADIENT_NORM)
                self.global_grad = tf.global_norm(clipped_grads)
                self.gradients = zip(clipped_grads, vars)
            else:
                self.global_grad = tf.global_norm(grads)
                self.gradients = zip(grads, vars)


            self.train_op = adam_optimizer.apply_gradients(self.gradients)

        self.init = tf.global_variables_initializer()


    def get_feed_dict(self, questions, dropout_val):
        """
        -arg questions: A list of list of ids representing the question sentence
        -arg contexts: A list of list of ids representing the context paragraph
        -arg dropout_val: A float representing the keep probability for dropout 

        :return: dict {placeholders: value}
        """

        padded_questions, question_lengths = pad_sequences([q.token_ids for q in questions], 0)
        padded_contexts, passage_lengths = pad_sequences([q.context.token_ids for q in questions], 0)

        passage_features = np.zeros((len(questions), padded_contexts.shape[1], len(self.feature_dict)))
        for i, question in enumerate(questions):
            features = question.vectorize(self.feature_dict)
            passage_features[i, :features.shape[0], :] = features

        feed = {
            self.question_ids : padded_questions,
            self.passage_ids : padded_contexts,
            self.passage_features : passage_features,
            self.question_lengths : question_lengths,
            self.passage_lengths : passage_lengths,
            self.labels : [q.answer_spans[0] for q in questions],
            self.dropout : dropout_val
        }

        return feed


    def setup_word_embeddings(self):
        '''
            Create an embedding matrix (initialised with pretrained glove vectors and updated only if self.config.train_embeddings is true)
            lookup into this matrix and apply dropout (which is 1 at test time and self.config.dropout at train time)
        '''
        with tf.variable_scope("vocab_embeddings"):
            _word_embeddings = tf.Variable(self.embeddings, name="_word_embeddings", dtype=tf.float32, trainable=config.TRAIN_EMBEDDINGS)
            question_emb = tf.nn.embedding_lookup(_word_embeddings, self.question_ids, name = "question") # (-1, Q, D)
            passage_emb = tf.nn.embedding_lookup(_word_embeddings, self.passage_ids, name = "passage") # (-1, P, D)
            # Apply dropout
            self.question = tf.nn.dropout(question_emb, self.dropout)
            self.passage  = tf.nn.dropout(passage_emb, self.dropout)
            



    def setup_placeholders(self):
        self.question_ids = tf.placeholder(tf.int32, shape = [None, None], name = "question_ids")
        self.passage_ids = tf.placeholder(tf.int32, shape = [None, None], name = "passage_ids")

        self.passage_features = tf.placeholder(tf.float32, shape=[None, None, len(self.feature_dict)],
                name="passage_features")

        self.question_lengths = tf.placeholder(tf.int32, shape=[None], name="question_lengths")
        self.passage_lengths = tf.placeholder(tf.int32, shape = [None], name = "passage_lengths")

        self.labels = tf.placeholder(tf.int32, shape = [None, 2], name = "gold_labels")
        self.dropout = tf.placeholder(tf.float32, shape=[], name = "dropout")

    def setup_system(self):
        """
           Apply the encoder to the question and passage embeddings. Follow that up by Match-LSTM and Answer-Ptr 
        """
        encoder = self.encoder
        decoder = self.decoder
        encoded_question, encoded_passage, q_rep, p_rep = encoder.encode(
                [self.question, self.passage, self.passage_features],
                [self.question_lengths, self.passage_lengths],
                encoder_state_input = None)

        if config.USE_MATCH:
            self.logger.info("\n========Using Match LSTM=========\n")
            logits= decoder.decode([encoded_question, encoded_passage], q_rep, [self.question_lengths, self.passage_lengths], self.labels)
        else:
            self.logger.info("\n========Using Vanilla LSTM=========\n")
            logits = decoder.decode_lstm([encoded_question, encoded_passage], q_rep, [self.question_lengths, self.passage_lengths], self.labels)


        self.logits = logits


    def setup_loss(self):
        """
        self.logits are the 2 sets of logit (num_classes) values for each example, masked with float(-inf) beyond the true sequence length
        :return: Loss for the current batch of examples
        """
      
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits[0], labels=self.labels[:,0])
        losses += tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits[1], labels=self.labels[:,1])
        self.loss = tf.reduce_mean(losses)


    def initialize_model(self, session, train_dir=None):
        """
            param: session managed from train.py
            param: train_dir : the directory in which models are saved

        """
        self.train_dir = train_dir or os.path.join(config.MODELS_DIR, self.model_name)

        ckpt = tf.train.get_checkpoint_state(self.train_dir)
        v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
        if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
            self.logger.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            self.saver.restore(session, ckpt.model_checkpoint_path)
        else:
            self.logger.info("Created model with fresh parameters.")
            session.run(self.init)                
            self.logger.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))




    def test(self, session, valid):
        """
        valid: a list of Questions
        :return: loss on the valid dataset and the logit values
        """

        # at test time we do not perform dropout.
        input_feed =  self.get_feed_dict(valid, 1.0)

        output_feed = [self.logits]

        outputs = session.run(output_feed, input_feed)

        return outputs[0][0], outputs[0][1]


    def answer(self, session, dataset):
        '''
            Get the answers for dataset. Independent of how data iteration is implemented
        '''

        answer_batches = []
        for batch in minibatches(dataset, config.BATCH_SIZE):
            answer_batches.append(self.test(session, batch))

        # -- Boundary Model with a max span restriction of 15
        def best_span(y1, y2):
            max_ans = -999999
            a_s, a_e= 0,0
            num_classes = len(y1)
            for i in xrange(num_classes):
                for j in xrange(15):
                    if i+j >= num_classes:
                        break

                    curr_a_s = y1[i];
                    curr_a_e = y2[i+j]
                    if (curr_a_e+curr_a_s) > max_ans:
                        max_ans = curr_a_e + curr_a_s
                        a_s = i
                        a_e = i+j

            return (a_s, a_e)


        a_s, a_e = [], []
        for batch in answer_batches:
            for start, end in zip(*batch):
                _a_s, _a_e = best_span(start, end)
                a_s.append(_a_s)
                a_e.append(_a_e)
 

        return (np.array(a_s), np.array(a_e))


    def answer_many_contexts(self, session, dataset, best_n=1):
        answer_batches = []
        for batch in minibatches(dataset, config.BATCH_SIZE):
            answer_batches.append(self.test(session, batch))

        spans = defaultdict(lambda: (0, ''))
        index = 0
        for batch in answer_batches:
            for start, end in zip(*batch):
                context = dataset[index].context
                num_classes = len(context.tokens)
                for i in range(num_classes):
                    for j in range(15):
                        if i+j >= num_classes:
                            break
                        confidence = start[i] + end[i + j]
                        tokens = context.tokens[i:i+j+1]
                        answer = ' '.join(tokens)
                        if confidence > spans[answer][0]:
                            spans[answer] = (confidence, context.doc_name)

                index += 1

        spans = [(ans, conf, doc) for ans, (conf, doc) in spans.items()]
        spans.sort(key=lambda pair: -pair[1])
        return spans[:best_n]


    def evaluate_model(self, session, dataset):
        """

    
        :param session: session should always be centrally managed in train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :return: exact match scores
        """

        sample = len(dataset)
        a_s, a_o = self.answer(session, dataset)
        answers = np.hstack([a_s.reshape([sample, -1]), a_o.reshape([sample,-1])])
        gold_answers = np.array([q.answer_spans[0] for q in dataset])


        em_score = 0
        em_1 = 0
        em_2 = 0
        intersect = 0
        for i in xrange(sample):
            gold_s, gold_e = gold_answers[i]
            s, e = answers[i]
            if (s==gold_s): em_1 += 1.0
            if (e==gold_e): em_2 += 1.0
            if (s == gold_s and e == gold_e):
                em_score += 1.0
            if s <= gold_e and e >= gold_s:
                intersect += 1.0

        em_score /= float(len(answers))
        em_1 /= float(len(answers))
        em_2 /= float(len(answers))
        intersect /= float(len(answers))
        self.logger.info("\nExact match on 1st token: %5.4f | Exact match on 2nd token: %5.4f" %(em_1, em_2))
        self.logger.info('Intersect match: %5.4f' % intersect)


        return em_score


    def run_epoch(self, session, train):
        """
        Perform one complete pass over the training data and evaluate on dev
        """
     
        nbatches = (len(train) + config.BATCH_SIZE - 1) / config.BATCH_SIZE
        prog = Progbar(target=nbatches)


        for i, q_batch in enumerate(minibatches(train, config.BATCH_SIZE)):

            # at training time, dropout needs to be on.
            input_feed = self.get_feed_dict(q_batch, config.DROPOUT_VAL)

            _, train_loss = session.run([self.train_op, self.loss], feed_dict=input_feed)
            prog.update(i + 1, [("train loss", train_loss)])




    def train(self, session, dataset):
        """
        Implement main training loop

        :param session: it should be passed in from train.py
        :param dataset: a list containing the training and dev data
        :param train_dir: path to the directory where you should save the model checkpoint
        :return:
        """

        if not tf.gfile.Exists(self.train_dir):
            tf.gfile.MkDir(self.train_dir)


        train, dev = dataset

        em = self.evaluate_model(session, dev)
        self.logger.info("\n#-----------Initial Exact match on dev set: %5.4f ---------------#\n" %em)
        #self.logger.info("#-----------Initial F1 on dev set: %5.4f ---------------#" %f1)

        best_em = 0

        for epoch in range(config.NUM_EPOCHS):
            train.shuffle()

            self.logger.info("\n*********************EPOCH: %d*********************\n" %(epoch+1))
            self.run_epoch(session, train)
            em = self.evaluate_model(session, dev)
            self.logger.info("\n#-----------Exact match on dev set: %5.4f #-----------\n" %em)
            #self.logger.info("#-----------F1 on dev set: %5.4f #-----------" %f1)

            #======== Save model if it is the best so far ========
            if (em > best_em):
                print('Improvement, saving to %s/best_model.chk' % self.train_dir)
                self.saver.save(session, "%s/best_model.chk" % self.train_dir)
                best_em = em

