# -*- coding: utf-8 -*-
import tensorflow as tf

class rnn_lm(object):
    """"
    A RNN classifier for text classification
    """
    def __init__(self, config, is_training=True):
        self.batch_size = config.batch_size
        self.vocab_size = config.vocab_size
        self.embedding_size = config.embedding_size
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.l2_reg_lambda = config.l2_reg_lambda
        self.grad_norm = config.grad_norm
        self.learning_rate = config.learning_rate

        self.input_x = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, None])
        self.keep_prob = tf.placeholder(dtype=tf.float32, shape=[])
        if is_training:
            self.input_y = tf.placeholder(dtype=tf.int64, shape=[self.batch_size, None])

        # L2 loss
        self.l2_loss = tf.constant(0.0)

        # LSTM Cell
        cell = tf.contrib.rnn.LSTMCell(self.hidden_size,
                                       forget_bias=1.0,
                                       state_is_tuple=True,
                                       reuse=tf.get_variable_scope().reuse)
        # Add dropout to cell output
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

        # Stacked LSTMs
        self.cell = tf.contrib.rnn.MultiRNNCell([cell]*self.num_layers, state_is_tuple=True)

        self.initial_state = self.cell.zero_state(self.batch_size, dtype=tf.float32)

        # Word embedding
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            # embedding = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
            #                         name='embedding')
            # better performance
            embedding = tf.get_variable('embedding',
                                        shape=[self.vocab_size, self.embedding_size],
                                        dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        # Input dropout
        inputs = tf.nn.dropout(inputs, keep_prob=self.keep_prob)

        # Dynamic LSTM
        with tf.variable_scope('LSTM'):
            outputs, state = tf.nn.dynamic_rnn(self.cell,
                                               inputs=inputs,
                                               initial_state=self.initial_state)

        self.final_state = state
        output = tf.reshape(outputs, [-1, self.hidden_size])

        # Softmax output layer
        with tf.name_scope('softmax'):
            softmax_w = tf.get_variable('softmax_w', shape=[self.hidden_size, self.vocab_size], dtype=tf.float32)
            softmax_b = tf.get_variable('softmax_b', shape=[self.vocab_size], dtype=tf.float32)

            # L2 regularization for output layer
            self.l2_loss += tf.nn.l2_loss(softmax_w)
            self.l2_loss += tf.nn.l2_loss(softmax_b)

            self.logits = tf.matmul(output, softmax_w) + softmax_b
            self.predictions = tf.nn.softmax(self.logits)

        if not is_training:
            return

        # Loss
        with tf.name_scope('loss'):
            tvars = tf.trainable_variables()

            # L2 regularization for LSTM weights
            for tv in tvars:
                if 'kernel' in tv.name:
                    self.l2_loss += tf.nn.l2_loss(tv)

            targets = tf.reshape(self.input_y, [-1])
            losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example([self.logits],
                                                                        [targets],
                                                                        [tf.ones_like(targets, dtype=tf.float32)],
                                                                        self.vocab_size)
            self.cost = tf.reduce_mean(losses) + self.l2_reg_lambda * self.l2_loss

        # Training procedure
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), self.grad_norm)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
