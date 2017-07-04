# -*- coding: utf-8 -*-
import os
import csv
import time
import json
import datetime
import pickle as pkl
import tensorflow as tf
from tensorflow.contrib import learn

import data_helper
from rnn_lm import rnn_lm

# Show warnings and errors only
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Parameters
# =============================================================================

# Data parameters
tf.flags.DEFINE_string('data_file', '1.csv', 'Data file path')
tf.flags.DEFINE_string('stop_word_file', 'stop_words_ch.txt', 'Stop word file path')
tf.flags.DEFINE_string('language', 'ch', "Language of the data file. You have two choices: ['ch', 'en']")
tf.flags.DEFINE_integer('vocab_size', 0, 'Vocabulary size')
tf.flags.DEFINE_integer('level', 2, '1 for char level, 2 for phrase level')

# Hyperparameters
tf.flags.DEFINE_integer('embedding_size', 128, 'Word embedding size')
tf.flags.DEFINE_integer('hidden_size', 128, 'Number of hidden units in the LSTM cell')  # RNN
tf.flags.DEFINE_integer('num_layers', 3, 'Number of the LSTM cells')  # RNN
tf.flags.DEFINE_integer('keep_prob', 0.5, 'Dropout keep probability')
tf.flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate')
tf.flags.DEFINE_float('l2_reg_lambda', 0.0, 'L2 regularization lambda')
tf.flags.DEFINE_float('grad_norm', 5.0, 'Gradient clipping')

# Training parameters
tf.flags.DEFINE_integer('batch_size', 128, 'Batch size')
tf.flags.DEFINE_integer('num_epochs', 50, 'Number of epochs')
tf.flags.DEFINE_integer('evaluate_every_steps', 1000, 'Evaluate the model after this many steps')
tf.flags.DEFINE_integer('save_every_steps', 1000, 'Save the model after this many steps')
tf.flags.DEFINE_integer('num_checkpoint', 20, 'Number of models to store')

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

# Output files directory
timestamp = str(int(time.time()))
outdir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
if not os.path.exists(outdir):
    os.makedirs(outdir)

# Save flags to file
params = FLAGS.__flags
params_file = open(os.path.join(outdir, 'params.pkl'), 'wb')
pkl.dump(params, params_file)
params_file.close()


# Load data
# =============================================================================

data, w_2_idx = data_helper.load_data(file_path=FLAGS.data_file,
                                      sw_path=FLAGS.stop_word_file,
                                      level=FLAGS.level,
                                      vocab_size=FLAGS.vocab_size,
                                      language=FLAGS.language,
                                      shuffle=False)

FLAGS.vocab_size = len(w_2_idx)

# iterator
batches = data_helper.batch_iter(data, w_2_idx, FLAGS.batch_size, FLAGS.num_epochs)

# Save vocabulary to file
vocab_file = open(os.path.join(outdir, 'vocab.pkl'), 'wb')
pkl.dump(w_2_idx, vocab_file)
vocab_file.close()

# Train
# =============================================================================

with tf.Graph().as_default():
    with tf.Session() as sess:
        lm = rnn_lm(FLAGS, is_training=True)

        # Summaries
        loss_summary = tf.summary.scalar('Loss', lm.cost)

        # Train summary
        train_summary_op = tf.summary.merge_all()
        train_summary_dir = os.path.join(outdir, 'summaries', 'train')
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        saver = tf.train.Saver(max_to_keep=FLAGS.num_checkpoint)

        sess.run(tf.global_variables_initializer())

        def run_step(input_data):
            """Run one step of the training process."""
            input_x, input_y = input_data

            fetches = {'step': lm.global_step,
                       'cost': lm.cost,
                       'final_state': lm.final_state,
                       'train_op': lm.train_op,
                       'summaries': train_summary_op}
            feed_dict = {lm.input_x: input_x,
                         lm.input_y: input_y,
                         lm.keep_prob: FLAGS.keep_prob}

            vars = sess.run(fetches, feed_dict)
            step = vars['step']
            cost = vars['cost']
            summaries = vars['summaries']
            train_summary_writer.add_summary(summaries, step)

            time_str = datetime.datetime.now().isoformat()
            print("{}: step: {}, loss: {:g}".format(time_str, step, cost))

            return cost


        print('Start training ...')
        start = time.time()

        total_cost = 0
        for batch in batches:
            cost = run_step(batch)
            total_cost += cost
            current_step = tf.train.global_step(sess, lm.global_step)

            if current_step % FLAGS.evaluate_every_steps == 0:
                aver_cost = total_cost / FLAGS.evaluate_every_steps
                print('\nAverage cost at step {}: {}'.format(current_step, aver_cost))
                total_cost = 0

            if current_step % FLAGS.save_every_steps == 0:
                save_path = saver.save(sess, os.path.join(outdir, 'model/lm'), current_step)
                print('\nModel saved to {}\n'.format(save_path))

        end = time.time()

        print(('\nRun time: {}'.format(end - start)))
        print('\nAll the files have been saved to {}'.format(outdir))
