import collections
import math
import time
import os

import tensorflow as tf
import numpy as np

from reader import build_dataset

# Only show warning and error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""
Implement skip gram word2vec
    args
        file_path: data file path
        sw_path: stop words file path
        vocab_size: the number of different words
        embedding_size: the number of embedding dimensions
        batch_size: the number of training sentences in one pass
        skip_window: window size, equals to |context| / 2
        learning_rate: learning rate
        num_sampled: the number of negative samples
        train_steps: the number of steps for training
        segmentation: word segmentation
"""
class word2vec(object):
    def __init__(self,
                 file_path,
                 sw_path=None,
                 vocab_size=30000,
                 embedding_size=200,
                 batch_size=1,
                 skip_window=2,
                 learning_rate=0.1,
                 num_sampled=100,
                 train_steps=100000,
                 segmentation=True
                 ):
        # Parameters
        self.batch_size = None
        self.embedding_size = embedding_size
        self.skip_window = skip_window
        self.learning_rate = learning_rate
        self.vocab_size = vocab_size
        self.num_sampled = num_sampled
        self.train_steps = train_steps
        self.batch_size = batch_size

        # data: sentences represented by word indices
        # count: word counts
        # w_2_idx: word to index
        # idx_2_w: index to word
        self.data, self.count, self.w_2_idx, self.idx_2_w = build_dataset(file_path,
                                                                          sw_path,
                                                                          segmentation=segmentation,
                                                                          n_words=self.vocab_size)

        # train records
        self.max_len = 1000
        self.loss_records = collections.deque(maxlen=self.max_len)
        self.num_steps = 0
        self.num_sentences = 0

        self.build_graph()
        self.initializer()

    def build_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.train_inputs = tf.placeholder(tf.int32, shape=[None])
            self.train_labels = tf.placeholder(tf.int32, shape=[None, 1])

            self.embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0))
            embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)

            # NCE parameters
            self.nce_weights = tf.Variable(tf.truncated_normal([self.vocab_size, self.embedding_size],
                                                               stddev=1.0/math.sqrt(self.embedding_size)))
            self.nce_biases = tf.Variable(tf.zeros([self.vocab_size]))

            # Compute NCE loss
            self.loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=self.nce_weights,
                               biases=self.nce_biases,
                               labels=self.train_labels,
                               inputs=embed,
                               num_sampled=self.num_sampled,
                               num_classes=self.vocab_size))

            # Gradient descent to update loss and embeddings
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

            # Initialization
            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

    def initializer(self):
        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init)

    def train(self, raw_input=[]):
        # Generate batch and labels
        batch = []
        labels = []

        for sentence in raw_input:

            for i in range(len(sentence)):
                start = max(0, i - self.skip_window)
                end = min(len(sentence), i + self.skip_window + 1)

                for j in range(start, end):
                    if j == i:
                        continue
                    else:
                        batch.append(sentence[i])
                        labels.append(sentence[j])

        if len(batch) == 0:
            return

        batch = np.array(batch, dtype=np.int32)
        labels = np.array(labels, dtype=np.int32)
        labels = np.reshape(labels, [len(labels), 1])

        feed_dict = {self.train_inputs: batch, self.train_labels: labels}
        _, loss_val = self.sess.run([self.optimizer, self.loss], feed_dict=feed_dict)
        self.loss_records.append(loss_val)

        self.num_sentences += len(raw_input)
        self.num_steps += 1

        if self.num_steps % self.max_len == 0:
            average_loss = np.mean(self.loss_records)
            print('Average loss at step', self.num_steps, ': ', average_loss)

    def run_epoch(self):
        index = 0
        print('Start training ...')
        start = time.time()
        for i in range(self.train_steps):
            raw_input = []
            for j in range(self.batch_size):
                raw_input.append(self.data[index])
                index += 1
                index %= len(self.data)
            self.train(raw_input=raw_input)
        end = time.time()
        print('Run time: ', end - start)

# loss: 7.70
config_1 = {'batch_size': 1, 'train_steps': 100000}
# loss: 7.42
config_2 = {'batch_size': 5, 'train_steps': 100000}
# loss: 6.93
config_2 = {'batch_size': 10, 'train_steps': 100000}
# loss: 6.98
config_3 = {'batch_size': 20, 'train_steps': 100000}
# loss: 5.36
config_4 = {'batch_size': 10, 'train_steps': 200000}

w2v = word2vec(file_path='1.csv',
               sw_path='stop_words.txt',
               vocab_size=30000,
               embedding_size=200,
               batch_size=10,
               skip_window=2,
               learning_rate=0.1,
               num_sampled=100,
               train_steps=200000,
               segmentation=True)

w2v.run_epoch()
