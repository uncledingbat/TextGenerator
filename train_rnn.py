import os

import tensorflow as tf
import numpy as np
import reader

# Show warnings and errors only
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class NeuralNetwork():
    def __init__(self,
                 data=None,  # list
                 w_2_idx=None,  # dictionary
                 batch_size=100,  # int
                 vocab_size=30000,
                 learning_rate=0.1,
                 keep_prob=0.5,
                 max_grad_norm=5,
                 hidden_size=128,
                 num_layers=2,
                 max_epoch=50,
                 pretrained_model_path=None,  # str
                 ):
        # assert (type(data) == list)
        self.data = data
        self.w_2_idx = w_2_idx
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self.max_grad_norm = max_grad_norm
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_epoch = max_epoch
        self.pretrained_model_path = pretrained_model_path

        if pretrained_model_path is not None:
            self.pretrained_embedding = self.load_pretrained_model()
            self.vocab_size, self.embedding_dim = self.pretrained_embedding.shape

        self.build_graph()
        self.initializer()

    def load_pretrained_model(self):
        pretrained_graph = tf.Graph()
        with pretrained_graph.as_default():
            saver = tf.train.import_meta_graph(self.pretrained_model_path)
            embeddings = pretrained_graph.get_tensor_by_name('embeddings:0')
            norm = pretrained_graph.get_tensor_by_name('norm:0')

        sess = tf.Session(graph=pretrained_graph)
        saver.restore(sess, tf.train.latest_checkpoint('model/'))
        pretrained_model = sess.run(embeddings / norm)

        return pretrained_model

    def build_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size, None])
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, None])

            # Define lstm cell
            cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size,
                                                forget_bias=0.0,
                                                state_is_tuple=True,
                                                reuse=tf.get_variable_scope().reuse)
            cell = tf.contrib.rnn.MultiRNNCell([cell] * self.num_layers, state_is_tuple=True)

            # Initialize cell state
            self.initial_state = cell.zero_state(self.batch_size, tf.float32)

            # Word embeddings
            if self.pretrained_model_path is not None:
                embedding = tf.get_variable('embedding', initializer=self.pretrained_embedding)
            else:
                embedding = tf.get_variable('embedding', [self.vocab_size, self.hidden_size], dtype=tf.float32, trainable=True)
            inputs = tf.nn.embedding_lookup(embedding, self.train_inputs)

            # Dropout
            if self.keep_prob < 1:
                inputs = tf.nn.dropout(inputs, self.keep_prob)

            # Dynamic RNN
            state = self.initial_state
            with tf.variable_scope('RNN'):
                outputs, state = tf.nn.dynamic_rnn(cell, inputs, initial_state=state)

            # Softmax output layer
            output = tf.reshape(outputs, [-1, self.hidden_size])
            softmax_w = tf.get_variable('softmax_w', [self.hidden_size, self.vocab_size], dtype=tf.float32)
            softmax_b = tf.get_variable('softmax_b', [self.vocab_size], dtype=tf.float32)
            logits = tf.matmul(output, softmax_w) + softmax_b
            targets = tf.reshape(self.train_labels, [-1])
            loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits],
                                                                      [targets],
                                                                      [tf.ones_like(targets, dtype=tf.float32)])

            # Loss
            self.cost = tf.reduce_mean(loss)
            self.final_state = state

            self.lr = tf.Variable(0.0, trainable=False)
            tvars = tf.trainable_variables()
            # Gradient clipping
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), self.max_grad_norm)
            # Gradient descent
            optimizer = tf.train.GradientDescentOptimizer(self.lr)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))

            # Learning rate decay
            self.new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
            self.lr_update = tf.assign(self.lr, self.new_lr)

            self.init = tf.global_variables_initializer()

    def assign_lr(self, session, lr_value):
        session.run(self.lr_update, feed_dict={self.new_lr: lr_value})

    def initializer(self):
        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init)

    def generate_batches(self):
        n_chunk = len(self.data) // self.batch_size
        x_batches = []
        y_batches = []
        sequence_length = []
        for i in range(n_chunk):
            start_index = i * self.batch_size
            end_index = start_index + self.batch_size

            batches = self.data[start_index: end_index]

            length = list(map(len, batches))  # Sequence length
            max_length = max(map(len, batches))

            xdata = np.full((self.batch_size, max_length), self.w_2_idx['。'], np.int32)

            for row in range(self.batch_size):
                xdata[row, :len(batches[row])] = batches[row]

            ydata = np.copy(xdata)
            ydata[:, :-1] = xdata[:, 1:]

            x_batches.append(xdata)
            y_batches.append(ydata)
            sequence_length.append(length)

        return x_batches, y_batches, sequence_length

rnn = NeuralNetwork()

