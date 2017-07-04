import os
import numpy as np
import pickle as pkl
import tensorflow as tf

from rnn_lm import rnn_lm

# Show warnings and errors only
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.flags.DEFINE_string('file_path', None, 'File path')
tf.flags.DEFINE_string('model_file', None, 'Model file')
tf.flags.DEFINE_integer('num_sentences', 100, 'Number of sentences to generate')

FLAGS = tf.flags.FLAGS

# Restore parameters
# ============================================================

# load vocabulary
vocab_file = open(os.path.join(FLAGS.file_path, 'vocab.pkl'), 'rb')
w_2_idx = pkl.load(vocab_file)
idx_2_w = dict(zip(w_2_idx.values(), w_2_idx.keys()))
vocab_file.close()

# load hyperparameters
params_file = open(os.path.join(FLAGS.file_path, 'params.pkl'), 'rb')
params = pkl.load(params_file)
params_file.close()

# initialize config with loaded hyperparameters
class config():
    vocab_size = len(w_2_idx)
    learning_rate = params['learning_rate']
    embedding_size = params['embedding_size']
    hidden_size = params['hidden_size']
    num_layers = params['num_layers']
    batch_size = 1
    l2_reg_lambda = params['l2_reg_lambda']
    grad_norm = params['grad_norm']
    keep_prob = 1.0


# Generate sentences
# ============================================================

def generate_sentence():
    def gen_word(probs):
        t = np.cumsum(probs)
        s = np.sum(probs)
        sample = int(np.searchsorted(t, np.random.rand(1) * s))
        return idx_2_w[sample]

    with tf.Session() as sess:
        lm = rnn_lm(config, is_training=False)
        sess.run(tf.global_variables_initializer())

        # Restore model
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(FLAGS.file_path, 'model/', FLAGS.model_file))

        state = sess.run(lm.cell.zero_state(config.batch_size, tf.float32))

        x = np.array([[w_2_idx['<START>']]])
        feed_dict = {lm.input_x: x,
                     lm.initial_state: state,
                     lm.keep_prob: config.keep_prob}
        [preds, state] = sess.run([lm.predictions, lm.final_state],
                                  feed_dict=feed_dict)
        word = gen_word(preds)

        sentence = ''
        while word != '<END>':
            sentence += word
            x = np.zeros((1, 1))
            x[0, 0] = w_2_idx[word]
            feed_dict = {lm.input_x: x,
                         lm.initial_state: state,
                         lm.keep_prob: config.keep_prob}
            [preds, state] = sess.run([lm.predictions, lm.final_state],
                                      feed_dict=feed_dict)
            word = gen_word(preds)

            if len(sentence) > 50:
                break

        return sentence

if __name__ == '__main__':
    for i in range(FLAGS.num_sentences):
        print(generate_sentence())
        tf.get_variable_scope().reuse_variables()
