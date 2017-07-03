# -*- coding: utf-8 -*-
import re
import os
import sys
import csv
import time
import collections

import jieba
import numpy as np


# Please download langconv.py and zh_wiki.py first
# langconv.py and zh_wiki.py are used for converting between languages
try:
    import langconv
except ImportError as e:
    error = "Please download langconv.py and zh_wiki.py at "
    error += "https://github.com/skydark/nstools/tree/master/zhtools."
    print(str(e) + ': ' + error)
    sys.exit()


def load_data(file_path, sw_path, level=2, vocab_size=0, language='ch', shuffle=True):
    """
    Build dataset for mini-batch iterator
    :param file_path: data file path
    :param sw_path: stop word file path
    :param level: 1 for char level, 2 for phrase level
    :param language: 'ch' for Chinese and 'en' for English
    :param vocab_size: vocabulary size
    :return data: a list of sentences. each sentence is a vector of integers
    :return vocab_processor: Tensorflow VocabularyProcessor object
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        print('Building dataset ...')
        start = time.time()
        incsv = csv.reader(f)
        header = next(incsv)  # Header
        content_idx = header.index('content')

        sentences = []
        words = []

        sw = _stop_words(sw_path)

        for line in incsv:
            sent = line[content_idx].strip()

            if language == 'ch':
                sent = _tradition_2_simple(sent)  # Convert traditional Chinese to simplified Chinese
            elif language == 'en':
                sent = sent.lower()

            sent = _clean_data(sent, sw, language=language)  # Remove stop words and special characters

            if len(sent) < 1:
                continue

            if level == 2:
                if language == 'ch':
                    sent = _word_segmentation(sent)
                word_list = sent.split(' ')
            elif level == 1:
                word_list = [char for char in sent]

            if '' in word_list:
                word_list.remove('')
            sentences.append(word_list)
            words.extend(word_list)

    count = [('<END>', 0), ('<START>', 1), ('<UNK>', 2), ('<PAD>', 3)]
    if vocab_size > 0:
        count.extend(collections.Counter(words).most_common(vocab_size - 4))
    else:
        count.extend(collections.Counter(words).most_common())

    words, _ = zip(*count)
    words = list(words)
    del count

    # Map words to indices
    w_2_idx = dict(zip(words, range(len(words))))

    data = []
    for sentence in sentences:
        temp = [w_2_idx['<START>']]
        for word in sentence:
            if word in words:
                temp.append(w_2_idx[word])
            else:
                temp.append(w_2_idx['<UNK>'])
        temp.append(w_2_idx['<END>'])
        data.append(temp)

    data_size = len(data)
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        data = data[shuffle_indices]

    end = time.time()

    print('Dataset has been built successfully.')
    print('Run time: {}'.format(end - start))
    print('Number of sentences: {}'.format(len(data)))
    print('Vocabulary size: {}'.format(len(w_2_idx)))

    return data, w_2_idx

def batch_iter(data, w_2_idx, batch_size, num_epochs):
    data_size = len(data)
    epoch_length = data_size // batch_size

    for _ in range(num_epochs):
        for i in range(epoch_length):
            start_index = i * batch_size
            end_index = start_index + batch_size

            batch = data[start_index: end_index]
            max_length = max(map(len, batch))

            xdata = np.full((batch_size, max_length), w_2_idx['<PAD>'], np.int32)
            for row in range(batch_size):
                xdata[row, :len(batch[row])] = batch[row]

            ydata = np.copy(xdata)
            ydata[:, :-1] = xdata[:, 1:]

            yield(xdata, ydata)


# Private methods
# =====================================================================

def _tradition_2_simple(sent):
    """ Convert Traditional Chinese to Simplified Chinese """
    return langconv.Converter('zh-hans').convert(sent)


def _word_segmentation(sent):
    """ Tokenizer """
    sent = ' '.join(list(jieba.cut(sent, cut_all=False, HMM=True)))
    return re.sub(r'\s+', ' ', sent)


def _stop_words(path):
    with open(path, 'r', encoding='utf-8') as f:
        sw = list()
        for line in f:
            sw.append(line.strip())

    return set(sw)


def _clean_data(sent, sw, language='ch'):
    """ Remove special characters and stop words """
    if language == 'ch':
        sent = re.sub(r"[^\u4e00-\u9fa5A-z0-9，。！、？]", " ", sent)
        sent = re.sub('\s+', '', sent)
        sent = re.sub('！+', '！', sent)
        sent = re.sub('？+', '！', sent)
        sent = re.sub('。+', '。', sent)
        sent = re.sub('，+', '，', sent)
    if language == 'en':
        sent = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", sent)
        sent = re.sub(r"\'s", " \'s", sent)
        sent = re.sub(r"\'ve", " \'ve", sent)
        sent = re.sub(r"n\'t", " n\'t", sent)
        sent = re.sub(r"\'re", " \'re", sent)
        sent = re.sub(r"\'d", " \'d", sent)
        sent = re.sub(r"\'ll", " \'ll", sent)
        sent = re.sub(r",", " , ", sent)
        sent = re.sub(r"!", " ! ", sent)
        sent = re.sub(r"\(", " \( ", sent)
        sent = re.sub(r"\)", " \) ", sent)
        sent = re.sub(r"\?", " \? ", sent)
        sent = re.sub(r"\s{2,}", " ", sent)
    sent = "".join([word for word in sent if word not in sw])

    return sent

if __name__ == '__main__':
    data, w_2_idx = load_data(file_path='test.csv', sw_path='stop_words_ch.txt', level=1, shuffle=False)
    idx_2_w = dict(zip(w_2_idx.values(), w_2_idx.keys()))
    print("".join([idx_2_w[word] for word in data[0]]))
    print(len(w_2_idx))
    for (xdata, ydata) in batch_iter(data, w_2_idx, 3, 1):
        print(xdata)
        print(ydata)
        break
