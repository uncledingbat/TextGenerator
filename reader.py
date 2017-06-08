# -*- coding: utf-8 -*-
import csv
import collections
import re
import sys
import time

import jieba

# Please download langconv.py and zh_wiki.py first
# langconv.py and zh_wiki.py are used for converting between languages
try:
    import langconv
except ImportError as e:
    error = "Please download langconv.py and zh_wiki.py at "
    error += "https://github.com/skydark/nstools/tree/master/zhtools."
    print(str(e) + ': ' + error)
    sys.exit()


"""
Read csv file, delete special characters and return reviews
    Args
        file_path: data file path
        sc_path: special character file path
        segmentation: determine to implement word segmentation or not
    Return
        words: all words in the dataset
        reviews: all reviews in the dataset
"""
def read_data(file_path, sw_path=None, segmentation=True):
    # Python 3.x
    with open(file_path, 'r', encoding='utf-8') as f:
        incsv = csv.reader(f)
        headers = next(incsv)  # Headers
        id = headers.index('content')

        reviews = list()  # A list of reviews
        if sw_path:
            sw = _stop_words(sw_path)
        words = list()  # A list of all the words from reviews
        for line in incsv:
            sent = _tradition_2_simple(line[id].strip())  # Convert traditional chinese to simplified chinese
            if sw_path:
                sent = _clean_data(sent, sw)  # Delete special characters using custom dictionary

            if len(sent) <= 1:
                continue

            if segmentation:
                words_list = _word_segmentation(sent)
                words.extend(words_list)
                reviews.append(words_list)
            else:
                words.extend([word for word in sent])
                reviews.append(sent)

    return words, reviews


"""
Build dataset for later usage
    Args
        n_words: vocabulary size
    Return
        data: A list of sentences consisted of indices
        count: A list of (word, count) pairs
        w_2_idx: Words to indices
        idx_2_w: Indices to words
"""
def build_dataset(file_path, sc_path, segmentation=True, n_words=20000):
    start = time.time()
    print('Building dataset ...')

    words, sentences = read_data(file_path, sc_path, segmentation)

    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    words, _ = zip(*count)
    words = list(words)

    # Map words to indices
    w_2_idx = dict(zip(words, range(len(words))))
    idx_2_w = dict(zip(w_2_idx.values(), w_2_idx.keys()))

    data = []
    UNK_count = 0
    for sentence in sentences:
        temp = []
        for word in sentence:
            if word in words:
                temp.append(w_2_idx[word])
            else:
                temp.append(w_2_idx['UNK'])
                UNK_count += 1
        data.append(temp)
    count[0][1] = UNK_count

    end = time.time()
    runtime = end - start

    print('--------- Summary of the Dataset ---------')
    print('Number of words: ', len(w_2_idx.keys()))
    print('Number of sentences: ', len(data))
    print('Data sample: ', data[:2])
    print('Most common 10 words: ', count[:10])
    print('Words to indices: ', sorted(w_2_idx.items(), key=lambda x: x[1])[:5])
    print('Indices to words: ', sorted(idx_2_w.items(), key=lambda x: x[0])[:5])
    print('Run time: ', runtime)
    print('-------------------------------------------')

    return data, count, w_2_idx, idx_2_w

# --------------- Private Methods ---------------

# Convert Traditional Chinese to Simplified Chinese
def _tradition_2_simple(sent):
    return langconv.Converter('zh-hans').convert(sent)


def _word_segmentation(sent):
    return list(jieba.cut(sent, cut_all=False, HMM=True))


def _stop_words(path):
    with open(path, 'r', encoding='utf-8') as f:
        sw = list()
        for line in f:
            sw.append(line.strip())

    return sw


# Delete stop words using custom dictionary
def _clean_data(sent, sc):
    sent = re.sub('\s+', '', sent)
    sent = re.sub('！+', '！', sent)
    sent = re.sub('。+', '。', sent)
    sent = re.sub('，+', '，', sent)
    sent = "".join([word for word in sent if word not in sc])

    return sent
