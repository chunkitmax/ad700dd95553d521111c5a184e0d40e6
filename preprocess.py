'''
preprocess.py

Data preprocessing
'''
import io
import re
from collections import Counter
import numpy as np

EPSILON = 1e-8

class DataManager:
  '''
  Manage data and convert it to bow vectors for the network
  '''
  def __init__(self, file_name, do_cleaning=True, max_vocab_size=None, drop_vocab_max_ratio=0.1):
    print('\rInitializing DataManager...', end='\033[K')
    # Settings
    self.do_cleaning = do_cleaning
    self.max_vocab_size = max_vocab_size
      # Determine the ratio of most freq to least freq words
    self.drop_vocab_max_ratio = drop_vocab_max_ratio

    # Stats
    self.max_len = 0        # Max document length
    self.doc_count = 0      # Number of documents
    self.word_count = 0
    self.input_vec_len = 0

    # Read data from file
    self.docs = []
    self.word_list = []

    self.docs, self.word_list, self.max_len, \
    self.word_count, self.doc_count = self._read_data(file_name)

    print('\nStart building dictionary...', end='')

    # Get word index dictionary
    self.word2idx = {}
    top_10_words = self._build_vocab()
    self.input_vec_len = len(self.word2idx.keys())

    print('\rFinish building dictionary...\n')

    # Print stats
    print("Number of words: ", self.word_count)
    print("Max sentence length: ", self.max_len)
    print("Number of sentences: ", self.doc_count)
    print("Number of vocabulary: ", len(self.word2idx))
    print("Top 10 most frequently words", top_10_words)

    print('\nDataManager initialized!\n')
  
  def _read_data(self, file_name):
    docs = []
    word_list = []

    max_len = 0
    word_count = 0
    doc_count = 0

    with io.open(file_name, "r", encoding="ISO-8859-1") as f:
      next(f)
      for line_no, line in enumerate(f):
        ID, label, sentence = line.split('\t', 2)
        label_idx = int(label == 'pos') # 1 for pos and 0 for neg
        sentence = sentence.strip()

        if self.do_cleaning:
          orig_tweet = self._clean_str(sentence)
        else:
          orig_tweet = sentence.lower()

        docs.append({'y':label_idx, 'txt':orig_tweet})
        word_list += orig_tweet.split()
        cur_sentence_len = len(orig_tweet.split())
        max_len = max(max_len, cur_sentence_len)
        word_count += cur_sentence_len
        doc_count += 1
        print('\r%d lines read'%(line_no + 1), end='\033[K')
    return docs, word_list, max_len, word_count, doc_count

  def extract_feature(self, do_normailzation=False, target_file_name=None):
    '''
    Return bow vectors and corresponding label
    '''
    if target_file_name is None:
      docs, doc_count = self.docs, self.doc_count
    else:
      docs, _, _, _, doc_count = self._read_data(target_file_name)
    ret_data = np.zeros([doc_count, self.input_vec_len])
    ret_label = []
    for doc_index, doc in enumerate(docs):
      counter = Counter(doc['txt'].split())
      for word, count in dict(counter).items():
        if word not in self.word2idx.keys():
          word = '<UNK>'
        ret_data[doc_index, self.word2idx[word]] += float(count)
      ret_label.append([doc['y']])
    if do_normailzation:
      mean = np.mean(ret_data, 1)
      std = np.std(ret_data, 1)
      std[std == 0.] = EPSILON
      ret_data = (ret_data - mean[:, None]) / std[:, None]
    return ret_data, np.array(ret_label)

  def _build_vocab(self):
    '''
    Build word index dictionary
    '''
    counter = Counter(self.word_list)
    words_len = len(counter.items())
    if self.max_vocab_size is not None:
      self.max_vocab_size = int(self.max_vocab_size)
      if self.max_vocab_size < words_len and self.max_vocab_size > 0:
        sorted_words = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        drop_count = words_len - self.max_vocab_size
        drop_most_freq_word_count = int(drop_count * self.drop_vocab_max_ratio)
        sorted_words = sorted_words[drop_most_freq_word_count:
                                    -(drop_count - drop_most_freq_word_count + 1)]
        counter = Counter(dict(sorted_words))
    self.word2idx = {str(key): index+1 for index, key in enumerate(dict(counter).keys())}
    self.word2idx['<UNK>'] = 0 # Unknown word
    top_10_words = [str(x) for x in dict(counter.most_common(10)).keys()]
    return top_10_words

  def _clean_str(self, string):
    '''
    Remove noise from input string
    '''
    ret_str = string
    # Remove HTML entity
    ret_str = re.sub(r'&[a-zA-Z];', '', ret_str)
    # Remove URL
    ret_str = re.sub(r' ?URL ?', ' ', ret_str)
    # Remove punctuation
    ret_str = re.sub(r'[^a-zA-Z0-9\' ]', ' ', ret_str)
    # Remove more than one space
    ret_str = re.sub(r'\ +', ' ', ret_str)
    return ret_str.strip()

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Building Interactive Intelligent Systems')
  parser.add_argument('-f', '--file', help='input csv file', required=False,
                      default='./twitter-sentiment.csv')
  parser.add_argument('-c', '--clean', help='True to do data cleaning, default is False',
                      action='store_true')
  parser.add_argument('-mv', '--max_vocab', help='max vocab size predifined, no limit if set -1',
                      required=False, default=None)
  args = vars(parser.parse_args())

  manager = DataManager(args['file'], args['clean'], args['max_vocab'])
  data, label = manager.extract_feature(True)
  print('Data shape: ', np.shape(data), ', Label shape: ', np.shape(label))
  # print(manager._clean_str('@ lol//ahh..there is really no come back (pardon pun) to that '))

  ## With old implementation:
  # revs, word2idx = data_preprocess(args['file'], args['clean'], int(args['max_vocab']))

  # data, label = feature_extraction_bow(revs, word2idx)
  # data = normalization(data)
