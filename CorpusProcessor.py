import io
import re
import string
import random

import tensorflow as tf
import tqdm

from tensorflow.keras import Model
from tensorflow.keras.layers import Dot, Embedding, Flatten
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

import nltk
import numpy as np

from nltk.corpus import brown
from nltk.corpus import wordnet

nltk.download("brown")
nltk.download("wordnet")


class CorpusProcessor:

    def __init__(self, train_size):
        self.vocab = None
        self.inv_vocab = None
        self.NON_WORD = '[PAD]'      # non word should be indexed at 0
        self.UNKNOWN_SYMBOL = '[UNK]'   # UNK should be indexed at 1 since it is highest frequency word.(after filtering based on frequency)
        self.vocab_size = None
        self.sampling_table = None
        self.train_sequences = None
        self.TRAIN_SIZE = train_size

    def index2word(self, idx):
        if idx >= self.vocab_size:
            print("word index out of range")
            exit(1)
        return self.inv_vocab[idx]

    def word2index(self, word):
        unk_idx = self.vocab[self.UNKNOWN_SYMBOL]
        return self.vocab.get(word, unk_idx)  # if word is not in vocabulary return unknown index.

    def get_train_data(self):
        print("Brown corpus Number of paragraphs ", len(brown.paras()))

        word_frequency = {}  # creating dictionary
        for para_id, paragraph in enumerate(brown.paras()):  # paragraph is list of sentence and sentence is list of words.
            for sentence in paragraph:
                for Word in sentence:
                    word = Word.lower()
                    word_frequency[word] = word_frequency.get(word, 0) + 1

        # sort the vocab with decreasing frequency most frequent words put in the beginning.
        word_frequency = {word: freq for word, freq in word_frequency.items() if freq >= 5}
        word_sorted = [word for word, _ in sorted(word_frequency.items(), key=lambda item: item[1], reverse=True)]

        word_sorted = [self.NON_WORD, self.UNKNOWN_SYMBOL] + word_sorted
        self.vocab_size = len(word_sorted)

        print("Vocabulary size: ", self.vocab_size)
        # self.list_to_file(vocab, 'data/vocabulary.txt')

        self.vocab = {word: idx for idx, word in enumerate(word_sorted)}
        self.inv_vocab = {idx: word for idx, word in enumerate(word_sorted)}

        # self.dict_to_file(self.vocab, "data/vocab.txt")

        self.train_sequences = []    # list of sentences. each sentence is list of word idx.
        train_count = 0
        for para_id, paragraph in enumerate(brown.paras()):
            for sentence in paragraph:
                words = [word.lower() for word in sentence]
                words_idx = list(map(self.word2index, words))

                self.train_sequences.append(words_idx)
                train_count += 1

        print("Total sentences in brown corpus: %d" % train_count)
        random.shuffle(self.train_sequences)
        self.train_sequences = self.train_sequences[:self.TRAIN_SIZE]
        print("Number of train_sequences(after shuffling): ", len(self.train_sequences))

        return self.train_sequences

    def get_train_data_paragraph(self, sent=None):
        sentence = 'The cat is walking in the bedroom'.split(" ")
        if sent is not None:
            sentence = sent.lower()

        input_data = []
        target_data = []
        for idx, word in enumerate(sentence):
            if idx + self.context_size >= len(sentence):
                break
            context_words = [word.lower() for word in sentence[idx:idx + self.context_size]]
            context_words_idx = list(map(self.word2index, context_words))

            target_word_idx = [self.word2index(sentence[idx + self.context_size].lower())]
            input_data.append(context_words_idx)
            target_data.append(target_word_idx)
        return input_data, target_data

    def add_words(self):
        with open("data/my_corpus.txt") as f:
            contents = f.readlines()
        sentences = [x.strip().lower().split(" ") for x in contents]

        words = set()
        for sentence in sentences:
            if len(sentence) <= 1:
                continue
            for idx, word in enumerate(sentence):
                words.add(word)

        self.vocab = self.vocab.union(words)   # adding frequents words to vocab NOTE: <UNK> is already added so do union.
        self.vocab_size = len(self.vocab)

    def get_train_data_from_sentences(self, sent=None):

        with open("data/my_corpus.txt") as f:
            contents = f.readlines()
        sentences = [x.strip().lower().split(" ") for x in contents]

        input_data = []
        target_data = []
        for sentence in sentences:
            for idx, word in enumerate(sentence):
                if idx + self.context_size >= len(sentence):
                    break
                context_words = [word for word in sentence[idx:idx + self.context_size]]
                context_words_idx = list(map(self.word2index, context_words))

                target_word_idx = [self.word2index(sentence[idx + self.context_size])]
                input_data.append(context_words_idx)
                target_data.append(target_word_idx)
        return input_data, target_data

    def print_words_from_train_data(self, train, target):
        sentences = []
        for i in range(len(train)):
            sentence = []
            for j in range(len(train[i])):
                sentence.append(self.index2word(train[i][j]))
            sentence.append(self.index2word(target[i][0]))
            sentences.append(sentence)
        print(sentences)

    def get_vocab(self):
        vocab = [word for word, idx in self.vocab.items()][1:]  # remove padding word.
        return vocab

    def get_corpus_paras(self):
        return brown.paras()

    def list_to_file(self, mylist, filename):
        with open(filename, 'w') as file:
            for element in mylist:
                file.write('%s\n' % element)

    def dict_to_file(self, mydict, filename):
        with open(filename, 'w') as file:
            for key, value in mydict.items():
                file.write(str(value)+':  '+str(key)+'\n')

