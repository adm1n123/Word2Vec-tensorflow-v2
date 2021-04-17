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

    def __init__(self, train_size, min_count):
        self.vocab = None
        self.inv_vocab = None
        self.NON_WORD = '[PAD]'      # non word should be indexed at 0
        self.UNKNOWN_SYMBOL = '[UNK]'   # UNK should be indexed at 1 since it is highest frequency word.(after filtering based on frequency)
        self.vocab_size = None
        self.sampling_table = None
        self.TRAIN_SIZE = train_size
        self.MIN_COUNT = min_count
        self.vocab_freq = None

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
        self.vocab_freq = word_frequency
        word_frequency = {word: freq for word, freq in word_frequency.items() if freq >= self.MIN_COUNT}
        word_sorted = [word for word, _ in sorted(word_frequency.items(), key=lambda item: item[1], reverse=True)]

        word_sorted = [self.NON_WORD, self.UNKNOWN_SYMBOL] + word_sorted
        self.vocab_size = len(word_sorted)

        print("Vocabulary size: ", self.vocab_size)
        # self.list_to_file(vocab, 'data/vocabulary.txt')

        self.vocab = {word: idx for idx, word in enumerate(word_sorted)}
        self.inv_vocab = {idx: word for idx, word in enumerate(word_sorted)}

        self.dict_to_file(self.vocab_freq, "data/vocab.txt")

        train_sequences = []    # list of sentences. each sentence is list of word idx.
        train_count = 0
        for para_id, paragraph in enumerate(brown.paras()):
            for sentence in paragraph:
                words = [word.lower() for word in sentence]
                words_idx = list(map(self.word2index, words))

                train_sequences.append(words_idx)
                train_count += 1

        print("Total sentences in brown corpus: %d" % train_count)
        random.shuffle(train_sequences)
        train_sequences = train_sequences[:self.TRAIN_SIZE]
        print("Number of train_sequences(after shuffling): ", len(train_sequences))

        return train_sequences

    def get_embedding_init(self, embedding_dim):
        metadata_file = 'data/pre-trained-metadata-freq-5.tsv'
        vectors_file = 'data/pre-trained-vectors-freq-5.tsv'

        embeddings = np.random.uniform(-1, 1, (self.vocab_size, embedding_dim))
        hits = 0
        misses = 0
        try:
            meta_fd = io.open(metadata_file, 'r')
            vectors_fd = io.open(vectors_file, 'r')

            vocab = [line.strip() for line in meta_fd.readlines()]
            vectors = [np.fromstring(line, dtype=float, sep='\t') for line in vectors_fd.readlines()]
            if vectors[0].size != embedding_dim:
                raise IndexError

            old_embeddings = dict(zip(vocab, vectors))

            for word, idx in self.vocab.items():
                old_vec = old_embeddings.get(word)
                if old_vec is not None:
                    embeddings[idx] = old_vec
                    hits += 1
                else:
                    misses += 1

            meta_fd.close()
            vectors_fd.close()
            print(f'Using pretrained word embeddings, hits:{hits}, misses:{misses}')
            if hits < 5:
                print('Hits are low not using pretrained embeddings')
                embeddings = None
        except IndexError:
            print('Dimension mismatch could not load pretrained vectors')
            embeddings = None
        except:
            print('could not read pretrained embedding files')
            embeddings = None

        return embeddings

    def get_train_data_paragraph(self, sent=None):
        sentence = 'The cat is walking in the bedroom'.lower().split(" ")
        if sent is not None:
            sentence = sent.lower().split(" ")

        input_data = list(map(self.word2index, sentence))
        return input_data

    def extend_vocab(self):
        with open("data/my_corpus.txt") as f:
            contents = f.readlines()
        sentences = [x.strip().lower().split(" ") for x in contents]

        words = set()
        misses = []
        for sentence in sentences:
            if len(sentence) <= 1:
                continue
            for word in sentence:
                words.add(word)
        for word in words:
            if self.word2index(word) == self.word2index(self.UNKNOWN_SYMBOL):
                misses.append(word)
                self.vocab[word] = self.vocab_size
                self.inv_vocab[self.vocab_size] = word
                self.vocab_size += 1

        print(f'{len(misses)} New words added:{misses}')


    def get_train_data_from_sentences(self, sent=None):

        with open("data/my_corpus.txt") as f:
            contents = f.readlines()
        sentences = [x.strip().lower().split(" ") for x in contents]

        input_data = [list(map(self.word2index, sentence)) for sentence in sentences]

        return input_data



    def list_to_file(self, mylist, filename):
        with open(filename, 'w') as file:
            for element in mylist:
                file.write('%s\n' % element)

    def dict_to_file(self, mydict, filename):
        with open(filename, 'w') as file:
            for key, value in mydict.items():
                file.write(str(value)+':  '+str(key)+'\n')
