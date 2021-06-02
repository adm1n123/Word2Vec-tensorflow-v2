import io
import random
import sys

import nltk
import numpy as np

from nltk.corpus import brown, gutenberg, webtext, reuters, stopwords

nltk.download("brown")
nltk.download("gutenberg")
nltk.download("webtext")
nltk.download("reuters")

nltk.download('punkt')
nltk.download('stopwords')

class CorpusProcessor:

    def __init__(self, train_size, min_count):
        self.vocab = None
        self.inv_vocab = None
        # self.NON_WORD = '[PAD]'
        self.UNKNOWN_SYMBOL = '[UNK]'   # UNK should be indexed at 0 since it is highest frequency word.(after filtering based on frequency)
        self.vocab_size = None
        self.sampling_table = None
        self.TRAIN_SIZE = train_size
        self.MIN_COUNT = min_count
        self.words = None
        self.freq_sum = None

    def idx2word(self, idx):
        return self.inv_vocab[idx]

    def word2idx(self, word):
        unk_idx = self.vocab[self.UNKNOWN_SYMBOL]
        return self.vocab.get(word, unk_idx)  # if word is not in vocabulary return unknown index.

    def get_train_seqs(self):
        rmv_words = self.prepare_vocab()  # these words should be removed from training data.
        train_seqs = []  # list of sentences. each sentence is list of word idx.
        print('Getting training sequences from corpora')
        seq_count = 0

        for sents in [self.get_corp_sents()]:
            for sent in sents:
                words = [word.lower() for word in sent if not rmv_words.get(word.lower(), False)]
                ids = list(map(self.word2idx, words))
                train_seqs.append(ids)
                seq_count += 1

        # print(f'Brown sents {len(brown.sents())}, gutenberg sents: {len(gutenberg.sents())}, webtext sents {len(webtext.sents())}, Total sentences in corpus:{seq_count}')
        random.shuffle(train_seqs)
        train_seqs = train_seqs[:self.TRAIN_SIZE]
        print("Number of train_sequences(after slicing): ", len(train_seqs))
        return train_seqs

    def prepare_vocab(self):
        word_freq = self.get_words_freq()  # creating dictionary
        stop_words = stopwords.words('english')
        rmv_words = {word: True for word in stop_words}
        print('number of english stopwords', len(stop_words))

        for stop_w in stop_words:  # remove stopwords
            word_freq.pop(stop_w, None)

        word_freq = {word: freq for word, freq in word_freq.items() if freq >= self.MIN_COUNT}  # remove rare words
        self.words = []
        for item in word_freq.items():
            word, freq = item
            self.words.append(Vocab(word=word, freq=freq))

        self.words.sort(key=lambda obj: obj.freq, reverse=True)  # sort the vocab with decreasing frequency most frequent words put in the beginning.

        for idx, obj in enumerate(self.words):  # set the word index
            obj.idx = idx

        dump_vocab(self.words)
        self.vocab_size = len(self.words)
        print(f"Vocabulary size(freq >={self.MIN_COUNT}) : {self.vocab_size}")

        self.vocab = {obj.word: obj.idx for _, obj in enumerate(self.words)}
        self.inv_vocab = {obj.idx: obj.word for _, obj in enumerate(self.words)}
        self.freq_sum = 0
        for obj in self.words:  # cal sum of frequencies
            if obj.word == self.UNKNOWN_SYMBOL:
                continue
            self.freq_sum += obj.freq
        return rmv_words

    def get_words_freq(self):
        # TODO: remove non words heuristically eg. no. of alphabets < len/2   if len > 4, etc.
        word_freq = {self.UNKNOWN_SYMBOL: sys.maxsize}

        words = brown.words()
        words = [word.lower() for word in words]
        print(f"brown corpus words: {len(words)}, unique {len(set(words))}")
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1

        # words = gutenberg.words()
        # words = [word.lower() for word in words]
        # print(f"gutenberg corpus words: {len(words)}, unique {len(set(words))}")
        # for word in words:
        #     word_freq[word] = word_freq.get(word, 0) + 1
        #
        # words = webtext.words()
        # words = [word.lower() for word in words]
        # print(f"webtext corpus words: {len(words)}, unique {len(set(words))}")
        # for word in words:
        #     word_freq[word] = word_freq.get(word, 0) + 1

        # words = custm_corp_words()
        # words = [word.lower() for word in words]
        # print(f"custom corpus words: {len(words)}, unique {len(set(words))}")
        # for word in words:
        #     word_freq[word] = word_freq.get(word, 0) + 1

        return word_freq

    def get_corp_sents(self):
        sents = brown.sents()
        # sents.extend(gutenberg.sents())
        # sents.extend(webtext.sents())
        # sents.extend(self.my_corpus_sents())
        # sents = custm_corp_sents()
        return sents

    def get_pre_trained_embeds(self, embedding_dim):
        metadata_file = 'ttdata/pre-trained-metadata-freq-5.tsv'
        vectors_file = 'ttdata/pre-trained-vectors-freq-5.tsv'

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
                old_vec = old_embeddings.get(word, None)
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






class Vocab:
    def __init__(self, word=None, idx=None, freq=None):
        self.word = word # word string
        self.idx = idx  # word id
        self.freq = freq # word freq



def custm_corp_words():
    sents = custm_corp_sents()
    words = []
    for sent in sents:
        if len(sent) <= 1:
            continue
        for word in sent:
            words.append(word)
    return words


def custm_corp_sents():
    with open("data/my_corpus.txt") as f:
        contents = f.readlines()
    sents = [x.strip().lower().split(" ") for x in contents]
    return sents


def list_to_file(mylist, filename):
    with open(filename, 'w') as file:
        for element in mylist:
            file.write('%s\n' % element)


def dict_to_file(mydict, filename):
    with open(filename, 'w') as file:
        for key, value in mydict.items():
            file.write(str(value)+':  '+str(key)+'\n')


def dump_vocab(words):
    with open('data/vocab.txt', 'w') as file:
        for obj in words:
            file.write(f'{obj.idx},     {obj.word}, {obj.freq}\n')

