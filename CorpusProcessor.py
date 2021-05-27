import io
import random

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
        word_freq = self.get_vocab_words()  # creating dictionary

        stopwords_eng = stopwords.words('english')
        stop_dict = {word: True for word in stopwords_eng}
        print('number of english stopwords', len(stopwords_eng))

        # sort the vocab with decreasing frequency most frequent words put in the beginning.
        self.vocab_freq = word_freq
        word_freq = {word: freq for word, freq in word_freq.items() if freq >= self.MIN_COUNT}
        word_sorted = [word for word, _ in sorted(word_freq.items(), key=lambda item: item[1], reverse=True)]

        word_sorted = [word for word in word_sorted if not stop_dict.get(word, False)]   # removing stopwords from vocab.
        word_sorted = [self.NON_WORD, self.UNKNOWN_SYMBOL] + word_sorted    # try with UNK = NON_WORD
        self.vocab_size = len(word_sorted)

        print(f"Vocabulary size(freq >={self.MIN_COUNT}) : {self.vocab_size}")
        # self.list_to_file(vocab, 'data/vocabulary.txt')

        self.vocab = {word: idx for idx, word in enumerate(word_sorted)}
        self.inv_vocab = {idx: word for idx, word in enumerate(word_sorted)}

        # self.dict_to_file(self.vocab_freq, "data/vocab.txt")

        train_sequences = []  # list of sentences. each sentence is list of word idx.
        print('Getting training sequences from corpora')
        train_count = 0

        for sents in [brown.sents(), gutenberg.sents(), webtext.sents()]:
            for sent in sents:
                words = [word.lower() for word in sent if not stop_dict.get(word.lower(), False)]
                words_idx = list(map(self.word2index, words))

                train_sequences.append(words_idx)
                train_count += 1
        print(f'Brown sents {len(brown.sents())}, gutenberg sents: {len(gutenberg.sents())}, webtext sents {len(webtext.sents())}')
        print("Total sentences in corpus: %d" % train_count)

        random.shuffle(train_sequences)
        train_sequences = train_sequences[:self.TRAIN_SIZE]
        print("Number of train_sequences(after shuffling): ", len(train_sequences))

        return train_sequences

    def get_vocab_words(self):
        word_freq = {}
        words = brown.words()
        words = [word.lower() for word in words]
        print(f"brown corpus words: {len(words)}, unique {len(set(words))}")
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1

        words = gutenberg.words()
        words = [word.lower() for word in words]
        print(f"gutenberg corpus words: {len(words)}, unique {len(set(words))}")
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1

        words = webtext.words()
        words = [word.lower() for word in words]
        print(f"webtext corpus words: {len(words)}, unique {len(set(words))}")
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1

        words = self.my_corpus_words()
        words = [word.lower() for word in words]
        print(f"custom corpus words: {len(words)}, unique {len(set(words))}")
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1

        return word_freq

    def get_embedding_init(self, embedding_dim):
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

    def my_corpus_words(self):
        sents = self.my_corpus_sents()
        words = []
        for sent in sents:
            if len(sent) <= 1:
                continue
            for word in sent:
                words.extend(word)
        return words

    def my_corpus_sents(self):
        with open("data/my_corpus.txt") as f:
            contents = f.readlines()
        sents = [x.strip().lower().split(" ") for x in contents]
        return sents



    def list_to_file(self, mylist, filename):
        with open(filename, 'w') as file:
            for element in mylist:
                file.write('%s\n' % element)

    def dict_to_file(self, mydict, filename):
        with open(filename, 'w') as file:
            for key, value in mydict.items():
                file.write(str(value)+':  '+str(key)+'\n')
