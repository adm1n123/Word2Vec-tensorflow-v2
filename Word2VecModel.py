import io
import math
import random
import re
import string
import sys
import numpy as np

import tensorflow as tf
import tqdm

from tensorflow.keras import Model
from tensorflow.keras.layers import Dot, Embedding, Flatten
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


class Word2Vec(Model):

    def __init__(self, corpus, embedding_dim, num_ns, window_size, seed, T, embeddings_init=None):
        super(Word2Vec, self).__init__()
        self.corpus = corpus
        self.embedding_dim = embedding_dim  # word vec dimensions
        self.num_ns = num_ns    # number of negative samples per positive sample
        self.target_embedding = None
        self.context_embedding = None
        self.dots = None
        self.flatten = None
        self.window_size = window_size
        self.sampling_table = None
        self.SEED = seed
        self.T = T
        self.embeddings_init = embeddings_init

    def call(self, pair, training=None, mask=None):
        target, context = pair
        we = self.target_embedding(target)
        ce = self.context_embedding(context)
        dots = self.dots([ce, we])
        return self.flatten(dots)

    def create_model(self):
        embeddings_init = 'uniform'
        if self.embeddings_init is not None:
            embeddings_init = tf.keras.initializers.Constant(self.embeddings_init)

        self.target_embedding = Embedding(
            input_dim=self.corpus.vocab_size,
            output_dim=self.embedding_dim,
            embeddings_initializer=embeddings_init,
            input_length=1,
            trainable=True,
            name="target_vectors"
        )

        self.context_embedding = Embedding(
            input_dim=self.corpus.vocab_size,
            output_dim=self.embedding_dim,
            input_length=self.num_ns + 1,
            trainable=True,
            name="context_vectors"
        )

        self.dots = Dot(axes=(3, 2))

        self.flatten = Flatten()

        self.compile(
            optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.001),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        return None

    # Generates skip-gram pairs with negative sampling for a list of sequences
    # (int-encoded sentences) based on window size, number of negative samples
    # and vocabulary size.
    def get_training_data(self, train_sequences):
        # Elements of each training example are appended to these lists.
        targets, contexts, labels = [], [], []

        # Build the sampling table for vocab_size tokens.
        self.sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(self.corpus.vocab_size)

        # Iterate over all sequences (sentences) in dataset.
        total = len(train_sequences)
        count = 0
        for sequence in train_sequences:
            # Generate positive skip-gram pairs for a sequence (sentence).
            positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
                sequence=sequence,
                vocabulary_size=self.corpus.vocab_size,
                sampling_table=self.sampling_table,
                window_size=self.window_size,
                negative_samples=0
            )

            # Iterate over each positive skip-gram pair to produce training examples
            # with positive context word and negative samples.
            for target_word, context_word in positive_skip_grams:
                context_class = tf.expand_dims(tf.constant([context_word], dtype="int64"), 1)
                negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
                    true_classes=context_class,
                    num_true=1,
                    num_sampled=self.num_ns,
                    unique=True,
                    range_max=self.corpus.vocab_size,
                    seed=self.SEED,
                    name="negative_sampling"
                )

                # Build context and label vectors (for one target word)
                negative_sampling_candidates = tf.expand_dims(negative_sampling_candidates, 1)

                context = tf.concat([context_class, negative_sampling_candidates], 0)
                label = tf.constant([1] + [0] * self.num_ns, dtype="int32")

                # Append each element from the training example to global lists.
                targets.append(target_word)
                contexts.append(context)
                labels.append(label)

            count += 1
            if count % 100 == 0:
                sys.stdout.write("\r %d/%d getting training data progress: %d%%" % (count, total, int(count*100/total)))
                sys.stdout.flush()
        print()
        return targets, contexts, labels






    def gen_train_input(self, sequences):
        targets, contexts, labels = [], [], []

        self.init_sampling_table()
        self.init_unigram_table()

        total = len(sequences)
        count = 0
        for seq in sequences:
            skipgrams = self.positive_skipgrams(seq)

            for skip_pair in skipgrams:
                target, context = skip_pair

                negs = self.negative_samples(skip_pair)

                context = tf.concat([[context], negs], axis=0)
                context = tf.expand_dims(context, axis=1)
                label = tf.constant([1] + [0]*len(negs), dtype='int32')

                targets.append(target)
                contexts.append(context)
                labels.append(label)
            count += 1
            if count % 100 == 0:
                sys.stdout.write(
                    "\r %d/%d getting training data progress: %d%%" % (count, total, int(count * 100 / total)))
                sys.stdout.flush()
        print()
        return targets, contexts, labels

    def init_sampling_table(self):
        # real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn; formula used by official impl.
        self.samp_table = np.zeros(self.corpus.vocab_size, dtype=float)  # subsampling doesn't affect much of the accuracy just a little improved.

        for word in self.corpus.words:
            fw = word.freq/self.corpus.freq_sum
            self.samp_table[word.idx] = math.sqrt(self.T/fw)    # samp_table[i] is probability of being taken as target word.  formula as per paper.
        return None

    def positive_skipgrams(self, sequence):

        skipgrams = []
        for i, wi in enumerate(sequence):
            if not wi: # ignore UNK
                continue
            if self.samp_table[wi] < random.random():   # word is not taken as 'target'(center) word.
                continue

            window_start = max(0, i - self.window_size)
            window_end = min(len(sequence), i + self.window_size + 1)
            for j in range(window_start, window_end):   # wi is 'target' word now take all its context words(within window) as positive.
                if j != i:
                    wj = sequence[j]
                    if not wj: # ignore UNK
                        continue
                    skipgrams.append([wi, wj])

        return skipgrams

    def init_unigram_table(self):
        self.table_size = int(1e8)  # take table_size proportional to vocab size (larger size is more accurate)
        self.unigram_table = np.zeros(self.table_size, dtype=int)
        z = 0   # denominator
        power = .75
        for word in self.corpus.words:
            if word.idx > 0:    # ignore UNK
                z += math.pow(word.freq, power)

        idx = 1     # not taking UNK(idx=0) in negative sample
        frac_sum = math.pow(self.corpus.words[idx].freq, power) / z
        for i in range(self.table_size):
            self.unigram_table[i] = idx
            if i > frac_sum * self.table_size:
                idx += 1    # now next word idx will be filled in unigram_table
                frac_sum += math.pow(self.corpus.words[idx].freq, power) / z # for idx = vocab_size it will overflow try putting next if statement before it.
            if idx >= self.corpus.vocab_size:
                idx = self.corpus.vocab_size - 1
        return None
        
    def negative_samples(self, pos_pair):
        # TODO: incorporate variable length input and check same sample is not sampled again.
        wi, wj = pos_pair
        neg_samples = []
        # for i in range(self.num_ns):
        i = self.num_ns
        while i > 0:
            neg = math.floor(random.random() * self.table_size)
            neg_idx = self.unigram_table[neg]
            if neg_idx == wi or neg_idx == wj:
                continue
            neg_samples.append(neg_idx)
            i -= 1

        return neg_samples
