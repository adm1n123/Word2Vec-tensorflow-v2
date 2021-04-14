import io
import re
import string
import sys

import tensorflow as tf
import tqdm

from tensorflow.keras import Model
from tensorflow.keras.layers import Dot, Embedding, Flatten
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


class Word2Vec(Model):

    def __init__(self, vocab_size, embedding_dim, num_ns, window_size, seed):
        super(Word2Vec, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim  # word vec dimensions
        self.num_ns = num_ns    # number of negative samples per positive sample
        self.target_embedding = None
        self.context_embedding = None
        self.dots = None
        self.flatten = None
        self.window_size = window_size
        self.sampling_table = None
        self.SEED = seed

    def call(self, pair, training=None, mask=None):
        target, context = pair
        we = self.target_embedding(target)
        ce = self.context_embedding(context)
        dots = self.dots([ce, we])
        return self.flatten(dots)

    def create_model(self):
        self.target_embedding = Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            input_length=1,
            name="target_vectors"
        )

        self.context_embedding = Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            input_length=self.num_ns + 1,
            name="context_vectors"
        )

        self.dots = Dot(axes=(3, 2))

        self.flatten = Flatten()

        self.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        return None

    # Generates skip-gram pairs with negative sampling for a list of sequences
    # (int-encoded sentences) based on window size, number of negative samples
    # and vocabulary size.
    def get_training_data(self, corpus):
        # Elements of each training example are appended to these lists.
        targets, contexts, labels = [], [], []

        # Build the sampling table for vocab_size tokens.
        self.sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(self.vocab_size)  #

        # Iterate over all sequences (sentences) in dataset.
        total = len(corpus.train_sequences)
        count = 0
        for sequence in corpus.train_sequences:
            # Generate positive skip-gram pairs for a sequence (sentence).
            positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
                sequence=sequence,
                vocabulary_size=self.vocab_size,
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
                    range_max=self.vocab_size,
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

        return targets, contexts, labels

