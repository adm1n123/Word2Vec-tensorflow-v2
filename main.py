import io
import re
import string
from datetime import datetime

import tensorflow as tf
import tqdm

from tensorflow.keras import Model
from tensorflow.keras.layers import Dot, Embedding, Flatten
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

from Word2VecModel import Word2Vec
from CorpusProcessor import CorpusProcessor
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE
TRAIN_SIZE = 10000


corpus = CorpusProcessor(train_size=TRAIN_SIZE)
corpus.get_train_data()

word2vec = Word2Vec(
    vocab_size=corpus.vocab_len,
    embedding_dim=50,
    window_size=2,
    num_ns=4,
    seed=SEED
)

word2vec.create_model()
print("Model is created", datetime.now())

targets, contexts, labels = word2vec.get_training_data(corpus)

print(f"size of targets {len(targets)}, contexts{len(contexts)}, labels{len(labels)}, Time {datetime.now()}")


BATCH_SIZE = 16
BUFFER_SIZE = 100
dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
print(dataset)

dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)
print(dataset)

# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")

word2vec.fit(dataset, epochs=1, callbacks=None)
print(word2vec.summary())

weights = word2vec.get_layer('target_vectors').get_weights()[0]
vocab = corpus.get_vocab()

out_v = io.open('data/vectors.tsv', 'w', encoding='utf-8')
out_m = io.open('data/metadata.tsv', 'w', encoding='utf-8')

for index, word in enumerate(vocab):
    if index == 0:
        continue  # skip 0, it's padding.
    vec = weights[index]
    out_v.write('\t'.join([str(x) for x in vec]) + "\n")
    out_m.write(word + "\n")
out_v.close()
out_m.close()
