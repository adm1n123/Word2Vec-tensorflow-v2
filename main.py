import io

from datetime import datetime

import tensorflow as tf

from Word2VecModel import Word2Vec
from CorpusProcessor import CorpusProcessor

SEED = 42
AUTOTUNE = tf.data.AUTOTUNE
TRAIN_SIZE = 100000  # max number of sequences(arbitrary length) from corpus

EMBEDDING_DIM = 200   # word vec dimensions
WINDOW_SIZE = 2     # window size for skip gram
NEG_SAMPLES_COUNT = 4     # negative samples per positive sample
MIN_COUNT = 5      # filter rare words with freq < 5

BATCH_SIZE = 128
BUFFER_SIZE = 1000
EPOCHS = 10



def main():

    print(f'BATCH_SIZE:{BATCH_SIZE}, EPOCHS:{EPOCHS}')
    print(f'TRAIN_SIZE:{TRAIN_SIZE}, EMBEDDING_DIM:{EMBEDDING_DIM}, WINDOW_SIZE:{WINDOW_SIZE}, '
          f'NEG_SAMPLES_COUNT:{NEG_SAMPLES_COUNT}, MIN_COUNT:{MIN_COUNT}')


    corpus = CorpusProcessor(train_size=TRAIN_SIZE, min_count=MIN_COUNT)
    train_sequences = corpus.get_train_data()

    corpus.extend_vocab()   # add new words from custom corpus.

    embeddings_init = corpus.get_embedding_init(embedding_dim=EMBEDDING_DIM)   # get pretrained word embeddings.

    word2vec = Word2Vec(
        vocab_size=corpus.vocab_size,
        embedding_dim=EMBEDDING_DIM,
        window_size=WINDOW_SIZE,
        num_ns=NEG_SAMPLES_COUNT,
        seed=SEED,
        embeddings_init=embeddings_init
    )

    word2vec.create_model()
    print("Model is created", datetime.now())

    train_from_brown_corpus(word2vec=word2vec, train_sequences=train_sequences) # Train for brown corpus ####################

    train_from_custom_corpus(word2vec=word2vec, corpus=corpus)  # Train for custom corpus ####################

    print(word2vec.summary())

    print("Dumping word embeddings in file")
    weights = word2vec.get_layer('target_vectors').get_weights()[0]

    unique = str(datetime.now()).replace(':', '-').replace(' ', '__')
    out_v = io.open(f'data/vectors-dim-{EMBEDDING_DIM}-{unique}.tsv', 'w', encoding='utf-8')
    out_m = io.open(f'data/metadata-dim-{EMBEDDING_DIM}-{unique}.tsv', 'w', encoding='utf-8')

    for word, idx in corpus.vocab.items():
        if idx == 0:
            continue  # skip 0, it's padding.
        vec = weights[idx]
        out_v.write('\t'.join([str(x) for x in vec]) + "\n")
        out_m.write(word + "\n")
    out_v.close()
    out_m.close()


def train_from_brown_corpus(word2vec, train_sequences):
    print("############### Training on brown corpus ###################")
    targets, contexts, labels = word2vec.get_training_data(train_sequences)
    print(f"size of targets {len(targets)}, contexts {len(contexts)}, labels {len(labels)}, Time {datetime.now()}")
    dataset = tf.data.Dataset.from_tensor_slices(
        ((targets, contexts), labels))  # (targets, contexts) is input and 'labels' is expected output.
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    print(dataset)
    dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)
    print(dataset)
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")
    word2vec.fit(
        dataset,
        epochs=EPOCHS,
        shuffle=True,
        callbacks=None,
        workers=1
    )


def train_from_custom_corpus(word2vec, corpus):
    print("############### Training on custom corpus ###################")
    custom_corp_train = corpus.get_train_data_from_sentences()
    targets, contexts, labels = word2vec.get_training_data(custom_corp_train)
    print(f"size of targets {len(targets)}, contexts {len(contexts)}, labels {len(labels)}, Time {datetime.now()}")
    dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))  # (targets, contexts) is input and 'labels' is expected output.
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    print(dataset)
    dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)
    print(dataset)
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")
    word2vec.fit(
        dataset,
        epochs=EPOCHS,
        shuffle=True,
        callbacks=None,
        workers=1
    )


if __name__ == '__main__':
    main()
