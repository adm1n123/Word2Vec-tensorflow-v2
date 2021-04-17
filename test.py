import functools
import io
import itertools
from pprint import pprint
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models.keyedvectors import KeyedVectors
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


import numpy as np






def main():

    embeddings = get_embedding_init()
    visualize_words(embeddings)
    exit(1)
    evaluate_context_learning(embeddings)
    cos_similarities(embeddings)
    run_test_for_class(embeddings)
    test_analogy(embeddings, 50)


    return None

def convert_bin_to_txt():
    model = KeyedVectors.load_word2vec_format('data/google-news-300/GoogleNews-vectors-negative300.bin', binary=True)
    model.save_word2vec_format('data/google-news-300/GoogleNews-vectors-negative300.txt', binary=False)
    return

def get_only_vocab_words(embeddings, words):
    p_words = []
    a_words = []
    for word in words:
        if embeddings.get(word) is not None:
            p_words.append(word)
        else:
            a_words.append(word)
    return p_words, a_words


def evaluate_context_learning(embeddings):
    present = ['eats', 'runs', 'plays', 'running', 'playing', 'running']
    past = ['ate', 'ran', 'played', 'walking', 'eating', 'playing']
    present, a_present = get_only_vocab_words(embeddings, words=present)
    past, a_past = get_only_vocab_words(embeddings, words=past)

    print('words not found: ', a_present, a_past)

    present_vec = np.array([embeddings[word] for word in present])
    past_vec = np.array([embeddings[word] for word in past])


    diff_word = []
    diff_vec = []
    for i in range(len(present)):
        diff_word.append(present[i]+'_vector - '+past[i]+'_vector')
        diff_vec.append(np.subtract(present_vec[i], past_vec[i]))

    for i in range(len(diff_word)):
        for j in range(i+1, len(diff_word)):
            print('cosine similarity: ('+diff_word[i]+').('+diff_word[j]+') is: ', cosine_similarity([diff_vec[i]], [diff_vec[j]]))


def cos_similarities(embeddings, words=None):
    data = 'cat dog human male female computer keyboard walking room'.lower()
    if words is not None:
        data = words

    words = data.split(" ")
    words, a_words = get_only_vocab_words(embeddings, words)
    print('words not found: ', a_words)

    word_vectors = np.array([embeddings[word] for word in words])

    for i in range(len(words)):
        for j in range(len(words)):
            if i >= j:
                continue
            print("word pair: [{}, {}], Cosine Similarity: {}".format(words[i], words[j], cosine_similarity([word_vectors[i]], [word_vectors[j]])))
            # print("word1: {}, word vector: {}\nword2: {}, word vector: {}".format(words[i], word_vectors[i], words[j], word_vectors[j]))


def run_test_for_class(embeddings):

    fruits = 'papaya banana grapes mango'
    activity = 'plays eats runs play eat run'
    other = 'today yesterday'
    hverbs = 'do does is has have did was were had will shall'
    nouns = 'raju amit robbin nancy david john alice'

    classes = [fruits, activity, other, hverbs, nouns]
    for class_ in classes:
        cos_similarities(embeddings, class_)

    return None

def topn(embeddings, target, n):
    words = {}
    for word, vector in embeddings.items():
        words[word] = np.around(cosine_similarity([target], [vector])[0][0], 3)

    sort = sorted(words.items(), key=lambda x: x[1], reverse=True)[:n]
    sort = [f'{word}:{val}' for word, val in sort]
    return sort

def test_analogy(embeddings, n, analogies=None):
    data = [
        ['king', 'man', 'queen', 'woman'],
        ['good', 'better', 'great', 'greater'],
        ['king', 'boy', 'queen', 'girl'],
        ['play', 'plays', 'eat', 'eats'],
        ['france', 'paris', 'india', 'delhi'],
        ['france', 'paris', 'england', 'london'],
        ['sun', 'day', 'moon', 'night'],
        ['politics', 'politician', 'engineering', 'engineer'],
        ['move', 'moving', 'run', 'running']
    ]

    for analogy in data:
        a_w = analogy[0]; b_w = analogy[1]; c_w = analogy[2]; d_w = analogy[3]

        print(f'\nTesting analogy: {analogy}')
        _, notfound = get_only_vocab_words(embeddings, analogy)
        if len(notfound) > 0:
            print(f'Words not found:{notfound} can not proceed with analogy testing')
            continue
        # a:b::?:d  ? = a-b+d
        a = embeddings.get(a_w)
        b = embeddings.get(b_w)
        d = embeddings.get(d_w)
        c = np.add(np.subtract(a, b), d)

        print(f'Top {n} words similar to [{a_w} - {b_w} + {d_w}] are: {topn(embeddings, c, n)}')


def visualize_words(embeddings):
    # data = [
    #     ['king', 'man', 'queen', 'woman'],
    #     ['good', 'better', 'great', 'greater'],
    #     ['king', 'boy', 'queen', 'girl'],
    #     ['play', 'plays', 'eat', 'eats'],
    #     ['france', 'paris', 'india', 'delhi'],
    #     ['france', 'paris', 'england', 'london'],
    #     ['sun', 'day', 'moon', 'night'],
    #     ['politics', 'politician', 'engineering', 'engineer'],
    #     ['move', 'moving', 'run', 'running']
    # ]
    words = ["boy", "girl", "man", "woman", "king", "queen", "banana", "apple", "mango", "fruit", "coconut", "orange"]
    # data = functools.reduce(lambda a,b: a+b, data)

    # data.extend(words)
    # words = list(set(data))

    words, not_found = get_only_vocab_words(embeddings, words)
    print(f'words not found could not visualize: {not_found}')

    labels = []
    wordvecs = []

    for word in words:
        wordvecs.append(embeddings.get(word))
        labels.append(word)

    tsne_model = TSNE(perplexity=3, n_components=2, init='pca', random_state=42)
    coordinates = tsne_model.fit_transform(wordvecs)

    x = []
    y = []
    for value in coordinates:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(8, 8))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(2, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()


def get_embedding_init():
    metadata_file = 'data/pretrained-200/metadata-dim-200.tsv'
    vectors_file = 'data/pretrained-200/vectors-dim-200.tsv'

    try:
        meta_fd = io.open(metadata_file, 'r')
        vectors_fd = io.open(vectors_file, 'r')

        vocab = [line.strip() for line in meta_fd.readlines()]
        vectors = [np.fromstring(line, dtype=float, sep='\t') for line in vectors_fd.readlines()]
        old_embeddings = dict(zip(vocab, vectors))

        meta_fd.close()
        vectors_fd.close()
        print(f'Using pretrained word embeddings')

        # sample = dict(itertools.islice(old_embeddings.items(), 5))
        # pprint(sample)

    except:
        print('could not read pretrained embedding files')
        old_embeddings = None

    return old_embeddings


if __name__ == '__main__':
    main()