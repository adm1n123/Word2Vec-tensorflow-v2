import nltk
import numpy as np

from nltk.corpus import brown, gutenberg, webtext, reuters, stopwords

nltk.download("brown")
nltk.download("gutenberg")
nltk.download("webtext")
nltk.download("reuters")

nltk.download('punkt')
nltk.download('stopwords')

from gensim.test.utils import common_texts
from gensim.models import Word2Vec

print(brown.sents()[:10])
model = Word2Vec(sentences=brown.sents(), vector_size=200, window=5, min_count=5, workers=4, sg=1, negative=5)
# model.save("word2vec.model")
# model = Word2Vec.load("word2vec.model")
# model.train([["hello", "world"]], total_examples=1, epochs=1)


vector = model.wv['computer']
print(vector)  # get numpy vector of a word
sims = model.wv.most_similar('computer', topn=10)  # get other similar words
print(sims)

sims = model.wv.most_similar('good', topn=10)  # get other similar words
print(sims)

sims = model.wv.most_similar('country', topn=10)  # get other similar words
print(sims)

sims = model.wv.most_similar('man', topn=10)  # get other similar words
print(sims)