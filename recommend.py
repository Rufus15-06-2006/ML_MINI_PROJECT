"""Load the trained model and recommend extra MeSH keywords for a query text."""
import json, numpy as np
from extract import MeshTrie

class Recommender:
    def __init__(self, model="model.npz", syn="synonyms.json"):
        z = np.load(model, allow_pickle=False)
        self.W, self.Theta, self.b = z["W"], z["Theta"], z["b"]
        self.vocab = [str(v) for v in z["vocab"]]; self.idx = {v.lower(): i for i, v in enumerate(self.vocab)}
        self.idf = np.log10(int(z["n_docs"]) / np.maximum(z["df"], 1))
        self.trie = MeshTrie(json.load(open(syn)))
        self.A = self.W * self.Theta
    def vectorize(self, text):
        x = np.zeros(len(self.vocab), np.float32); found = {}
        for concept, cnt in self.trie.extract(text).items():
            i = self.idx.get(concept.lower())
            if i is not None: x[i] = np.log10(1 + cnt) * self.idf[i]; found[self.vocab[i]] = cnt
        return x, found
    def recommend(self, text, k=5):
        x, found = self.vectorize(text)
        if not found: return found, []
        p = np.maximum(x @ self.A.T + self.b, 0); p[x > 0] = -1   # only suggest NEW terms
        top = [i for i in np.argsort(-p)[:k] if p[i] > 0]
        return found, [(self.vocab[i], float(p[i])) for i in top]
