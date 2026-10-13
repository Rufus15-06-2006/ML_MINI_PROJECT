"""Collaborative-filtering keyword recommender (MeSH terms, ClinicalTrials.gov).
Usage:
  python pipeline.py --synthetic --epochs 10                       # offline smoke test
  python pipeline.py --mesh desc2026.xml --epochs 60               # paper-style Trie extraction
  python pipeline.py --mesh desc2026.xml --epochs 60 --tune 30     # + hyperparameter search on dev set
"""
import argparse, json, os, csv, itertools, numpy as np
from collections import Counter

# ---------------------------------------------------------------- data
def fetch(n_pages=5, cache="trials.json"):
    if os.path.exists(cache):
        return json.load(open(cache))
    import requests, time
    docs, token = [], None
    for _ in range(n_pages):
        p = {"pageSize": 1000, "fields": "NCTId,BriefTitle,OfficialTitle,BriefSummary,DetailedDescription,Condition,EligibilityCriteria,ConditionBrowseModule,InterventionBrowseModule"}
        if token: p["pageToken"] = token
        r = requests.get("https://clinicaltrials.gov/api/v2/studies", params=p, timeout=60).json()
        for s in r["studies"]:
            ps, ds = s.get("protocolSection", {}), s.get("derivedSection", {})
            idm, dm = ps.get("identificationModule", {}), ps.get("descriptionModule", {})
            raw = " . ".join([idm.get("briefTitle", ""), idm.get("officialTitle", ""), dm.get("briefSummary", ""),
                              dm.get("detailedDescription", ""), " ; ".join(ps.get("conditionsModule", {}).get("conditions", [])),
                              ps.get("eligibilityModule", {}).get("eligibilityCriteria", "")])
            terms = []
            for k in ("conditionBrowseModule", "interventionBrowseModule"):
                m = ds.get(k, {})
                terms += [x["term"].lower() for x in m.get("meshes", [])]
                terms += [x["term"].lower() for x in m.get("ancestors", [])]
            docs.append({"text": raw.lower(), "terms": terms})
        print(f"fetched {len(docs)} trials"); token = r.get("nextPageToken")
        if not token: break
        time.sleep(0.3)
    json.dump(docs, open(cache, "w"))
    return docs

def synthetic(n=6000, m=300, seed=0):
    rng = np.random.default_rng(seed); topics = rng.integers(0, 10, m); docs = []
    for _ in range(n):
        t = rng.integers(0, 10); pool = np.where(topics == t)[0]
        ids = rng.choice(pool, size=min(len(pool), rng.integers(5, 20)), replace=False)
        docs.append({"text": "", "terms": [f"term{i}" for i in ids]})
    return docs

def build_tfidf(docs, top_m=500, max_df=0.5):
    df = Counter(t for d in docs for t in set(d["terms"]))
    vocab = [t for t, c in df.most_common() if c <= max_df * len(docs)][:top_m]; idx = {t: i for i, t in enumerate(vocab)}
    X = np.zeros((len(docs), len(vocab)), dtype=np.float32)
    for j, d in enumerate(docs):
        c = Counter(t for t in d["terms"] if t in idx)
        for t, k in c.items():
            k += d["text"].count(t)  # + mentions in free text
            X[j, idx[t]] = np.log10(1 + k) * np.log10(len(docs) / df[t])
    return X, vocab

# ---------------------------------------------------------------- model
def neighbor_weights(Xtr, tau, k):
    Z = Xtr / (np.linalg.norm(Xtr, axis=0, keepdims=True) + 1e-9)
    cos = np.clip(Z.T @ Z, 0, 1)
    W = np.exp(tau * (1 - cos) ** k); np.fill_diagonal(W, 0)
    return W.astype(np.float32)

class CF:
    """R_hat = ReLU(X (W*Theta)^T + b); hand-derived gradients from the paper."""
    def __init__(s, W, relu=True):
        s.W, s.relu = W, relu; m = W.shape[0]
        s.Theta = np.zeros((m, m), np.float32); s.b = np.zeros(m, np.float32)
    def pre(s, X): return X @ (s.W * s.Theta).T + s.b
    def predict(s, X):
        p = s.pre(X); return np.maximum(p, 0) if s.relu else p

def train(Xtr, W, lam=0.01, lr=0.1, epochs=30, bs=512, relu=True, verbose=True):
    model = CF(W, relu); n = len(Xtr)
    for ep in range(epochs):
        order = np.random.permutation(n); tot = 0
        for i in range(0, n, bs):
            B = Xtr[order[i:i+bs]]; p = model.pre(B)
            out = np.maximum(p, 0) if relu else p
            E = out - B
            if relu: E = E * (p >= 0)          # ReLU'(0)=1 as in the paper
            gT = model.W * (E.T @ B) / len(B) + lam * model.Theta
            gb = E.sum(0) / len(B)
            model.Theta -= lr * gT; model.b -= lr * gb
            tot += 0.5 * ((out - B) ** 2).sum()
        if verbose and (ep % 5 == 0 or ep == epochs - 1): print(f"epoch {ep} loss/doc {tot/n:.4f}")
    return model

def knn_model(Xtr, k=50):
    """Item-item cosine kNN: score(i) = sum over i's k nearest terms f of cos(i,f) * x_f."""
    Z = Xtr / (np.linalg.norm(Xtr, axis=0, keepdims=True) + 1e-9)
    S = Z.T @ Z; np.fill_diagonal(S, 0)
    drop = np.argsort(-S, axis=1)[:, k:]            # zero out all but top-k neighbours per term
    np.put_along_axis(S, drop, 0, axis=1)
    return lambda M: M @ S.T

# ---------------------------------------------------------------- evaluation
def evaluate(Xte, predict, pop, K=3, seed=0):
    rng = np.random.default_rng(seed); hits = {"model": [], "random": [], "popularity": []}
    Xm = Xte.copy()
    for j in range(len(Xte)):
        nz = np.flatnonzero(Xte[j]); hide = rng.choice(nz, size=len(nz) // 2, replace=False) if len(nz) >= 2 else []
        Xm[j, hide] = 0
    P = predict(Xm)
    err = (P - Xte) ** 2; mse = err.mean()
    mser = err.sum() / np.maximum(((Xte > 0) | (P > 0)).sum(), 1)
    for j in range(len(Xte)):
        cand = np.flatnonzero(Xm[j] == 0); truth = Xte[j] > 0
        if len(cand) < K or truth[cand].sum() == 0: continue
        top = cand[np.argsort(-P[j, cand])[:K]]; hits["model"].append(truth[top].mean())
        hits["random"].append(truth[rng.choice(cand, K, replace=False)].mean())
        hits["popularity"].append(truth[cand[np.argsort(-pop[cand])[:K]]].mean())
    return mse, mser, {k: float(np.mean(v)) for k, v in hits.items()}

# ---------------------------------------------------------------- tuning
def tune(tr, dv, pop, n_trials, n_docs, epochs, seed=0):
    """Random search on the DEV set (train subsampled for speed). Always includes the paper's defaults."""
    rng = np.random.default_rng(seed)
    grid = list(itertools.product([-10, -1, -0.1, -0.001], [1, 2, 4, 16, 1000], [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]))
    default = (-1, 4, 1e-2)
    picks = [default] + [grid[i] for i in rng.permutation(len(grid)) if grid[i] != default][:max(n_trials - 1, 0)]
    sub = tr[:n_docs]; rows = []
    for tau, k, lam in picks:
        W = neighbor_weights(sub, tau, k)
        m = train(sub, W, lam=lam, epochs=epochs, verbose=False)
        mse, mser, h = evaluate(dv, m.predict, pop)
        rows.append((tau, k, lam, h["model"], mser))
        print(f"  tau={tau:<7} k={k:<5} lam={lam:<7} dev APTK={h['model']:.4f}  MSER={mser:.5f}")
    rows.sort(key=lambda r: -r[3])
    with open("tuning_results.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["tau", "k", "lambda", "dev_APTK", "dev_MSER"]); w.writerows(rows)
    dflt = [r for r in rows if tuple(r[:3]) == default][0][3]
    print("\nBest on dev: tau=%s k=%s lam=%s (APTK %.4f) vs paper defaults (APTK %.4f)" % (*rows[0][:3], rows[0][3], dflt))
    return rows[0][:3]

# ---------------------------------------------------------------- main
if __name__ == "__main__":
    import pickle
    ap = argparse.ArgumentParser(); ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--mesh", help="path to MeSH descYYYY.xml: extract terms from text with a Trie")
    ap.add_argument("--top_m", type=int, default=1000, help="number of MeSH terms (model cost grows with top_m squared)")
    ap.add_argument("--max_df", type=float, default=0.5, help="drop terms present in more than this fraction of trials (e.g. 0.2)")
    ap.add_argument("--pages", type=int, default=50); ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--no_ablation", action="store_true", help="skip the no-ReLU model (halves training time)")
    ap.add_argument("--tune", type=int, default=0, help="number of configs to try on the dev set (0 = off)")
    ap.add_argument("--tune_docs", type=int, default=15000); ap.add_argument("--tune_epochs", type=int, default=15)
    ap.add_argument("--tau", type=float, default=-1.0); ap.add_argument("--k", type=int, default=4); ap.add_argument("--lam", type=float, default=0.01)
    a = ap.parse_args(); np.random.seed(0)

    syn, cache = None, "terms_mesh.pkl"          # cache of Trie output: scan text only once
    if a.mesh:
        from extract import load_mesh_synonyms, MeshTrie
        syn = load_mesh_synonyms(a.mesh)
        if os.path.exists(cache):
            docs = [{"text": "", "terms": t} for t in pickle.load(open(cache, "rb"))]; print("loaded cached Trie output:", len(docs), "docs")
        else:
            docs = fetch(a.pages); trie = MeshTrie(syn); print("trie built; scanning text...")
            for d in docs:
                c = trie.extract(d["text"]); d["terms"] = [t for t, n_ in c.items() for _ in range(n_)]; d["text"] = ""
            pickle.dump([d["terms"] for d in docs], open(cache, "wb"))
    else:
        docs = synthetic() if a.synthetic else fetch(a.pages)
    X, vocab = build_tfidf(docs, a.top_m, a.max_df); print("matrix", X.shape, "density", (X > 0).mean())

    perm = np.random.permutation(len(X)); n = len(X)
    tr, dv, te = X[perm[:int(.8*n)]], X[perm[int(.8*n):int(.9*n)]], X[perm[int(.9*n):]]
    pop = (tr > 0).sum(0)

    tau, k, lam = a.tau, a.k, a.lam
    if a.tune:
        print(f"\nTuning on dev set ({a.tune} configs, {a.tune_docs} train docs, {a.tune_epochs} epochs each)")
        tau, k, lam = tune(tr, dv, pop, a.tune, a.tune_docs, a.tune_epochs)
    print(f"\nFinal training with tau={tau} k={k} lambda={lam} lr={a.lr}")

    results, models = {}, {}
    variants = [("CF+ReLU", True)] + ([] if a.no_ablation else [("CF (no ReLU)", False)])
    for name, relu in variants:
        W = neighbor_weights(tr, tau, k)
        model = train(tr, W, lam=lam, lr=a.lr, epochs=a.epochs, relu=relu)
        results[name] = evaluate(te, model.predict, pop); models[name] = model
        if relu: print("  dev APTK of final model: %.4f" % evaluate(dv, model.predict, pop)[2]["model"])

    # ---- save trained ReLU model for the demo app ----
    m = models["CF+ReLU"]; df = (X > 0).sum(0)
    np.savez("model.npz", W=m.W, Theta=m.Theta, b=m.b, df=df, n_docs=len(X), vocab=np.array(vocab))
    vset = set(vocab)
    synm = {t: c for t, c in syn.items() if c.lower() in vset or c in vset} if syn else {v: v for v in vocab}
    json.dump(synm, open("synonyms.json", "w")); print("saved model.npz and synonyms.json")

    results["Item-kNN (k=50)"] = evaluate(te, knn_model(tr), pop)
    print("\n%-16s %-9s %-9s %-7s %-7s %-7s" % ("model", "MSE", "MSER", "APTK", "APRK", "PopP@3"))
    for name, (mse, mser, h) in results.items():
        print("%-16s %-9.5f %-9.5f %-7.4f %-7.4f %-7.4f" % (name, mse, mser, h["model"], h["random"], h["popularity"]))
