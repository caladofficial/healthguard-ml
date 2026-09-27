"""03 — CLUSTERING + DE-CLUSTERING (phenotype grouping & mixture-of-experts).
EVOLVEX IT SOLUTIONS PVT. LTD.

User's requested arsenal mapped:
  · clustering      → KMeans + GaussianMixture + DBSCAN phenotyping on scaled X
  · 'm mean' (k-means) → KMeans explicitly (with silhouette-driven k choice)
  · 'de clustering' → de-clustering = decomposing the mix into per-cluster expert
                      sub-models (mixture of experts) + cluster-purity analysis
  · 'skill grouping'→ sklearn grouping primitives (ColumnTransformer / pipelines)

Outputs: cluster labels as extra features (density of risk mix), cluster stats,
purity report, and a gating helper for MoE training.
"""
import sys, os, json
import numpy as np, pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

def _scaled(X):
    return StandardScaler().fit_transform(X)

def fit_scaler(X):
    return StandardScaler().fit(X)

def scale_with(scaler, X):
    return scaler.transform(X)

def choose_k(Z, kmin=2, kmax=8):
    """Silhouette-driven k for KMeans (k-means/'m mean'). Big-n: capped k range
    and SAMPLED silhouette (full pairwise is O(n²) memory — the OOM we hit)."""
    big = len(Z) > 20000
    if big:
        kmax = min(kmax, 4)
    best_k, best_s, scores = 2, -1, {}
    for k in range(kmin, min(kmax, len(Z) - 1) + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=C.SEED).fit(Z)
        if big:
            s = silhouette_score(Z, km.labels_, sample_size=8000, random_state=C.SEED)
        else:
            s = silhouette_score(Z, km.labels_)
        scores[k] = round(float(s), 4)
        if s > best_s:
            best_k, best_s = k, s
    return best_k, scores

def phenotype_report(X, y):
    """KMeans + GMM + DBSCAN phenotypes with purity vs outcome.
    Returns (km, gmm, scaler, report) — scaler is FIT ONCE and must be reused
    at inference (never refit per call/batch). DBSCAN skipped for n>30000
    (O(n²) memory) — flagged in the report."""
    scaler = fit_scaler(X)
    Z = scale_with(scaler, X)
    k, scores = choose_k(Z)
    km = KMeans(n_clusters=k, n_init=10, random_state=C.SEED).fit(Z)
    gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=C.SEED).fit(Z)
    gl = gmm.predict(Z)
    if len(Z) > 30000:
        db = None
    else:
        db = DBSCAN(eps=0.9, min_samples=5).fit_predict(Z)

    def purity(labels):
        ys = np.asarray(y)
        tab = {}
        for c in np.unique(labels):
            m = labels == c
            if m.sum() == 0: continue
            vals, cnt = np.unique(ys[m], return_counts=True)
            tab[int(c)] = {"n": int(m.sum()), "majority": int(vals[cnt.argmax()]),
                           "purity": round(float(cnt.max() / m.sum()), 3),
                           "class_mix": {int(v): int(n) for v, n in zip(vals, cnt)}}
        return tab

    report = {
        "kmeans_k": k, "silhouette_by_k": scores,
        "kmeans_purity": purity(km.labels_),
        "gmm_purity": purity(gl),
        "dbscan_clusters": (int(len(set(db) - {-1})) if db is not None else "skipped_big_n"),
        "dbscan_noise_frac": (round(float((db == -1).mean()), 3) if db is not None else "skipped_big_n"),
        "gmm_vs_kmeans_agreement": round(float((km.labels_ == gl).mean()), 3),
    }
    return km, gmm, scaler, report

def cluster_features(X, km, gmm, scaler):
    """De-clustering step 1: encode phenotype membership as model features
    (soft GMM posteriors + hard KMeans id)."""
    Z = scale_with(scaler, X)
    post = gmm.predict_proba(Z)
    hard = km.predict(Z)
    cols = {}
    for i in range(post.shape[1]):
        cols[f'gmm_p{i}'] = post[:, i]
    cols['kmeans_id'] = hard
    return pd.DataFrame(cols)  # positional index — aligns with reset_index frames

def cluster_indices(X, km, scaler):
    """De-clustering step 2: per-cluster sample indices for MoE experts."""
    Z = scale_with(scaler, X)
    labels = km.predict(Z)
    return {int(c): np.where(labels == c)[0] for c in np.unique(labels)}

if __name__ == '__main__':
    from datasets import load_all
    out = {}
    for name, (X, y, task) in load_all().items():
        km, gmm, scaler, rep = phenotype_report(X, y)
        out[name] = rep
        print(f"\n== {name} == k={rep['kmeans_k']} sil(k)={rep['silhouette_by_k']} "
              f"DBSCAN(noise={rep['dbscan_noise_frac']}) agreement={rep['gmm_vs_kmeans_agreement']}")
        for c, st in rep['kmeans_purity'].items():
            print(f"   cluster {c}: n={st['n']} purity={st['purity']} mix={st['class_mix']}")
    with open(os.path.join(C.REPORTS, 'cluster_report.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n→ reports/cluster_report.json")
