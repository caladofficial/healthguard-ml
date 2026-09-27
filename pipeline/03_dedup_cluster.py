"""Stage 3 — DEDUPLICATION & CLUSTERING.

  1. exact duplicates  : dropped by row_hash.
  2. near duplicates   : k-means (fast, fixed k) AND HDBSCAN (density-based, finds
                         noise and arbitrarily-shaped blobs) over the embeddings.
  3. cluster curation  : keep at most `cap` representatives per cluster, chosen as
                         the members closest to the cluster centroid (medoid-ish),
                         so the surviving set is the *typical* record rather than
                         an outlier.
  4. mega-cluster flag : a cluster holding more than `mega_frac` of the corpus is
                         almost always boilerplate / degenerate sampling, not
                         genuine diversity -> flagged, and capped hardest.

Runs the curation on one embedding method at a time (--method) so the three
similarity notions can be compared on the same corpus.
"""
import os, sys, time, json, argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import pipeline_cfg as C          # noqa: E402

CAP = 40              # max representatives kept per cluster
MEGA_FRAC = 0.02      # >2% of the corpus in one cluster = suspicious
K_CLUSTERS = 4000      # k-means k  (~0.13% of the corpus)
HDBSCAN_MIN = 60      # min_cluster_size for the density pass


CORPUS = ['raw.parquet']


def load(method):
    df = pd.read_parquet(os.path.join(C.PIPE, CORPUS[0]))
    E = np.load(os.path.join(C.PIPE, f'emb_{method}.npy'))
    return df, E


def exact_dups(df):
    """Exact duplicates by content hash.

    NaN hashes must NOT count as duplicates: the harmonised real rows had no
    hash of their own and pandas' duplicated() would have collapsed 1,317 of
    them into 1, masquerading as '1,316 exact duplicates found'.
    """
    keep = [c for c in C.FEATURES if c in df.columns]
    h = df['row_hash']
    if h.isna().any():
        h = h.copy()
        miss = h.isna()
        h.loc[miss] = C.row_hash(df.loc[miss], keep)
    return h.duplicated(keep='first').values, h


def kmeans_labels(E, k, seed=C.SEED):
    from sklearn.cluster import MiniBatchKMeans
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=4096,
                         n_init=3, max_iter=100)
    return km.fit_predict(E)


def hdbscan_labels(E):
    import hdbscan
    cl = hdbscan.HDBSCAN(min_cluster_size=HDBSCAN_MIN, min_samples=5,
                         core_dist_n_jobs=2, prediction_data=False)
    return cl.fit_predict(E.astype('float64'))


def curate(E, labels, cap=CAP, mega_frac=MEGA_FRAC):
    """Keep <= cap members per cluster, nearest to the centroid first."""
    keep = np.zeros(len(labels), dtype=bool)
    n = len(labels)
    info = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        mega = len(idx) > mega_frac * n
        if c == -1:                      # HDBSCAN noise: keep, it is the diversity
            keep[idx] = True
            info.append({'cluster': -1, 'size': len(idx), 'kept': len(idx),
                         'mega': False, 'note': 'noise / unclustered'})
            continue
        sub = E[idx]
        cen = sub.mean(axis=0)
        d = ((sub - cen) ** 2).sum(axis=1)
        order = idx[np.argsort(d)]
        lim = max(1, cap // 4) if mega else cap
        keep[order[:lim]] = True
        info.append({'cluster': int(c), 'size': int(len(idx)), 'kept': int(lim),
                     'mega': bool(mega)})
    return keep, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', default='raw.parquet')
    ap.add_argument('--clusters', type=int, default=K_CLUSTERS)
    ap.add_argument('--method', default='numeric')
    ap.add_argument('--algo', default='kmeans', choices=['kmeans', 'hdbscan'])
    ap.add_argument('--cap', type=int, default=CAP)
    a = ap.parse_args()

    t0 = time.time()
    CORPUS[0] = a.corpus
    df, E = load(a.method)
    n_raw = len(df)

    dup, hashes = exact_dups(df)
    df['row_hash'] = hashes
    df = df.loc[~dup].reset_index(drop=True)
    E = E[~dup]
    n_after_exact = len(df)
    print(f'exact duplicates dropped: {int(dup.sum()):,} -> {n_after_exact:,} rows', flush=True)

    labels = (kmeans_labels(E, a.clusters) if a.algo == 'kmeans' else hdbscan_labels(E))
    keep, info = curate(E, labels, cap=a.cap)
    info.sort(key=lambda r: -r['size'])

    df['cluster'] = labels                      # every deduped row gets its cluster
    df.to_parquet(os.path.join(C.PIPE, f'clustered_{a.method}_{a.algo}.parquet'), index=False)
    df = df.loc[keep].reset_index(drop=True)
    out = os.path.join(C.PIPE, f'curated_{a.method}_{a.algo}.parquet')
    df.to_parquet(out, index=False)

    sizes = np.array([r['size'] for r in info])
    mega = [r for r in info if r['mega']]
    rec = {
        'stage': '03_dedup_cluster', 'method': a.method, 'algo': a.algo,
        'rows_raw': n_raw, 'exact_duplicates': int(dup.sum()),
        'rows_after_exact': n_after_exact,
        'clusters': int(len(info)), 'cap_per_cluster': a.cap,
        'rows_after_cluster_cap': int(len(df)),
        'removed_by_curation': int(n_after_exact - len(df)),
        'pct_removed_total': round(100 * (n_raw - len(df)) / n_raw, 2),
        'mega_clusters': len(mega),
        'mega_cluster_sizes': [r['size'] for r in mega[:5]],
        'largest_cluster': int(sizes.max()) if len(sizes) else 0,
        'median_cluster_size': float(np.median(sizes)) if len(sizes) else 0,
        'seconds': round(time.time() - t0, 1),
        'output': out,
    }
    C.log_json(f'03_dedup_{a.method}_{a.algo}.json', rec)
    print(json.dumps(rec, indent=2), flush=True)


if __name__ == '__main__':
    main()
