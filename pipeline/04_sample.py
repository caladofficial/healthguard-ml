"""Stage 4 — BALANCED SAMPLING with cluster-disjoint splits.

Why cluster-disjoint: if validation rows are near-duplicates of training rows the
score is contaminated and always looks better than reality. So we split by CLUSTER
ID, not by row: whole clusters go to train / val / test.

We also build a RAW CONTROL: the same number of training rows, drawn from the same
training clusters but WITHOUT de-duplication or per-cluster capping. Training the
same model on both isolates the effect of curation — same test set, same size,
same clusters; the only difference is whether redundancy was removed.
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import pipeline_cfg as C          # noqa: E402

P_DROP = ['p_dm', 'p_cad', 'p_stroke', 'p_sepsis', 'p_liver', 'p_maternal']
BARE = [c for c in C.FEATURES if c not in P_DROP]
TRAIN_FRAC, VAL_FRAC = 0.80, 0.10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', default='raw.parquet')
    ap.add_argument('--clusters', type=int, default=4000)
    ap.add_argument('--method', default='numeric')
    ap.add_argument('--algo', default='kmeans')
    a = ap.parse_args()

    cur = pd.read_parquet(os.path.join(C.PIPE, f'curated_{a.method}_{a.algo}.parquet'))
    # clustered = every deduplicated row WITH its cluster id; this is the pool the
    # control group is drawn from, so control vs curated differ ONLY by curation
    raw = pd.read_parquet(os.path.join(C.PIPE, f'clustered_{a.method}_{a.algo}.parquet'))

    # real-world proxy rows: never trained on, kept only as an external slice.
    # Sourced from the ORIGINAL raw corpus: after harmonisation ~90% of their
    # features are missing, so exact-dedup collapsed 1,317 rows to 1 unique
    # pattern - they are far too degenerate to train on.
    raw0 = pd.read_parquet(os.path.join(C.PIPE, a.corpus))
    ext = raw0[raw0['source_id'].isin(['uci_863', 'uci_45'])].copy()

    rng = np.random.default_rng(C.SEED)
    clusters = np.sort(cur['cluster'].unique())
    rng.shuffle(clusters)
    ntr, nva = int(TRAIN_FRAC * len(clusters)), int(VAL_FRAC * len(clusters))
    tr_c, va_c, te_c = set(clusters[:ntr]), set(clusters[ntr:ntr + nva]), set(clusters[ntr + nva:])

    train = cur[cur['cluster'].isin(tr_c)].copy()
    val = cur[cur['cluster'].isin(va_c)].copy()
    test = cur[cur['cluster'].isin(te_c)].copy()

    # raw control: same clusters, same size, redundancy left in
    pool = raw[raw['cluster'].isin(tr_c) & raw['source_id'].str.startswith('synth')]
    assert len(pool) >= len(train), f'pool {len(pool)} < train {len(train)}'
    raw_control = pool.sample(n=len(train), random_state=C.SEED).copy()

    for name, d in [('train_curated', train), ('train_raw_control', raw_control),
                    ('val', val), ('test', test), ('external_real', ext)]:
        d.reset_index(drop=True).to_parquet(os.path.join(C.PIPE, f'{name}.parquet'), index=False)

    def cls(d):
        return {C.CLASSES[k]: int(v) for k, v in d[C.LABEL].value_counts().sort_index().items()}

    rec = {
        'stage': '04_sample',
        'split_rule': 'cluster-disjoint (whole clusters assigned to train/val/test)',
        'clusters_total': int(len(clusters)),
        'clusters_train': len(tr_c), 'clusters_val': len(va_c), 'clusters_test': len(te_c),
        'train_curated_rows': int(len(train)),
        'train_raw_control_rows': int(len(raw_control)),
        'val_rows': int(len(val)), 'test_rows': int(len(test)),
        'external_real_rows': int(len(ext)),
        'external_real_unique_patterns': int(ext['row_hash'].nunique()),
        'external_real_note': ('harmonised real rows: ~90% features missing, so exact '
                               'dedup collapsed them to 1 unique pattern -> excluded from '
                               'training, kept only as an external rule-agreement slice'),
        'train_class_counts': cls(train), 'test_class_counts': cls(test),
        'raw_control_class_counts': cls(raw_control),
        'max_rows_per_cluster_in_train': int(train.groupby('cluster').size().max()),
        'raw_control_max_rows_per_cluster': int(raw_control['cluster'].value_counts().max()),
        'raw_control_distinct_clusters': int(raw_control['cluster'].nunique()),
        'note': 'raw_control drawn from the SAME train clusters so only curation differs',
    }
    C.log_json('04_sampling.json', rec)
    print(json.dumps(rec, indent=2))


if __name__ == '__main__':
    main()
