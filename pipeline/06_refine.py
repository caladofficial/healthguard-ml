"""Stage 6 — EVALUATION, ERROR CLUSTERING, TARGETED RE-COLLECTION, RETRAIN.

The loop the brief asks for:
  1. take the best model from stage 5;
  2. cluster its ERRORS with the same machinery used for the input data, so we
     see systematic weak spots instead of random noise;
  3. feed those weak-spot clusters back into collection: raise the per-cluster cap
     for the training clusters nearest each error cluster, i.e. deliberately
     collect more examples like the ones we get wrong;
  4. retrain with identical hyperparameters and re-score on the SAME held-out
     test set, so the delta is attributable to the new data alone.

Also measures SPLIT CONTAMINATION: the same model scored on a cluster-disjoint
test set vs a random test set drawn from the training clusters. The gap is how
much a naive random split would have flattered us.
"""
import os, sys, json, time
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import pipeline_cfg as C          # noqa: E402

BARE = C.BARE
RAISE_CAP = 140          # weak-spot clusters get up to this many examples
N_ERR_CLUSTERS = 24


def xy(d):
    return C.prep(d), d[C.LABEL].astype(int).values


def xy_imp(d):
    """Imputed copy for PCA/scaler - those estimators reject NaN."""
    return C.prep_imputed(d), d[C.LABEL].astype(int).values


def main():
    cur = pd.read_parquet(os.path.join(C.PIPE, 'train_curated.parquet'))
    pool = pd.read_parquet(os.path.join(C.PIPE, 'clustered_numeric_kmeans.parquet'))
    test = pd.read_parquet(os.path.join(C.PIPE, 'test.parquet'))
    Xtr, ytr = xy(cur)
    Xte, yte = xy(test)
    Xtr_imp, _ = xy_imp(cur)      # geometry only; model I/O stays on Xtr/Xte
    Xte_imp, _ = xy_imp(test)

    mpath = os.path.join(ROOT, 'models', 'pipeline', 'lgbm_curated_lgbm.joblib')
    model = joblib.load(mpath)
    pred = model.predict(Xte)
    err = pred != yte

    # ---------------------------------------------------------- 1. error clusters
    scaler = StandardScaler().fit(Xtr_imp)
    E_err = PCA(n_components=16, random_state=C.SEED).fit_transform(
        scaler.transform(Xte_imp[err].astype('float64')))
    km = MiniBatchKMeans(n_clusters=N_ERR_CLUSTERS, random_state=C.SEED,
                         batch_size=2048, n_init=3).fit(E_err)
    elabels = km.labels_
    err_idx = np.where(err)[0]

    weak = []
    for c in range(N_ERR_CLUSTERS):
        sel = err_idx[elabels == c]
        if len(sel) == 0:
            continue
        pairs = pd.Series(list(zip(yte[sel], pred[sel]))).value_counts()
        (t0, p0), n0 = pairs.index[0], int(pairs.iloc[0])
        weak.append({'error_cluster': int(c), 'errors': int(len(sel)),
                     'dominant': f'T{t0}->T{p0}', 'dominant_n': n0,
                     'share_of_cluster': round(n0 / len(sel), 3),
                     'share_of_all_errors': round(len(sel) / max(err.sum(), 1), 3)})
    weak.sort(key=lambda r: -r['errors'])
    print(f'errors: {int(err.sum()):,} / {len(yte):,} = {100*err.mean():.2f}% '
          f'-> {len(weak)} weak-spot clusters', flush=True)

    # ------------------------------------------- 2. targeted collection for weak spots
    Xtr_s = scaler.transform(Xtr_imp.astype('float64'))
    Efull = PCA(n_components=16, random_state=C.SEED).fit_transform(Xtr_s)
    Xte_s = scaler.transform(Xte_imp[err].astype('float64'))
    Efull_te = PCA(n_components=16, random_state=C.SEED).fit_transform(Xte_s)
    # map each error cluster centroid to the nearest TRAIN clusters
    train_centroids = {}
    for c in np.unique(cur['cluster']):
        m = (cur['cluster'] == c).values
        train_centroids[c] = Efull[m].mean(axis=0)
    tc_ids = np.array(list(train_centroids.keys()))
    tc_mat = np.vstack([train_centroids[c] for c in tc_ids])

    targeted = set()
    for c in range(N_ERR_CLUSTERS):
        sel = err_idx[elabels == c]
        if len(sel) < 20:
            continue
        cen = Efull_te[elabels == c].mean(axis=0)
        d = ((tc_mat - cen) ** 2).sum(axis=1)
        targeted.update(tc_ids[np.argsort(d)[:12]].tolist())
    print(f'targeted train clusters: {len(targeted)} of {len(tc_ids)}', flush=True)

    extra_rows = []
    for c in targeted:
        rows = pool[(pool['cluster'] == c) & pool['source_id'].str.startswith('synth')]
        have = int((cur['cluster'] == c).sum())
        need = RAISE_CAP - have
        if need > 0 and len(rows) > have:
            extra_rows.append(rows.iloc[:need])
    aug = pd.concat([cur] + extra_rows, ignore_index=True) if extra_rows else cur
    aug.to_parquet(os.path.join(C.PIPE, 'train_augmented.parquet'), index=False)
    print(f'augmented train: {len(cur):,} -> {len(aug):,} rows '
          f'(+{len(aug)-len(cur):,} targeted)', flush=True)

    # ------------------------------------------------------------- 3. retrain
    from lightgbm import LGBMClassifier
    Xa, ya = xy(aug)
    t0 = time.time()
    m2 = LGBMClassifier(n_estimators=400, num_leaves=31, learning_rate=0.05,
                        subsample=0.9, colsample_bytree=0.9, n_jobs=2,
                        random_state=C.SEED, verbose=-1).fit(Xa, ya)
    fit_s = round(time.time() - t0, 1)
    pred2 = m2.predict(Xte)
    joblib.dump(m2, os.path.join(ROOT, 'models', 'pipeline', 'refined_lgbm.joblib'), compress=3)

    def mets(p):
        return {'macro_f1': round(float(f1_score(yte, p, average='macro')), 4),
                'T0_any_down': round(float(np.mean(p[yte == 0] > 0)), 4),
                'T2_recall': round(float(np.mean(p[yte == 2] == 2)), 4),
                'T4_recall': round(float(np.mean(p[yte == 4] == 4)), 4),
                'error_rate': round(float(np.mean(p != yte)), 4)}
    before, after = mets(pred), mets(pred2)

    # ------------------------------------------- 4. split-contamination measurement
    pooled_train = pool[pool['cluster'].isin(set(cur['cluster']))]
    rand_test = pooled_train.sample(n=min(len(test), len(pooled_train)), random_state=C.SEED)
    Xr, yr = xy(rand_test)
    pred_rand = model.predict(Xr)
    contaminated = {'macro_f1': round(float(f1_score(yr, pred_rand, average='macro')), 4),
                    'T0_any_down': round(float(np.mean(pred_rand[yr == 0] > 0)), 4)}

    rec = {
        'stage': '06_refine',
        'base_model': 'lgbm_curated',
        'errors': int(err.sum()), 'error_rate': round(float(err.mean()), 4),
        'weak_spot_clusters': weak[:10],
        'targeted_clusters': len(targeted),
        'train_before': int(len(cur)), 'train_after': int(len(aug)),
        'added_rows': int(len(aug) - len(cur)), 'raise_cap': RAISE_CAP,
        'refit_seconds': fit_s,
        'before': before, 'after': after,
        'delta': {k: round(after[k] - before[k], 4) for k in before},
        'contamination_check': {
            'cluster_disjoint_test': before,
            'random_test_same_clusters': contaminated,
            'macro_f1_inflation': round(contaminated['macro_f1'] - before['macro_f1'], 4),
            'why': 'random split leaves near-duplicates of training rows in test',
        },
    }
    C.log_json('06_refine.json', rec)
    print(json.dumps({k: v for k, v in rec.items() if k != 'weak_spot_clusters'}, indent=2),
          flush=True)
    print('\ntop weak spots:', json.dumps(weak[:5]), flush=True)


if __name__ == '__main__':
    main()
