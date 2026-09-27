"""Stage 9 - per-site hold-out: does the model generalise to an unseen data source?

Section 10 flagged this as unmeasured: external_real is the only true out-of-domain
test, and it is 1,317 sparse real rows. Here the three synthetic sites stand in for
three hospitals. For each site: train on the other two, score on the held-out one, and
compare against the model that did see that site's data.

The gap is what "generalises to a new hospital" costs, measured rather than assumed.
Everything goes through C.prep so preprocessing matches stage 5 exactly.
"""
import os, sys, json, time
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pipeline_cfg as C          # noqa: E402
from lightgbm import LGBMClassifier

SITES = ['synth_site_a', 'synth_site_b', 'synth_site_c']
LGB_PARAMS = dict(n_estimators=400, num_leaves=31, learning_rate=0.05, subsample=0.9,
                  colsample_bytree=0.9, n_jobs=2, random_state=C.SEED, verbose=-1)


def metrics(y, p):
    return {
        'macro_f1': round(float(f1_score(y, p, average='macro')), 4),
        'accuracy': round(float(np.mean(p == y)), 4),
        'T0_any_down': round(float(np.mean(p[y == 0] > 0)), 4),
        'T0_deep_to_T2plus': round(float(np.mean(p[y == 0] > 1)), 4),
        'over_triage': round(float(np.mean(p[y >= 3] <= 1)), 4),
    }


def fit(Xtr, ytr):
    m = LGBMClassifier(**LGB_PARAMS)
    t0 = time.time()
    m.fit(Xtr, ytr)
    return m, round(time.time() - t0, 1)


def main():
    train = pd.read_parquet(os.path.join(C.PIPE, 'train_curated.parquet'))
    test = pd.read_parquet(os.path.join(C.PIPE, 'test.parquet'))
    synth = train[train['source_id'].isin(SITES)].copy()

    out = {}
    for site in SITES:
        te = test[test['source_id'] == site]
        if len(te) < 200:
            continue
        Xte, yte = C.prep(te), te[C.LABEL].astype(int).values

        # (a) model that has NEVER seen this site
        tr_out = synth[synth['source_id'] != site]
        m_out, s_out = fit(C.prep(tr_out), tr_out[C.LABEL].astype(int).values)

        # (b) model trained on all sites (the shipped setup)
        m_in, s_in = fit(C.prep(synth), synth[C.LABEL].astype(int).values)

        un = metrics(yte, m_out.predict(Xte))
        seen = metrics(yte, m_in.predict(Xte))
        out[site] = {
            'test_rows': int(len(te)),
            'train_rows_unseen': int(len(tr_out)),
            'unseen_site': un,
            'seen_site': seen,
            'delta': {k: round(un[k] - seen[k], 4) for k in seen},
            'fit_seconds': {'unseen': s_out, 'seen': s_in},
        }
        print(f'{site}: test={len(te):,} | unseen macro_f1 {un["macro_f1"]} '
              f'vs seen {seen["macro_f1"]} (delta {un["macro_f1"]-seen["macro_f1"]:+.4f}) '
              f'| T0_down {un["T0_any_down"]} vs {seen["T0_any_down"]}', flush=True)

    rows = [v for v in out.values()]
    rec = {
        'stage': '09_site_holdout',
        'question': 'cost of deploying to a data source the model has never seen',
        'sites': SITES,
        'per_site': out,
        'mean_macro_f1_unseen': round(float(np.mean([r['unseen_site']['macro_f1'] for r in rows])), 4),
        'mean_macro_f1_seen': round(float(np.mean([r['seen_site']['macro_f1'] for r in rows])), 4),
        'mean_T0_down_unseen': round(float(np.mean([r['unseen_site']['T0_any_down'] for r in rows])), 4),
        'mean_T0_down_seen': round(float(np.mean([r['seen_site']['T0_any_down'] for r in rows])), 4),
    }
    rec['generalisation_gap_macro_f1'] = round(
        rec['mean_macro_f1_seen'] - rec['mean_macro_f1_unseen'], 4)
    rec['generalisation_gap_T0_down'] = round(
        rec['mean_T0_down_unseen'] - rec['mean_T0_down_seen'], 4)
    C.log_json('09_site_holdout.json', rec)
    print(json.dumps({k: v for k, v in rec.items() if k != 'per_site'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
