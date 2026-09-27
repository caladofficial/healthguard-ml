"""Stage 5 — FINE-TUNING.

Method choice (compute budget): full gradient-boosted trees, not LoRA. The task is
45-column tabular clinical risk scoring on a 2-CPU / 2 GB box; a tree ensemble is
both the right model class and the only one that trains here. The LoRA/QLoRA path
for a text model is provided in 08_lora_finetune.py for a GPU machine.

Every run logs: dataset, model, hyperparameters, wall/CPU seconds, loss curves and
the eval metrics. The train/val/test split was FIXED in stage 4 (cluster-disjoint)
before any of this, so no hyperparameter choice can leak from the test set.
"""
import os, sys, time, json, argparse, resource
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import pipeline_cfg as C          # noqa: E402

P_DROP = ['p_dm', 'p_cad', 'p_stroke', 'p_sepsis', 'p_liver', 'p_maternal']
BARE = C.BARE     # 45 features, as the shipped model

# Hyperparameters copied from the shipped bake-off winner so the comparison is
# apples-to-apples: only the DATA changes between the runs below.
XGB_PARAMS = dict(n_estimators=300, max_depth=5, learning_rate=0.06, subsample=0.9,
                  colsample_bytree=0.9, n_jobs=2, random_state=C.SEED,
                  tree_method='hist', eval_metric='mlogloss')
LGB_PARAMS = dict(n_estimators=400, num_leaves=31, learning_rate=0.05, subsample=0.9,
                  colsample_bytree=0.9, n_jobs=2, random_state=C.SEED, verbose=-1)


def load_split(name):
    d = pd.read_parquet(os.path.join(C.PIPE, f'{name}.parquet'))
    return C.prep(d), d[C.LABEL].astype(int).values


def safety_metrics(y_true, y_pred, probs):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
    yt = np.asarray(y_true)
    return {
        'macro_f1': round(float(f1_score(yt, y_pred, average='macro')), 4),
        'weighted_f1': round(float(f1_score(yt, y_pred, average='weighted')), 4),
        'per_class_recall': {f'T{c}': round(float(cm[c, c] / max(cm[c].sum(), 1)), 4)
                             for c in range(5)},
        'T0_any_down': round(float(np.mean(y_pred[yt == 0] > 0)) if (yt == 0).any() else 0.0, 4),
        'T0_deep_to_T2plus': round(float(np.mean(y_pred[yt == 0] > 1)) if (yt == 0).any() else 0.0, 4),
        'beyond_adjacent': round(float(np.mean(
            y_pred[np.isin(yt, [0, 1])] > np.maximum(yt[np.isin(yt, [0, 1])], 1))), 4),
        'over_triage': round(float(np.mean(y_pred[yt >= 3] <= 1)), 4),
        'confusion': cm.tolist(),
    }


def fit_xgb(Xtr, ytr, Xva, yva):
    import xgboost as xgb
    m = xgb.XGBClassifier(**XGB_PARAMS)
    t0 = time.time()
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    history = m.evals_result_['validation_0']['mlogloss']
    return m, time.time() - t0, history, 'xgboost'


def fit_lgb(Xtr, ytr, Xva, yva):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(**LGB_PARAMS)
    t0 = time.time()
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)],
          callbacks=[lgb.log_evaluation(0)])
    history = []
    return m, time.time() - t0, history, 'lightgbm'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='xgb', choices=['xgb', 'lgbm', 'blend'])
    ap.add_argument('--dataset', default='train_curated',
                    choices=['train_curated', 'train_raw_control'])
    ap.add_argument('--tag', default='')
    a = ap.parse_args()

    Xtr, ytr = load_split(a.dataset)
    Xva, yva = load_split('val')
    Xte, yte = load_split('test')

    runs, probs_parts = {}, {}
    if a.model in ('xgb', 'blend'):
        m, dt, hist, name = fit_xgb(Xtr, ytr, Xva, yva)
        runs['xgb'], probs_parts['xgb'] = (m, dt, hist, name), m.predict_proba(Xte)
    if a.model in ('lgbm', 'blend'):
        m, dt, hist, name = fit_lgb(Xtr, ytr, Xva, yva)
        runs['lgbm'], probs_parts['lgbm'] = (m, dt, hist, name), m.predict_proba(Xte)

    if a.model == 'blend':
        probs = 0.5 * probs_parts['xgb'] + 0.5 * probs_parts['lgbm']
    else:
        probs = probs_parts[a.model]

    ypred = probs.argmax(axis=1)
    mets = safety_metrics(yte, ypred, probs)
    ext = pd.read_parquet(os.path.join(C.PIPE, 'external_real.parquet'))
    ext_pred, ext_agree = None, None
    if len(ext):
        Xe = ext[BARE].astype('float32')
        Xe = Xe.fillna(Xe.median())
        best = runs['xgb'][0] if 'xgb' in runs else runs['lgbm'][0]
        ext_pred = best.predict(Xe)
        ext_agree = round(float((ext_pred == ext[C.LABEL].values).mean()), 4)

    cpu = resource.getrusage(resource.RUSAGE_SELF).ru_utime + \
        resource.getrusage(resource.RUSAGE_SELF).ru_stime
    rec = {
        'stage': '05_finetune', 'dataset': a.dataset, 'model': a.model,
        'tag': a.tag,
        'n_train': int(len(Xtr)), 'n_val': int(len(Xva)), 'n_test': int(len(Xte)),
        'n_features': int(Xtr.shape[1]),
        'split': 'cluster-disjoint (fixed in stage 4 before any tuning)',
        'hyperparameters': {'xgb': XGB_PARAMS, 'lgbm': LGB_PARAMS} if a.model == 'blend'
                            else (XGB_PARAMS if a.model == 'xgb' else LGB_PARAMS),
        'fit_seconds': {k: round(v[1], 1) for k, v in runs.items()},
        'cpu_seconds_total': round(cpu, 1),
        'loss_curve_last': {k: (round(v[2][-1], 4) if v[2] else None) for k, v in runs.items()},
        'loss_curve_first': {k: (round(v[2][0], 4) if v[2] else None) for k, v in runs.items()},
        'test_metrics': mets,
        'external_real_rule_agreement': ext_agree,
        'external_real_rows': int(len(ext)),
    }
    tag = a.tag or f'{a.model}_{a.dataset}'
    C.log_json(f'05_finetune_{tag}.json', rec)
    print(json.dumps({k: v for k, v in rec.items() if k != 'hyperparameters'}, indent=2),
          flush=True)

    os.makedirs(os.path.join(ROOT, 'models', 'pipeline'), exist_ok=True)
    for k, (m, _dt, _h, _n) in runs.items():
        import joblib
        joblib.dump(m, os.path.join(ROOT, 'models', 'pipeline', f'{tag}_{k}.joblib'),
                    compress=3)


if __name__ == '__main__':
    main()
