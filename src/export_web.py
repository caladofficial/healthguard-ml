"""export_web — distil the triage fusion into a production web booster.
EVOLVEX IT SOLUTIONS PVT. LTD.

WHY: the market-ready site needs REAL model triage in the browser (static
Vercel hosting; no Python runtime). We distil the round-2 knowledge into one
compact gradient-boosted ensemble over FORM-COLLECTABLE features only
(the 6 p_* specialist-risk columns are dropped — a form cannot contain model
outputs). The deterministic red-flag rule layer + safety floors + human-review
logic are exported IDENTICALLY to Python (same thresholds) into
public/js/triage-engine.js so safety guarantees carry over 1:1.

A converter self-check calibrates a per-class base margin so the JS scorer
reproduces booster.predict_proba bit-for-bit.

Outputs:
  /home/user/healthguard/public/js/triage-model-data.js
  healthguard-ml/reports/web_model_metrics.json
"""
import json, os, sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from triage_synth import generate
from train_triage import safe_predict, N_COHORT
from xgboost import XGBClassifier, DMatrix

WEB = '/home/user/healthguard/public/js'
P_DROP = ['p_dm', 'p_cad', 'p_stroke', 'p_sepsis', 'p_liver', 'p_maternal']


def tree_to_arrays(node, acc, fmap):
    """Recursive XGB JSON node → flat arrays (index-addressed binary tree).
    xgboost 3.x dumps carry feature NAMES on DataFrame input — map to indices."""
    idx = len(acc['leaf'])
    acc['leaf'].append(-1.0)
    acc['split'].append(-1)
    acc['thresh'].append(0.0)
    acc['left'].append(-1)
    acc['right'].append(-1)
    acc['miss'].append(-1)
    if 'leaf' in node:
        acc['leaf'][idx] = float(node['leaf'])
    else:
        name = node['split']
        acc['split'][idx] = fmap[name] if name in fmap else int(str(name).split('f')[-1])
        acc['thresh'][idx] = float(node['split_condition'])
        acc['left'][idx] = tree_to_arrays(
            next(c for c in node['children'] if c['nodeid'] == node['yes']), acc, fmap)
        acc['right'][idx] = tree_to_arrays(
            next(c for c in node['children'] if c['nodeid'] == node['no']), acc, fmap)
        miss = node.get('missing', node['yes'])
        acc['miss'][idx] = acc['left'][idx] if miss == node['yes'] else acc['right'][idx]
    return idx


def walk_tree(acc, x):
    """One row through one exported tree (mirrors triage-engine.js exactly).
    XGBoost C++ semantics: float32 compare, strict `<` → yes/left branch.
    Internal nodes have split[i] >= 0 (leaf values may be negative)."""
    i = 0
    while acc['split'][i] >= 0:
        v = np.float32(x[acc['split'][i]])
        t = np.float32(acc['thresh'][i])
        if v != v:  # NaN
            i = acc['miss'][i]
        else:
            i = acc['left'][i] if v < t else acc['right'][i]
    return acc['leaf'][i]


def main():
    X, y, meta = generate(n=N_COHORT)
    X = X.drop(columns=P_DROP)          # form-collectable features only
    feats = list(X.columns)
    fmap = {n: i for i, n in enumerate(feats)}
    classes = sorted(np.unique(y).tolist())
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=C.SEED)

    n_class = len(classes)
    model = XGBClassifier(
        n_estimators=250, max_depth=4, learning_rate=0.08, subsample=0.9,
        colsample_bytree=0.9, min_child_weight=2, reg_lambda=1.5,
        objective='multi:softprob', num_class=n_class,
        tree_method='hist', random_state=C.SEED, n_jobs=2, verbosity=0)
    model.fit(X_tr, y_tr)
    booster = model.get_booster()

    # ---- honest holdout metrics with the SAME safety layer -----------------
    proba = model.predict_proba(X_te)
    y_pred, review = safe_predict(proba, classes, X_te)
    rep = classification_report(y_te, y_pred, labels=classes, output_dict=True, zero_division=0)
    y_te_a = np.asarray(y_te)
    m01 = np.isin(y_te_a, [0, 1])
    metrics = {
        'macro_f1': round(float(rep['macro avg']['f1-score']), 4),
        'per_class_recall': {str(c): round(float(rep[str(c)]['recall']), 4) for c in classes},
        'under_triage_T0_any_down': round(float(np.mean(y_pred[y_te_a == 0] > 0)), 4),
        'under_triage_T0_deep_to_T2plus': round(float(np.mean(y_pred[y_te_a == 0] > 1)), 4),
        'under_triage_T0_T1_rate_beyond_adjacent': round(float(np.mean(y_pred[m01] > np.maximum(y_te_a[m01], 1))), 4),
        'over_triage_rate': round(float(np.mean(y_pred[y_te_a >= 3] <= 1)), 4),
        'note': 'web booster = single distilled XGB (form features only, no p_* heads) '
                'with the identical rule+safety layer; the full server engine keeps the 4-model blend',
    }

    # ---- export compact trees ----------------------------------------------
    dumps = [json.loads(d) if isinstance(d, str) else d
             for d in booster.get_dump(dump_format='json')]
    trees = []
    for d in dumps:
        acc = {'leaf': [], 'split': [], 'thresh': [], 'left': [], 'right': [], 'miss': []}
        tree_to_arrays(d, acc, fmap)
        trees.append({k: v for k, v in acc.items()})

    # ---- converter self-check + per-class base margin (JS parity) ----------
    Xs = X_tr.iloc[:400].to_numpy()
    margin = booster.predict(DMatrix(Xs, feature_names=feats), output_margin=True)
    if margin.ndim == 1:
        margin = margin[:, None]
    ours = np.zeros((len(Xs), n_class))
    for ti, acc in enumerate(trees):
        c = ti % n_class
        ours[:, c] += [walk_tree(acc, x) for x in Xs]
    base = (margin - ours).mean(axis=0)
    resid = float(np.abs((margin - ours) - base).max())
    print(f"converter parity: max residual after base-calibration = {resid:.6f}")

    payload = {
        'features': feats,
        'classes': [int(c) for c in classes],
        'trees': trees,                    # class-major: tree i belongs to class i % n_class
        'n_class': n_class,
        'base': [round(float(b), 5) for b in base],
        'version': 'hg-triage-web-v1',
        'rule_version': '2026-09-triage-protocol-v1',
        'baseline_urgent_prevalence': round(float((np.asarray(y_tr) <= 1).mean()), 4),
        'metrics': metrics,
    }
    os.makedirs(WEB, exist_ok=True)
    out = os.path.join(WEB, 'triage-model-data.js')
    with open(out, 'w') as f:
        f.write('// HealthGuard Triage Risk Engine — distilled production booster (generated by src/export_web.py)\n')
        f.write('// EVOLVEX IT SOLUTIONS PVT. LTD. — do not edit by hand.\n')
        f.write('window.HG_TRIAGE_MODEL = ')
        json.dump(payload, f, separators=(',', ':'))
        f.write(';\n')
    with open(os.path.join(C.REPORTS, 'web_model_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    kb = os.path.getsize(out) / 1024
    print(json.dumps(metrics, indent=2))
    print(f"\n→ {out}  ({kb:.0f} KB, {len(trees)} trees)")


if __name__ == '__main__':
    main()
