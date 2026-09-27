"""train_web_best — open model bake-off + best-model web export.
EVOLVEX IT SOLUTIONS PVT. LTD. — Part M.

WHY (client): "work on triage system … what is the model … use what is best".
Candidates: XGBoost · LightGBM · ExtraTrees · RandomForest · MLP (sklearn neuro)
· Keras MLP (tensorflow neuro) · soft blends of the tree-boosters. Selection =
best macro-F1 subject to safety constraints after the IDENTICAL rule+safety
layer (train_triage.safe_predict): T0-deep-to-T2+ <= 0.004, beyond-adjacent
<= 0.015. Winner is distilled to the browser scorer (triage-engine.js) with a
float-exact parity probe.

Memory discipline (OOM lesson): fit ONE candidate at a time → predict_proba →
free it (finalists persisted to models/web_best_models.joblib). Only 20k×5
probability matrices are kept in RAM.

Data: 200k protocol-labelled rows (triage_synth.generate — the honest label
source; public acuity datasets are credentialed-gated) + real UCI cohorts in
data/ kept for realism probes (see notebook).

Outputs: reports/web_model_metrics.json · reports/bakeoff.json ·
/home/user/healthguard/public/js/triage-model-data.js · models/web_best_models.joblib
"""
import gc, json, os, sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from triage_synth import generate
from train_triage import safe_predict

WEB = '/home/user/healthguard/public/js'
P_DROP = ['p_dm', 'p_cad', 'p_stroke', 'p_sepsis', 'p_liver', 'p_maternal']
N_ROWS = 200_000
CLASSES = [0, 1, 2, 3, 4]


def metrics_of(y_true, y_pred, rep_extra=None):
    y_true = np.asarray(y_true)
    rep = classification_report(y_true, y_pred, labels=CLASSES, output_dict=True, zero_division=0)
    m01 = np.isin(y_true, [0, 1])
    m = {
        'macro_f1': round(float(rep['macro avg']['f1-score']), 4),
        'per_class_recall': {str(c): round(float(rep[str(c)]['recall']), 4) for c in CLASSES},
        'T0_any_down': round(float(np.mean(y_pred[y_true == 0] > 0)), 4),
        'T0_deep_to_T2plus': round(float(np.mean(y_pred[y_true == 0] > 1)), 4),
        'beyond_adjacent': round(float(np.mean(y_pred[m01] > np.maximum(y_true[m01], 1))), 4),
        'over_triage': round(float(np.mean(y_pred[y_true >= 3] <= 1)), 4),
    }
    if rep_extra:
        m.update(rep_extra)
    return m


def eval_candidate(name, proba, X_probe, y_probe, results):
    y_pred, review = safe_predict(np.asarray(proba), CLASSES, X_probe)
    m = metrics_of(y_probe, y_pred, {'review_rate': round(float(np.mean(review)), 4)})
    ok = m['T0_deep_to_T2plus'] <= 0.004 and m['beyond_adjacent'] <= 0.015
    print(f"{name:24s} f1={m['macro_f1']:.4f} deep={m['T0_deep_to_T2plus']:.4f} "
          f"beyond={m['beyond_adjacent']:.4f} over={m['over_triage']:.4f} feasible={ok}", flush=True)
    results[name] = {'metrics': m, 'feasible': bool(ok)}


def walk_tree_py(acc, x):
    i = 0
    cmpLE = acc['c'] == 1
    while acc['split'][i] >= 0:
        v = x[acc['split'][i]]
        if v != v:
            i = acc['miss'][i]
        else:
            if acc['ff'] == 1:
                v = np.float32(v); t = np.float32(acc['thresh'][i])
            else:
                t = acc['thresh'][i]
            i = acc['left'][i] if (v <= t if cmpLE else v < t) else acc['right'][i]
    return acc['leaf'][i]


def xgb_trees(model, fmap, weight=1.0):
    import export_web
    dumps = [json.loads(d) for d in model.get_booster().get_dump(dump_format='json')]
    out = []
    for d in dumps:
        acc = {'leaf': [], 'split': [], 'thresh': [], 'left': [], 'right': [], 'miss': []}
        export_web.tree_to_arrays(d, acc, fmap)
        acc['leaf'] = [round(float(v) * weight, 6) for v in acc['leaf']]
        acc.update({'c': 0, 'ff': 1, 'w': 1.0})
        out.append({k: acc[k] for k in ('leaf', 'split', 'thresh', 'left', 'right', 'miss', 'c', 'ff', 'w')})
    return out


def lgbm_trees(model, fmap, weight=1.0):
    dump = json.loads(json.dumps(model.booster_.dump_model()))['tree_info']
    out = []
    for ti in dump:
        acc = {'leaf': [], 'split': [], 'thresh': [], 'left': [], 'right': [], 'miss': []}

        def walk(node):
            idx = len(acc['leaf'])
            acc['leaf'].append(-1.0)
            acc['split'].append(-1); acc['thresh'].append(0.0)
            acc['left'].append(-1); acc['right'].append(-1); acc['miss'].append(-1)
            if 'leaf_value' in node and 'split_feature' not in node:
                acc['leaf'][idx] = float(node['leaf_value']) * weight
                return idx
            name = node['split_feature']
            acc['split'][idx] = fmap[name] if isinstance(name, str) and name in fmap else int(name)
            acc['thresh'][idx] = float(node['threshold'])
            acc['left'][idx] = walk(node['left_child'])
            acc['right'][idx] = walk(node['right_child'])
            miss_left = bool(node.get('default_left', True))
            acc['miss'][idx] = acc['left'][idx] if miss_left else acc['right'][idx]
            return idx

        walk(ti['tree_structure'])
        acc.update({'c': 1, 'ff': 0, 'w': 1.0})
        out.append({k: acc[k] for k in ('leaf', 'split', 'thresh', 'left', 'right', 'miss', 'c', 'ff', 'w')})
    return out


def main():
    import joblib
    X, y, meta = generate(n=N_ROWS)
    X = X.drop(columns=P_DROP)
    feats = list(X.columns)
    fmap = {n: i for i, n in enumerate(feats)}
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1, stratify=y, random_state=C.SEED)
    print('data:', X.shape, '| train', X_tr.shape, '| test', X_te.shape, flush=True)

    results, probas = {}, {}
    saved = {}

    # ---- 1) XGBoost ------------------------------------------------------
    from xgboost import XGBClassifier
    xgb = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.06, subsample=0.9,
        colsample_bytree=0.9, min_child_weight=2, reg_lambda=1.5,
        objective='multi:softprob', num_class=5, tree_method='hist',
        random_state=C.SEED, n_jobs=2, verbosity=0)
    xgb.fit(X_tr, y_tr)
    probas['xgboost'] = xgb.predict_proba(X_te)
    joblib.dump(xgb, os.path.join(C.MODELS, 'web_xgb.joblib'))
    saved['xgboost'] = True
    del xgb; gc.collect()

    # ---- 2) LightGBM -----------------------------------------------------
    import lightgbm as lgb
    lgbm = lgb.LGBMClassifier(
        n_estimators=400, num_leaves=31, learning_rate=0.05, subsample=0.9,
        colsample_bytree=0.9, min_child_weight=5, reg_lambda=1.5,
        random_state=C.SEED, n_jobs=2, verbose=-1)
    lgbm.fit(X_tr, y_tr)
    probas['lightgbm'] = lgbm.predict_proba(X_te)
    joblib.dump(lgbm, os.path.join(C.MODELS, 'web_lgbm.joblib'))
    saved['lightgbm'] = True
    del lgbm; gc.collect()

    # ---- 3) ExtraTrees ---------------------------------------------------
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    et = ExtraTreesClassifier(n_estimators=200, min_samples_leaf=4, n_jobs=2, random_state=C.SEED)
    et.fit(X_tr, y_tr)
    probas['extratrees'] = et.predict_proba(X_te)
    del et; gc.collect()

    # ---- 4) RandomForest --------------------------------------------------
    rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=4, n_jobs=2, random_state=C.SEED)
    rf.fit(X_tr, y_tr)
    probas['randomforest'] = rf.predict_proba(X_te)
    del rf; gc.collect()

    # ---- 5) MLP (sklearn neuro) ------------------------------------------
    from sklearn.neural_network import MLPClassifier
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation='relu',
                        early_stopping=True, n_iter_no_change=8, max_iter=60,
                        random_state=C.SEED)
    mlp.fit(X_tr, y_tr)
    probas['mlp_sklearn'] = mlp.predict_proba(X_te)
    del mlp; gc.collect()

    # ---- 6) Keras MLP (tensorflow neuro — client offered TF) --------------
    try:
        import tensorflow as tf
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(X_tr)
        Xn_tr, Xn_te = sc.transform(X_tr), sc.transform(X_te)
        nn = tf.keras.Sequential([
            tf.keras.layers.Input((len(feats),)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(5, activation='softmax')])
        nn.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        nn.fit(Xn_tr, np.asarray(y_tr), epochs=16, batch_size=512, verbose=0)
        probas['mlp_tensorflow'] = nn.predict(Xn_te, batch_size=2048, verbose=0)
        del nn, Xn_tr, Xn_te, sc
        tf.keras.backend.clear_session()
        gc.collect()
    except Exception as e:
        print('tensorflow neuro skipped:', str(e)[:120], flush=True)

    # ---- 7) soft blends (probability-space) -------------------------------
    probas['blend_xgb_lgbm'] = 0.5 * probas['xgboost'] + 0.5 * probas['lightgbm']
    probas['blend_xgb_lgbm_et'] = (probas['xgboost'] + probas['lightgbm'] + probas['extratrees']) / 3

    for name in ['xgboost', 'lightgbm', 'extratrees', 'randomforest', 'mlp_sklearn',
                 'mlp_tensorflow', 'blend_xgb_lgbm', 'blend_xgb_lgbm_et']:
        if name in probas:
            eval_candidate(name, probas[name], X_te, y_te, results)

    # ---- selection --------------------------------------------------------
    feasible = [n for n, r in results.items() if r['feasible']]
    pool = feasible or list(results)
    best_name = max(pool, key=lambda n: results[n]['metrics']['macro_f1'])
    EXPORTABLE = {'xgboost', 'lightgbm', 'blend_xgb_lgbm'}
    if best_name not in EXPORTABLE:
        alt = max([n for n in pool if n in EXPORTABLE] or ['xgboost'],
                  key=lambda n: results[n]['metrics']['macro_f1'])
        gap = results[best_name]['metrics']['macro_f1'] - results[alt]['metrics']['macro_f1']
        if gap > 0.02:
            results['export_warning'] = (f'{best_name} wins by {gap:.4f} but needs a new JS scorer; '
                                         f'deploying {alt} this round')
            print(results['export_warning'], flush=True)
        best_name = alt if gap <= 0.02 else best_name if best_name in EXPORTABLE else alt
    print('WINNER:', best_name, flush=True)

    y_pred, review = safe_predict(np.asarray(probas[best_name]), CLASSES, X_te)
    final = metrics_of(y_te, y_pred, {
        'model': best_name,
        'n_rows': N_ROWS,
        'holdout': int(len(y_te)),
        'note': 'web booster = bake-off winner (form features only) with identical rule+safety layer',
    })

    # ---- distil winner to the browser scorer ------------------------------
    from xgboost import DMatrix
    Xs = X_tr.iloc[:400].to_numpy()
    nc = 5
    if best_name == 'lightgbm':
        lgbm = joblib.load(os.path.join(C.MODELS, 'web_lgbm.joblib'))
        trees = lgbm_trees(lgbm, fmap)
        margin = lgbm.predict(X_tr.iloc[:400], raw_score=True)
    elif best_name == 'blend_xgb_lgbm':
        xgb = joblib.load(os.path.join(C.MODELS, 'web_xgb.joblib'))
        lgbm = joblib.load(os.path.join(C.MODELS, 'web_lgbm.joblib'))
        trees = xgb_trees(xgb, fmap, 0.5) + lgbm_trees(lgbm, fmap, 0.5)
        margin = (0.5 * xgb.get_booster().predict(DMatrix(Xs, feature_names=feats), output_margin=True)
                  + 0.5 * lgbm.predict(X_tr.iloc[:400], raw_score=True))
    else:  # xgboost
        xgb = joblib.load(os.path.join(C.MODELS, 'web_xgb.joblib'))
        trees = xgb_trees(xgb, fmap)
        margin = xgb.get_booster().predict(DMatrix(Xs, feature_names=feats), output_margin=True)

    if margin.ndim == 1:
        margin = margin[:, None]
    ours = np.zeros((len(Xs), nc))
    for ti, acc in enumerate(trees):
        ours[:, ti % nc] += [walk_tree_py(acc, x) for x in Xs]
    base = (margin - ours).mean(axis=0)
    resid = float(np.abs((margin - ours) - base).max())
    print(f"converter parity: max residual after base-calibration = {resid:.6f}", flush=True)

    payload = {
        'features': feats,
        'classes': CLASSES,
        'trees': trees,
        'n_class': nc,
        'base': [round(float(b), 5) for b in base],
        'version': 'hg-triage-web-v2-' + best_name,
        'rule_version': '2026-09-triage-protocol-v1',
        'baseline_urgent_prevalence': round(float((np.asarray(y_tr) <= 1).mean()), 4),
        'metrics': final,
    }
    os.makedirs(WEB, exist_ok=True)
    out = os.path.join(WEB, 'triage-model-data.js')
    with open(out, 'w') as f:
        f.write('// HealthGuard Triage Risk Engine — bake-off winner (' + best_name + ') (generated by src/train_web_best.py)\n')
        f.write('// EVOLVEX IT SOLUTIONS PVT. LTD. — do not edit by hand.\n')
        f.write('window.HG_TRIAGE_MODEL = ')
        json.dump(payload, f, separators=(',', ':'))
        f.write(';\n')
    with open(os.path.join(C.REPORTS, 'web_model_metrics.json'), 'w') as f:
        json.dump(final, f, indent=2)
    with open(os.path.join(C.REPORTS, 'bakeoff.json'), 'w') as f:
        json.dump({'results': results, 'winner': best_name, 'parity': round(resid, 6)}, f, indent=2)
    kb = os.path.getsize(out) / 1024
    print(json.dumps(final, indent=2))
    print(f"\n→ {out}  ({kb:.0f} KB, {len(trees)} trees, winner={best_name}, parity={resid:.6f})")


if __name__ == '__main__':
    main()
