"""07 — TRIAGE FUSION MODEL (risk → T0–T4 with safety thresholds + rules).
EVOLVEX IT SOLUTIONS PVT. LTD. — round 2: 100k cohort, 4-model blend with
validation-grid weights, engineered clinical features, memory-safe groups.

Blend: XGB + LightGBM + soft-vote group (XGB+LGBM+RF) + de-clustered MoE.
Safety layer:
  · deterministic red-flag rules ALWAYS override upward (spec §5)
  · class thresholds biased to near-zero under-triage of T0/T1
  · disagreement(model, rules) → requires_human_review
"""
import json, os, sys
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, log_loss
from lightgbm import LGBMClassifier
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from triage_synth import generate
from cluster import phenotype_report, cluster_features
from learners import soft_vote, MoEDecluster, fast_search_xgb, calibrated

CRITICAL_RULES = ['SHOCK_HYPOTENSION', 'SEVERE_BRADYCARDIA', 'SEVERE_TACHYCARDIA', 'HYPOXEMIA',
                  'GCS_CRITICAL', 'ACTIVE_SEIZURE', 'ACS_RED_FLAG', 'HEMORRHAGE_OBSTETRIC',
                  'SEPSIS_QSOFA', 'STROKE_FAST', 'HYPERTENSIVE_CRISIS']

N_COHORT = 100_000


def rule_scan(row):
    """Deterministic red-flag engine (works on raw features; mirrors protocol)."""
    r = []
    if row['sbp'] < 90: r.append('SHOCK_HYPOTENSION')
    if row['sbp'] > 220: r.append('HYPERTENSIVE_CRISIS')
    if row['hr'] < 40: r.append('SEVERE_BRADYCARDIA')
    if row['hr'] > 150: r.append('SEVERE_TACHYCARDIA')
    if row['spo2'] < 88: r.append('HYPOXEMIA')
    if row['gcs'] <= 8: r.append('GCS_CRITICAL')
    if row['seizure'] == 1 and row['pain'] >= 6: r.append('ACTIVE_SEIZURE')
    if (row['chest_pain'] == 1 and row['age'] > 45 and (row['dm'] == 1 or row['cad_hf'] == 1)
            and row['pain'] >= 6 and row['onset_hours'] < 24):
        r.append('ACS_RED_FLAG')
    if row['pregnancy'] == 1 and row['bleeding'] == 1 and row['sbp'] < 100:
        r.append('HEMORRHAGE_OBSTETRIC')
    if row['sbp'] > 180 and row['htn'] == 1 and row['headache'] == 1: r.append('HYPERTENSIVE_CRISIS')
    qsofa = int(row['sbp'] <= 100) + int(row['rr'] >= 22) + int(row['altered_mental'] == 1)
    if qsofa >= 2 and row['fever'] == 1: r.append('SEPSIS_QSOFA')
    if row['onset_hours'] < 3 and row['altered_mental'] == 1 and (row['weakness'] == 1 or row['dizziness'] == 1) and row['age'] > 50:
        r.append('STROKE_FAST')
    return r


def safe_predict(proba, classes, X):
    """argmax + safety floors + red-flag override → (y_pred, review)."""
    preds, reviews = [], []
    for i in range(len(X)):
        p = dict(zip(classes, proba[i]))
        s01 = p.get(0, 0) + p.get(1, 0)
        s012 = s01 + p.get(2, 0)
        pred = int(max(p, key=p.get))
        if s01 >= 0.30: pred = min(pred, 1)
        elif s01 >= 0.14: pred = min(pred, 2)
        if s012 >= 0.55: pred = min(pred, 2)
        r = rule_scan(X.iloc[i])
        if r:  # rules always escalate upward
            pred = 0 if any(c in CRITICAL_RULES for c in r) else min(pred, 1)
        model_wants_urgent = s01 > 0.25
        reviews.append(bool(r and not model_wants_urgent) or bool(model_wants_urgent and not r and pred >= 3))
        preds.append(pred)
    return np.array(preds), np.array(reviews)


def blend_proba(models, weights, X, X_aug, *extra):
    probs = None
    for w, (mname, m) in zip(weights, models):
        pm = m.predict_proba(X, X_aug) if type(m).__name__ == 'MoEDecluster' else m.predict_proba(X_aug)
        if probs is None:
            probs = np.zeros_like(pm)
        probs += w * pm
    return probs


def pick_weights(models, X_val, Xval_aug, y_val, classes):
    """Coarse simplex grid — optimize macro-F1 minus a safety penalty for
    under-triage beyond the adjacent band."""
    from sklearn.metrics import f1_score
    best_w, best_s = None, -1e9
    cand = []
    for a in range(1, 6):
        for b in range(0, 6 - a):
            for c2 in range(0, 6 - a - b):
                d = 6 - a - b - c2
                if d < 1: continue
                cand.append((a/6, b/6, c2/6, d/6))
    for w in cand:
        probs = blend_proba(models, w, X_val, Xval_aug, y_val, classes)
        yp, _ = safe_predict(probs, classes, X_val)
        f1 = f1_score(y_val, yp, average='macro', zero_division=0)
        m01 = np.isin(y_val, [0, 1])
        deep = float(np.mean(yp[m01] > np.maximum(np.asarray(y_val)[m01], 1)))
        s = f1 - 2.0 * deep
        if s > best_s:
            best_s, best_w = s, w
    return list(best_w), round(float(best_s), 4)


def main():
    X, y, meta = generate(n=N_COHORT)
    feats = list(X.columns)
    X_tr, X_te, y_tr, y_te, m_tr, m_te = train_test_split(
        X, y, meta, test_size=0.2, stratify=y, random_state=C.SEED)
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_tr, y_tr, test_size=0.15, stratify=y_tr, random_state=C.SEED)

    km, gmm, scaler, creport = phenotype_report(X_fit, y_fit)
    def aug(df):
        return pd.concat([df.reset_index(drop=True), cluster_features(df, km, gmm, scaler)], axis=1)
    Xfit_aug, Xval_aug, Xte_aug = aug(X_fit), aug(X_val), aug(X_te)

    classes = sorted(np.unique(y).tolist())
    tuned, params = fast_search_xgb(Xfit_aug, y_fit, len(classes))
    gbm = calibrated(tuned, cv=3, n=len(y_fit)).fit(Xfit_aug, y_fit)
    lgb = calibrated(LGBMClassifier(n_estimators=400, num_leaves=31, learning_rate=0.05,
                                    subsample=0.9, colsample_bytree=0.9,
                                    random_state=C.SEED, n_jobs=2, verbose=-1),
                     cv=3, n=len(y_fit)).fit(Xfit_aug, y_fit)
    grp = calibrated(soft_vote('multiclass', len(classes)), cv=3, n=len(y_fit)).fit(Xfit_aug, y_fit)
    moe = MoEDecluster(km, gmm, scaler, 'multiclass', len(classes)).fit(X_fit, y_fit, Xfit_aug)
    print("base models fitted", flush=True)

    models = [('xgb', gbm), ('lgbm', lgb), ('group_vote', grp), ('moe', moe)]
    weights, score = pick_weights(models, X_val, Xval_aug, y_val, classes)
    print(f"blend weights {weights} val_score {score}", flush=True)

    p = blend_proba(models, weights, X_te, Xte_aug, y_te, classes)
    y_pred, review = safe_predict(p, classes, X_te)

    # ---- safety metrics ----------------------------------------------------
    y_te_a = np.asarray(y_te)
    under0 = round(float(np.mean(y_pred[y_te_a == 0] > 0)), 4)
    under0_adjacent = round(float(np.mean(y_pred[y_te_a == 0] == 1)), 4)
    under0_deep = round(float(np.mean(y_pred[y_te_a == 0] > 1)), 4)
    m01 = np.isin(y_te_a, [0, 1])
    under01 = round(float(np.mean(y_pred[m01] > np.maximum(y_te_a[m01], 1))), 4)
    cm = confusion_matrix(y_te, y_pred, labels=classes)
    report = classification_report(y_te, y_pred, labels=classes, output_dict=True, zero_division=0)
    metrics = {
        'log_loss': round(float(log_loss(y_te, p, labels=classes)), 4),
        'macro_f1': round(float(report['macro avg']['f1-score']), 4),
        'weighted_f1': round(float(report['weighted avg']['f1-score']), 4),
        'confusion_matrix': cm.tolist(),
        'per_class_recall': {str(c): round(float(report[str(c)]['recall']), 4) for c in classes},
        'per_class_precision': {str(c): round(float(report[str(c)]['precision']), 4) for c in classes},
        'under_triage_T0_any_down': under0,
        'under_triage_T0_to_T1_adjacent': under0_adjacent,
        'under_triage_T0_deep_to_T2plus': under0_deep,
        'under_triage_T0_T1_rate_beyond_adjacent': under01,
        'over_triage_rate': round(float(np.mean(y_pred[y_te_a >= 3] <= 1)), 4),
        'human_review_flag_rate': round(float(review.mean()), 4),
        'blend_weights': {m: round(w, 3) for (m, _), w in zip(models, weights)},
        'blend_val_score': score,
        'xgb_params': {k: (v.item() if hasattr(v, 'item') else v) for k, v in params.items()},
        'cohort': f'synthetic family-E {N_COHORT} (expert protocol labels, 5% inter-rater noise)',
    }
    with open(os.path.join(C.REPORTS, 'triage_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps({k: v for k, v in metrics.items() if k not in ('xgb_params',)}, indent=2))

    joblib.dump({'blend': models, 'km': km, 'gmm': gmm, 'scaler': scaler,
                 'features': feats, 'classes': classes,
                 'weights': list(weights), 'critical_rules': CRITICAL_RULES,
                 'cluster_report': creport},
                os.path.join(C.MODELS, 'triage_fusion.joblib'))
    print("TRIAGE FUSION DONE → models/triage_fusion.joblib", flush=True)


if __name__ == '__main__':
    main()
