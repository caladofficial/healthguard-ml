"""06 — SPECIALIST RISK MODEL TRAINING (de-clustered high-artillery MoE).
EVOLVEX IT SOLUTIONS PVT. LTD. — round 2: size-aware ensembles (memory-safe),
skip-if-trained resume, 12 heads.

Per condition: KMeans/GMM phenotypes → cluster-augmented features →
XGB-search + grouping candidates + calibrated probabilities →
MoE de-clustered experts. Artifacts + metrics per task.
"""
import json, os, sys, time
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
from sklearn.preprocessing import label_binarize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from datasets import load_all
from cluster import phenotype_report, cluster_features
from learners import stack, soft_vote, calibrated, fast_search_xgb, MoEDecluster, base_learners

BIG_N = 15000   # above this: light candidates only (memory guard)


def eval_binary(p, y):
    p1 = p[:, 1] if p.ndim > 1 else p
    return {
        'roc_auc': round(float(roc_auc_score(y, p1)), 4),
        'pr_auc': round(float(average_precision_score(y, p1)), 4),
        'brier': round(float(brier_score_loss(y, p1)), 4),
        'log_loss': round(float(log_loss(y, p, labels=[0, 1])), 4),
    }

def eval_multi(p, y, k):
    yb = label_binarize(y, classes=list(range(k)))
    return {
        'roc_auc_ovr': round(float(roc_auc_score(yb, p, average='macro', multi_class='ovr')), 4),
        'log_loss': round(float(log_loss(y, p, labels=list(range(k)))), 4),
    }


def train_one(name, X, y, task):
    t0 = time.time()
    n_classes = len(np.unique(y))
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=C.SEED)

    km, gmm, scaler, creport = phenotype_report(X_tr, y_tr)
    Xtr_aug = pd.concat([X_tr.reset_index(drop=True), cluster_features(X_tr, km, gmm, scaler)], axis=1)
    Xte_aug = pd.concat([X_te.reset_index(drop=True), cluster_features(X_te, km, gmm, scaler)], axis=1)

    # ---- candidate comparison (grouping; size-aware) ----------------------
    if len(X_tr) >= BIG_N:
        cands = {}
    else:
        cands = {
            'stack_group': stack(task, n_classes),
            'soft_vote_group': soft_vote(task, n_classes),
        }
    best_name, best_model, best_score = None, None, -np.inf
    results = {}
    for cname, model in cands.items():
        m = calibrated(model, cv=3, n=len(y_tr))
        m.fit(Xtr_aug, y_tr)
        p = m.predict_proba(Xte_aug)
        sc = eval_binary(p, y_te)['roc_auc'] if n_classes == 2 else eval_multi(p, y_te, n_classes)['roc_auc_ovr']
        results[cname] = sc
        if sc > best_score:
            best_name, best_model, best_score = cname, m, sc

    tuned, params = fast_search_xgb(Xtr_aug, y_tr, n_classes)
    tm = calibrated(tuned, cv=3, n=len(y_tr))
    tm.fit(Xtr_aug, y_tr)
    p = tm.predict_proba(Xte_aug)
    sc = eval_binary(p, y_te)['roc_auc'] if n_classes == 2 else eval_multi(p, y_te, n_classes)['roc_auc_ovr']
    results['xgb_tuned'] = sc
    if sc > best_score:
        best_name, best_model, best_score = 'xgb_tuned', tm, sc

    # ---- de-clustered MoE over the winning group --------------------------
    moe = MoEDecluster(km, gmm, scaler, task, n_classes).fit(X_tr, y_tr, Xtr_aug)
    pm = moe.predict_proba(X_te, Xte_aug)
    sc = eval_binary(pm, y_te)['roc_auc'] if n_classes == 2 else eval_multi(pm, y_te, n_classes)['roc_auc_ovr']
    results['moe_declustered'] = sc

    final_model, final_tag = (moe, 'moe_declustered') if sc >= best_score else (best_model, best_name)
    pfinal = pm if final_tag == 'moe_declustered' else best_model.predict_proba(Xte_aug)
    metrics = eval_binary(pfinal, y_te) if n_classes == 2 else eval_multi(pfinal, y_te, n_classes)
    metrics.update({
        'model': final_tag, 'holdout': f'{len(y_te)} rows', 'n_classes': n_classes,
        'candidate_scores': results, 'xgb_search_params': {k: (v if not isinstance(v, np.generic) else v.item()) for k, v in params.items()},
        'baseline_prevalence': round(float(np.mean(y)), 4),
        'train_seconds': round(time.time() - t0, 1),
    })

    joblib.dump({'model': final_model, 'km': km, 'gmm': gmm, 'scaler': scaler,
                 'features': list(X.columns), 'task': task,
                 'classes': sorted(np.unique(y).tolist()),
                 'baseline_prevalence': float(np.mean(np.asarray(y) == np.bincount(y.astype(int)).argmax())),
                 'positive_prevalence': float(np.mean(y)) if n_classes == 2 else None,
                 'cluster_report': creport},
                os.path.join(C.MODELS, f'{name}.joblib'))
    with open(os.path.join(C.REPORTS, f'{name}_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    keep = {k: v for k, v in metrics.items() if k in ('roc_auc', 'roc_auc_ovr', 'pr_auc', 'brier')}
    print(f"✓ {name:18s} model={final_tag:18s} metrics={keep} ({metrics['train_seconds']}s)", flush=True)
    return metrics


def main():
    data = load_all()
    allm = {}
    for name, (X, y, task) in data.items():
        mf = os.path.join(C.MODELS, f'{name}.joblib')
        rf = os.path.join(C.REPORTS, f'{name}_metrics.json')
        if os.path.exists(mf) and os.path.exists(rf):
            print(f"· {name:18s} already trained — skipping (resume)", flush=True)
            allm[name] = json.load(open(rf))
            continue
        try:
            allm[name] = train_one(name, X, y, task)
        except Exception as e:
            print(f"✗ {name} failed: {type(e).__name__}: {e}", flush=True)
    with open(os.path.join(C.REPORTS, 'specialists_metrics.json'), 'w') as f:
        json.dump(allm, f, indent=2)
    print("\nSPECIALISTS DONE → models/*.joblib + reports/specialists_metrics.json", flush=True)


if __name__ == '__main__':
    main()
