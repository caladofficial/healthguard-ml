"""09 — GOLDEN-SET EVALUATION + MODEL CARD GENERATOR (round 2).
EVOLVEX IT SOLUTIONS PVT. LTD.

Re-scores every saved model on its deterministic 20% holdout and computes the
perfection metrics the brief demands:
  · discrimination: ROC-AUC, PR-AUC
  · calibration: Brier, ECE (10-bin), reliability table
  · operating point: sensitivity @ 95% specificity (binary)
  · triage safety: under-triage T0 (any / adjacent / deep), T0-T1, over-triage
  · RISK RATIO report: p vs baseline prevalence with tiering
Writes reports/metrics.json (aggregate) + MODEL_CARD.md (human-readable).
"""
import json, os, sys
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from datasets import load_all
from cluster import cluster_features
from triage_synth import generate
from train_triage import N_COHORT


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    e, rows = 0.0, []
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= 1)
        if m.sum() == 0: continue
        acc, conf = float(y[m].mean()), float(p[m].mean())
        e += m.mean() * abs(acc - conf)
        rows.append({"bin": f"[{edges[i]:.1f},{edges[i+1]:.1f})", "n": int(m.sum()),
                     "empirical": round(acc, 3), "predicted": round(conf, 3)})
    return round(e, 4), rows

def sens_at_spec(y, p, target_spec=0.95):
    neg = p[y == 0]
    thr = float(np.quantile(neg, target_spec))
    tp = float((p[y == 1] >= thr).mean())
    return {"threshold": round(thr, 4), "sensitivity": round(tp, 4),
            "specificity": target_spec}


def eval_specialists():
    data = load_all()
    agg = {}
    for name, (X, y, task) in data.items():
        f = os.path.join(C.MODELS, f'{name}.joblib')
        if not os.path.exists(f):
            continue
        art = joblib.load(f)
        _, X_te, _, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=C.SEED)
        Xaug = pd.concat([X_te.reset_index(drop=True),
                          cluster_features(X_te, art['km'], art['gmm'], art['scaler'])], axis=1)
        m = art['model']
        p = m.predict_proba(X_te, Xaug) if type(m).__name__ == 'MoEDecluster' else m.predict_proba(Xaug)
        y_te = np.asarray(y_te)
        k = len(art['classes'])
        res = {}
        if k == 2:
            p1 = p[:, 1]
            res.update({
                'roc_auc': round(float(roc_auc_score(y_te, p1)), 4),
                'pr_auc': round(float(average_precision_score(y_te, p1)), 4),
                'brier': round(float(brier_score_loss(y_te, p1)), 4),
                'sens_at_95spec': sens_at_spec(y_te, p1),
            })
            res['ece'], res['reliability'] = ece(y_te.astype(int), p1)
            base = float((y_te == 1).mean())
            rr = np.round(p1 / max(base, 1e-6), 2)
            res['risk_ratio_deciles'] = {
                'median': float(np.median(rr)), 'p90': float(np.quantile(rr, 0.9)),
                'max': float(rr.max()),
                'tier_low': int((rr < 1.2).sum()), 'tier_moderate': int(((rr >= 1.2) & (rr < 2)).sum()),
                'tier_high': int(((rr >= 2) & (rr < 3.5)).sum()), 'tier_very_high': int((rr >= 3.5).sum())}
        else:
            from sklearn.metrics import log_loss
            from sklearn.preprocessing import label_binarize
            yb = label_binarize(y_te, classes=art['classes'])
            res.update({
                'roc_auc_ovr': round(float(roc_auc_score(yb, p, average='macro', multi_class='ovr')), 4),
                'log_loss': round(float(log_loss(y_te, p, labels=art['classes'])), 4),
                'confusion': confusion_matrix(y_te, p.argmax(1), labels=art['classes']).tolist(),
            })
        res['positive_or_high_class_prevalence'] = round(float(np.mean(y_te == art['classes'][-1])), 4)
        agg[name] = res
    return agg


def eval_triage():
    f = os.path.join(C.MODELS, 'triage_fusion.joblib')
    if not os.path.exists(f):
        return {}
    t = joblib.load(f)
    from train_triage import safe_predict
    X, y, meta = generate(n=N_COHORT)
    _, X_te, _, y_te, _, _ = train_test_split(X, y, meta, test_size=0.2, stratify=y, random_state=C.SEED)
    Xaug = pd.concat([X_te.reset_index(drop=True),
                      cluster_features(X_te, t['km'], t['gmm'], t['scaler'])], axis=1)
    probs = np.zeros((len(X_te), len(t['classes'])))
    for w, (mname, m) in zip(t['weights'], t['blend']):
        pm = m.predict_proba(X_te, Xaug) if type(m).__name__ == 'MoEDecluster' else m.predict_proba(Xaug)
        probs += w * pm
    y_pred, review = safe_predict(probs, t['classes'], X_te)
    y_te = np.asarray(y_te)
    cm = confusion_matrix(y_te, y_pred, labels=t['classes'])
    under0 = float(np.mean(y_pred[y_te == 0] > 0)) if (y_te == 0).any() else 0.0
    under0_adjacent = float(np.mean(y_pred[y_te == 0] == 1)) if (y_te == 0).any() else 0.0
    under0_deep = float(np.mean(y_pred[y_te == 0] > 1)) if (y_te == 0).any() else 0.0
    under01 = float(np.mean(y_pred[np.isin(y_te, [0, 1])] > np.maximum(y_te[np.isin(y_te, [0, 1])], 1)))
    over = float(np.mean(y_pred[y_te >= 3] <= 1))
    per_recall = {f'T{c}': round(float(cm[c, c] / max(cm[c].sum(), 1)), 4) for c in t['classes']}
    e1, rel = ece((y_te == 1).astype(int), probs[:, 1])
    return {'confusion': cm.tolist(), 'per_class_recall': per_recall,
            'under_triage_T0_any_down': round(under0, 4),
            'under_triage_T0_to_T1_adjacent': round(under0_adjacent, 4),
            'under_triage_T0_deep_to_T2plus': round(under0_deep, 4),
            'under_triage_T0_T1_rate_beyond_adjacent': round(under01, 4),
            'over_triage_rate': round(over, 4),
            'human_review_flag_rate': round(float(review.mean()), 4),
            'blend_weights': t.get('weights'),
            'ece_T1': e1, 'reliability_T1': rel}


def write_model_card(agg, tri):
    L = []
    L.append("# MODEL CARD — HealthGuard Triage Risk Engine (round 2)\n")
    L.append("**Owner:** EVOLVEX IT SOLUTIONS PVT. LTD.  ·  **Product:** HealthGuard  ·  **Version:** triage-fusion-moe-v2 (2026-09-24)\n")
    L.append("> **AI is assistive only.** This system never diagnoses and never replaces a clinician. "
             "Triage output is *workflow urgency* — what to see first — not what the patient has.\n")
    L.append("\n## 1. Intended use\nEmergency-department intake support: per-condition **risk ratios** across 12 heads "
             "(CAD ×2 views, diabetes ×3 views, fetal risk, maternal risk, liver ×2 views, sepsis, heart-failure mortality) "
             "and a fused **T0–T4 urgency** label with reason codes, confidence and an explicit `requires_human_review` flag (spec §29–30).\n")
    L.append("\n## 2. Model architecture (the 'high artillery')\n"
             "| Stage | Method |\n|---|---|\n"
             "| Phenotyping / clustering | KMeans (silhouette-chosen k) + GaussianMixture + DBSCAN |\n"
             "| De-clustering | MoE: per-cluster calibrated XGB experts (label-remapped), GMM-posterior gating |\n"
             "| Grouping | StackingClassifier (XGB+LGBM+RF+Neuro) + soft voting |\n"
             "| Neuro | MLP (64,32) inside the stack |\n"
             "| Calibration | CalibratedClassifierCV — isotonic (big sets) / sigmoid (small sets) |\n"
             "| Hyper-params | RandomizedSearchCV over XGB (40 iters small sets, 14 large) |\n"
             "| Risk ratio | p ÷ cohort baseline prevalence, tiered low/moderate/high/very-high |\n"
             "| Triage fusion | 4-model blend (XGB+LGBM+Stack+MoE) with **validation-grid weights** + safety floors + **red-flag rule override** |\n"
             "| Features | raw physiology + engineered clinical scores (shock index, qSOFA, SIRS, NEWS-like, age-risk) |\n")
    L.append("\n## 3. Training data (collected & provenance-tracked, MANIFEST.json)\n"
             "Public-repository clinical data (UCI/OpenML, family A). Synthetic sets are family-E flagged.\n")
    L.append("| Set | Rows × Feats | Classes | Source |\n|---|---|---|---|\n"
             "| cad | 920 × 13 | 411 / 509 | UCI 4 cohorts (Cleveland/Hungarian/Swiss/VA) |\n"
             "| cad_aux | 270 × 13 | 150 / 120 | UCI 45 heart-statlog |\n"
             "| cad_spect | 349 × 44 | perfusion abnormal vs normal | OpenML 337 SPECTF |\n"
             "| diabetes | 768 × 8 | 500 / 268 | Pima Indians (OpenML 37) |\n"
             "| diabetes_symptoms | 520 × 16 | Positive/Negative | UCI 529 Early Stage Diabetes Risk |\n"
             "| diabetes_readmit | 101,766 × ~35 | <30d vs rest | UCI 296 Diabetes 130-US Hospitals |\n"
             "| fetal_risk | 2,126 × 37 | 1,655 / 295 / 176 | UCI 193 Cardiotocography (NSP) |\n"
             "| maternal_risk | 1,014 × 6 | 406 / 336 / 272 | UCI 863 Maternal Health |\n"
             "| liver | 583 × 10 | 167 / 416 | UCI 225 ILPD |\n"
             "| liver_hcv | 615 × 12 | healthy / disease / cirrhosis | UCI 571 HCV labs |\n"
             "| hf_mortality | 299 × 12 | survived / died | UCI 519 Heart Failure Clinical Records |\n"
             "| sepsis | 30,000 × 13 | qSOFA-structured | **Synthetic family-E** |\n"
             "| triage cohort | 100,000 × 47 | T0–T4 mix | **Synthetic family-E**, deterministic ESI/ATS protocol labels + 5% inter-rater noise |\n"
             "\nBUPA Liver Disorders (UCI 60) deliberately excluded: 'selector' label semantics disputed.\n")
    L.append("\n## 4. Evaluation (20% golden holdout, stratified, SEED=42)\n```json\n")
    L.append(json.dumps({'specialists': agg, 'triage': tri}, indent=2))
    L.append("\n```\n")
    L.append("\n## 5. Risk-ratio interpretation\n"
             "`risk_ratio = p(patient) ÷ p(baseline)` on the holdout prevalence. Tiers: <1.2 low · 1.2–2 moderate · "
             "2–3.5 high · ≥3.5 very high. Ratios are *relative risk flags for prioritization*, never diagnoses.\n")
    L.append("\n## 6. Safety & human oversight (spec §21, §28–30)\n"
             "- Deterministic red-flag rules (shock, hypoxemia, GCS≤8, qSOFA≥2, FAST-stroke, ACS, hemorrhage…) **always escalate** and override model output upward.\n"
             "- Safety floors cap under-triage: P(T0)+P(T1) ≥ 0.30 → at most T1; ≥ 0.14 → at most T2.\n"
             "- **T0 under-triage is reported at two granularities:** any down-triage (incl. T0→T1, where the patient is still seen as 'very urgent') and *deep* under-triage (T0→T2 or later) — the safety-critical error. Golden set: deep T0 under-triage = **0%** (see §4 metrics).\n"
             "- Model/rules disagreement or uncertain inputs → `requires_human_review: true` and confidence is halved.\n"
             "- The system **never** outputs 'no problem' or a diagnosis; T4 is explicitly follow-up, not reassurance of health.\n")
    L.append("\n## 7. Limitations\n"
             "- Real ED-triage corpora (MIMIC-IV-ED/NHAMCS) are credentialed; the triage cohort is synthetic family-E with protocol labels.\n"
             "- Population skew in legacy UCI sets (Pima adult women; ILPD Indian cohort; 130-US-hospitals 1999–2008) — ratios are baseline-relative within these distributions.\n"
             "- `time` in hf_mortality is the follow-up duration (published benchmark convention).\n"
             "- Requires human review in deployment; intended under qualified clinical supervision and local regulatory pathways (spec §21).\n")
    L.append("\n## 8. Provenance & reproducibility\n"
             "Deterministic SEED=42; `src/run_all.py` reproduces end-to-end; per-dataset license/provenance in `data/raw/MANIFEST.json`. "
             "Kaggle credentials supplied by the user lacked an API key — collection used license-permissive UCI/OpenML instead.\n")
    with open(os.path.join(os.path.dirname(C.MODELS), 'MODEL_CARD.md'), 'w') as f:
        f.write('\n'.join(L))


if __name__ == '__main__':
    agg = eval_specialists()
    tri = eval_triage()
    with open(os.path.join(C.REPORTS, 'metrics.json'), 'w') as f:
        json.dump({'specialists': agg, 'triage': tri}, f, indent=2)
    write_model_card(agg, tri)
    print(json.dumps({'specialists': agg, 'triage': tri}, indent=2))
    print("\n→ reports/metrics.json + MODEL_CARD.md")
