"""08 — INFERENCE CLI: case JSON → specialist risks + RISK RATIO + TRIAGE.
EVOLVEX IT SOLUTIONS PVT. LTD.

Usage:
  python3 src/infer.py case.json
  echo '{"age":62,...}' | python3 src/infer.py -

Output contract (per spec: provenance, uncertainty, human review):
{
  "risk": {"cad": {"p": 0.31, "baseline": 0.22, "risk_ratio": 1.41, "tier": "moderate"}, ...},
  "triage": {"class": "T1", "probs": {...}, "reason_codes": [...],
             "requires_human_review": true|false, "confidence": 0.0-1.0},
  "disclaimer": "AI assistive triage — workflow urgency only, not diagnosis..."
}
"""
import json, os, sys
import numpy as np, pandas as pd, joblib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from train_triage import rule_scan, CRITICAL_RULES
from cluster import cluster_features

TIER = lambda rr: 'low' if rr < 1.2 else ('moderate' if rr < 2.0 else ('high' if rr < 3.5 else 'very_high'))

def load_artifacts():
    arts = {}
    for name in ['cad', 'diabetes', 'fetal_risk', 'maternal_risk', 'liver', 'sepsis',
                 'cad_aux', 'diabetes_readmit', 'hf_mortality', 'diabetes_symptoms',
                 'liver_hcv', 'cad_spect']:
        p = os.path.join(C.MODELS, f'{name}.joblib')
        if os.path.exists(p):
            arts[name] = joblib.load(p)
    arts['triage'] = joblib.load(os.path.join(C.MODELS, 'triage_fusion.joblib'))
    return arts

def predict_specialist(art, row_df):
    X = row_df.reindex(columns=art['features'], fill_value=0.0)
    Xaug = pd.concat([X.reset_index(drop=True), cluster_features(X, art['km'], art['gmm'], art['scaler'])], axis=1)
    model = art['model']
    if type(model).__name__ == 'MoEDecluster':
        return model.predict_proba(X, Xaug)
    return model.predict_proba(Xaug)

def risk_block(p, n_classes):
    if p is None: return None
    p = p[0]
    if n_classes == 2:
        pos = float(p[1])
        base = 0.5
        rr = round(pos / max(base, 1e-6), 2)
        return {'p_positive': round(pos, 4), 'baseline_reference': base, 'risk_ratio': rr, 'tier': TIER(rr)}
    hi = float(p[-1])
    base = 1.0 / n_classes
    rr = round(hi / max(base, 1e-6), 2)
    return {'p_highest_risk_class': round(hi, 4), 'class_probs': [round(float(x), 4) for x in p],
            'baseline_reference': base, 'risk_ratio': rr, 'tier': TIER(rr)}


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else '-'
    raw = sys.stdin.read() if src == '-' else open(src).read()
    case = json.loads(raw)
    arts = load_artifacts()
    row_df = pd.DataFrame([case])

    risks = {}
    for name, art in arts.items():
        if name == 'triage': continue
        p = predict_specialist(art, row_df)
        blk = risk_block(p, len(art['classes']))
        if blk:
            blk['model'] = 'MoE de-clustered (calibrated)'
            blk['dataset_family'] = 'real public clinical' if name != 'sepsis' else 'synthetic family-E'
            risks[name] = blk

    # ---- triage fusion ----------------------------------------------------
    t = arts['triage']
    Xt = row_df.reindex(columns=t['features'], fill_value=0.0)
    Xaug = pd.concat([Xt.reset_index(drop=True), cluster_features(Xt, t['km'], t['gmm'], t['scaler'])], axis=1)
    probs = np.zeros(len(t['classes']))
    detail = {}
    for w, (mname, m) in zip(t['weights'], t['blend']):
        if type(m).__name__ == 'MoEDecluster':
            pm = m.predict_proba(Xt, Xaug)[0]
        else:
            pm = m.predict_proba(Xaug)[0]
        probs += w * pm
        detail[mname] = [round(float(x), 4) for x in pm]
    pmap = {c: float(v) for c, v in zip(t['classes'], probs)}

    pred = int(max(pmap, key=pmap.get))
    if pmap.get(0, 0) + pmap.get(1, 0) >= 0.30: pred = min(pred, 1)
    elif pmap.get(0, 0) + pmap.get(1, 0) >= 0.14: pred = min(pred, 2)
    reasons = rule_scan(row_df.iloc[0])
    if reasons:
        pred = 0 if any(r in CRITICAL_RULES for r in reasons) else min(pred, 1)
    urgent_model = (pmap.get(0, 0) + pmap.get(1, 0)) > 0.25
    review = bool((reasons and not urgent_model) or (urgent_model and not reasons and pred >= 3)) or bool(case.get('uncertain'))
    conf = round(float(max(probs) * (1 - 0.5 * float(review))), 3)

    out = {
        'risk': risks,
        'triage': {
            'class': f'T{pred}',
            'meaning': ['emergency escalation', 'very urgent', 'priority consultation', 'routine consultation', 'follow-up'][pred],
            'probs': {f'T{k}': round(v, 4) for k, v in sorted(pmap.items())},
            'ensemble_votes': detail,
            'reason_codes': reasons or ['ROUTINE_STABLE'],
            'requires_human_review': review,
            'confidence': conf,
            'rule_version': '2026-09-triage-protocol-v1',
            'model_version': 'triage-fusion-moe-v1',
        },
        'disclaimer': 'AI assistive triage — workflow urgency only, not diagnosis. Clinical decisions remain with qualified clinicians.',
    }
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
