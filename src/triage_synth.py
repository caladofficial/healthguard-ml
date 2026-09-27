"""05 — SYNTHETIC EMERGENCY-DEPARTMENT TRIAGE COHORT (family E, expert rules).
EVOLVEX IT SOLUTIONS PVT. LTD. — round 2: 100k cases + engineered clinical features.

Why synthetic: public real ED-triage datasets (MIMIC-IV-ED, NHAMCS extracts)
need credentialed access; Kaggle auth was unavailable (username-only key).
Per spec §11 family E, synthetic data is legitimate when provenance-flagged —
and the labeling here is deterministic clinical-protocol logic (ESI/ATS-shaped),
NOT model output. Borderline noise (~5%) simulates inter-rater disagreement.

Round-2 feature engineering: shock_index, pulse_pressure, temp_deviation,
spo2_gap, qsofa/sirs composites, age risk flags — derived clinical scores give
the learners direct access to the same physiology the protocol encodes.
"""
import numpy as np, pandas as pd, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

SYMPTOMS = ['chest_pain','dyspnea','abd_pain','fever','cough','headache','weakness',
            'bleeding','vomiting','dizziness','syncope','seizure','altered_mental',
            'palpitations','jaundice','rash']
COMORB = ['htn','dm','cad_hf','copd','ckd','cancer','stroke_hx','immuno']
SYNDROMES = ['cardiac','resp','sepsis','gi','neuro','obstetric','minor']


def generate(n=100000, seed=42):
    rng = np.random.default_rng(seed)
    age = rng.integers(16, 95, n).astype(float)
    sex = rng.integers(0, 2, n)
    preg = ((sex == 1) & (rng.random(n) < 0.12) & (age < 45)).astype(int)
    syn = rng.choice(len(SYNDROMES), n, p=[0.18, 0.18, 0.10, 0.12, 0.10, 0.08, 0.24])
    sev = rng.gamma(2.0, 0.8, n) * (1 + 0.45 * (syn == 2) + 0.35 * (syn == 0) + 0.3 * (syn == 3))
    sev = np.clip(sev, 0, 10) / 10.0

    # vitals as functions of severity + syndrome
    hr = 72 + 55 * sev + rng.normal(0, 12, n) + 12 * (syn == 2) - 8 * (syn == 5)
    sbp = 122 - 38 * sev + rng.normal(0, 14, n) - 12 * (syn == 2)
    dbp = 76 - 18 * sev + rng.normal(0, 9, n)
    rr = 15 + 13 * sev + rng.normal(0, 3.5, n) + 5 * (syn == 1) + 4 * (syn == 2)
    temp = 36.6 + 1.6 * sev + rng.normal(0, 0.5, n) + 1.2 * (syn == 2) + 0.7 * (syn == 1)
    spo2 = 98 - 13 * sev + rng.normal(0, 1.6, n) - 4 * (syn == 1)
    gcs = 15 - np.round(5 * sev + rng.normal(0, 0.8, n) + 2 * (syn == 3)).clip(0, 12)
    pain = np.clip(2 + 6 * sev + rng.normal(0, 1.6, n) + 1.5 * ((syn == 0) | (syn == 2)), 0, 10).round()
    onset = rng.exponential(18, n).clip(0.2, 720)
    followup = rng.random(n) < 0.18

    # comorbidities
    com = {k: ((rng.random(n) < (0.05 + 0.22 * (age > 55) + 0.03 * sev)).astype(int)) for k in COMORB}
    com['htn'] = (rng.random(n) < 0.12 + 0.3 * (age > 50)).astype(int)
    com['dm'] = (rng.random(n) < 0.06 + 0.18 * (age > 50)).astype(int)

    # symptoms per syndrome
    sym = {k: np.zeros(n, dtype=int) for k in SYMPTOMS}
    def mark(name, mask, p):
        sym[name][mask] = (rng.random(mask.sum()) < p).astype(int)
    m = syn == 0; mark('chest_pain', m, .85); mark('dyspnea', m, .55); mark('palpitations', m, .4)
    m = syn == 1; mark('dyspnea', m, .85); mark('cough', m, .7); mark('fever', m, .35)
    m = syn == 2; mark('fever', m, .8); mark('altered_mental', m, .35); mark('weakness', m, .6); mark('dyspnea', m, .4); mark('vomiting', m, .3)
    m = syn == 3; mark('abd_pain', m, .85); mark('vomiting', m, .55); mark('jaundice', m, .15)
    m = syn == 4; mark('headache', m, .7); mark('dizziness', m, .5); mark('weakness', m, .5); mark('seizure', m, .12); mark('altered_mental', m, .15)
    m = syn == 5; mark('abd_pain', m, .5); mark('bleeding', m, .25); mark('dizziness', m, .3)
    m = syn == 6; mark('cough', m, .35); mark('headache', m, .3); mark('rash', m, .2); mark('fever', m, .15)
    sym['syncope'] = ((rng.random(n) < 0.05 + 0.2 * sev).astype(int))

    # risk heads (specialist calibrated risks) — correlated with latent state
    def sigmoid(x): return 1 / (1 + np.exp(-x))
    p_dm = sigmoid(-2.2 + 0.03 * (age - 45) + 2.2 * com['dm'] + 0.10 * sev + rng.normal(0, .4, n))
    p_cad = sigmoid(-2.6 + 0.035 * (age - 50) + 2.5 * com['cad_hf'] + 1.1 * com['htn'] + 0.9 * (syn == 0) + 0.25 * sev + rng.normal(0, .4, n))
    p_stroke = sigmoid(-3.4 + 0.03 * (age - 55) + 2.6 * com['stroke_hx'] + 1.0 * com['htn'] + 0.8 * (syn == 3) + 0.5 * sym['altered_mental'] + rng.normal(0, .4, n))
    p_sepsis = sigmoid(-3.0 + 2.2 * (syn == 2) + 0.9 * sev + 0.03 * (temp - 37) * 3 + 0.8 * sym['altered_mental'] + 0.5 * com['immuno'] + rng.normal(0, .4, n))
    p_liver = sigmoid(-2.8 + 2.0 * sym['jaundice'] + 0.015 * (age - 45) + rng.normal(0, .4, n))
    p_maternal = sigmoid(-1.2 + 2.4 * preg + 0.8 * (syn == 5) + 0.03 * (sbp - 120) * 0.4 + rng.normal(0, .5, n))

    # ---- deterministic protocol labeling (ESI/ATS-shaped) with reason codes
    y = np.full(n, 3)  # default T3
    reasons = [set() for _ in range(n)]
    def flag(mask, cls, code):
        idx = np.where(mask)[0]
        for i in idx:
            reasons[i].add(code)
        y[mask] = np.minimum(y[mask], cls)

    crit = (sbp < 90) | (sbp > 220) | (hr < 40) | (hr > 150) | (spo2 < 88) | (gcs <= 8)
    flag(crit & (sbp < 90), 0, 'SHOCK_HYPOTENSION')
    flag((sbp > 220) | ((sbp > 180) & (com['htn'] == 1) & (sym['headache'] == 1)), 0, 'HYPERTENSIVE_CRISIS')
    flag(hr < 40, 0, 'SEVERE_BRADYCARDIA')
    flag(hr > 150, 0, 'SEVERE_TACHYCARDIA')
    flag(spo2 < 88, 0, 'HYPOXEMIA')
    flag(gcs <= 8, 0, 'GCS_CRITICAL')
    flag(sym['seizure'].astype(bool) & (sev > 0.45), 0, 'ACTIVE_SEIZURE')
    flag((syn == 0) & (sym['chest_pain'] == 1) & (age > 45) & ((com['dm'] == 1) | (com['cad_hf'] == 1)) & (sev > 0.5), 0, 'ACS_RED_FLAG')
    flag((syn == 5) & (sym['bleeding'] == 1) & (sbp < 100), 0, 'HEMORRHAGE_OBSTETRIC')
    qsofa = ((sbp <= 100).astype(int) + (rr >= 22).astype(int) + (sym['altered_mental'].astype(bool)).astype(int))
    flag((syn == 2) & (qsofa >= 2), 0, 'SEPSIS_QSOFA')
    flag((onset < 3) & (sym['altered_mental'] == 1) & ((sym['weakness'] == 1) | (sym['dizziness'] == 1)) & (age > 50), 0, 'STROKE_FAST')

    maj = ((spo2 < 92) | (gcs <= 12) | (pain >= 8) | (temp > 39.5) | (qsofa == 1) |
           ((syn == 0) & (sym['chest_pain'] == 1) & (onset < 12)) | (p_sepsis > 0.55) |
           ((syn == 5) & preg & (sbp > 150)))
    flag(maj & (spo2 < 92), 1, 'HYPOXEMIA_MODERATE')
    flag(gcs <= 12, 1, 'ALTERED_GCS')
    flag(pain >= 8, 1, 'SEVERE_PAIN')
    flag(temp > 39.5, 1, 'HIGH_FEVER')
    flag(qsofa == 1, 1, 'SEPSIS_RISK')
    flag((syn == 0) & (sym['chest_pain'] == 1) & (onset < 12), 1, 'CHEST_PAIN_URGENT')
    flag((syn == 5) & preg & (sbp > 150), 1, 'PREECLAMPSIA_RISK')

    pri = ((pain >= 5) | (temp > 38.2) | (hr > 105) | (sbp > 160) | (rr > 22) |
           (sym['vomiting'] == 1) | (sym['palpitations'] == 1) | (p_cad > 0.45))
    flag(pri, 2, 'ABNORMAL_VITALS')
    flag((pain >= 5) & (pain < 8), 2, 'MODERATE_PAIN')

    fu = (followup & (y >= 3)).astype(bool)
    y[fu] = 4
    for i in np.where(fu)[0]:
        reasons[i].add('FOLLOW_UP')

    # inter-rater noise (borderline ±1 class) + uncertainty labels
    uncertain = np.zeros(n, dtype=int)
    borderline = (np.random.default_rng(seed + 1).random(n) < 0.05)
    y[borderline] = np.clip(y[borderline] + np.random.default_rng(seed + 2).choice([-1, 1], borderline.sum()), 0, 4)
    uncertain[borderline] = 1

    # ---- round-2 engineered clinical scores -------------------------------
    shock_index = hr / sbp
    pulse_pressure = sbp - dbp
    temp_deviation = np.abs(temp - 36.8)
    spo2_gap = np.clip(100 - spo2, 0, None)
    qsofa_feat = ((sbp <= 100).astype(int) + (rr >= 22).astype(int) + (sym['altered_mental']).astype(int))
    sirs_feat = ((hr > 90).astype(int) + (rr > 20).astype(int) + ((temp > 38) | (temp < 36)).astype(int))
    news_like = (np.clip((sbp - 110) / 20, -2, 2).round() * -1 + np.clip((hr - 85) / 25, -2, 2).round() +
                 np.clip((rr - 17) / 5, -2, 2).round() + np.clip((spo2 - 96) / 3, -2, 2).round() * -1 +
                 np.clip((temp - 37) / 1.2, -2, 2).round())
    age_risk = ((age > 65).astype(int) + (age > 80).astype(int))

    X = pd.DataFrame(dict(
        age=age, sex=sex, pregnancy=preg, hr=hr.round(1), sbp=sbp.round(1), dbp=dbp.round(1),
        rr=rr.round(1), temp_c=temp.round(2), spo2=spo2.round(1), gcs=gcs.astype(int),
        pain=pain.astype(int), onset_hours=onset.round(1), followup_flag=followup.astype(int),
        shock_index=shock_index.round(4), pulse_pressure=pulse_pressure.round(1),
        temp_deviation=temp_deviation.round(2), spo2_gap=spo2_gap.round(1),
        qsofa_feat=qsofa_feat.astype(int), sirs_feat=sirs_feat.astype(int),
        news_like=news_like.round(1), age_risk=age_risk.astype(int),
        p_dm=np.round(p_dm, 4), p_cad=np.round(p_cad, 4), p_stroke=np.round(p_stroke, 4),
        p_sepsis=np.round(p_sepsis, 4), p_liver=np.round(p_liver, 4), p_maternal=np.round(p_maternal, 4),
    ))
    for k, v in list(sym.items()) + list(com.items()):
        X[k] = v
    meta = pd.DataFrame({
        'reason_codes': [';'.join(sorted(r)) if r else 'NONE' for r in reasons],
        'syndrome': [SYNDROMES[s] for s in syn],
        'uncertain': uncertain,
    })
    return X, y.astype(int), meta


if __name__ == '__main__':
    X, y, meta = generate()
    out = pd.concat([X, meta, pd.Series(y, name='triage')], axis=1)
    out.to_csv(os.path.join(C.PROC, 'triage_cohort.csv'), index=False)
    print('triage cohort', out.shape, '| class counts', np.bincount(y).tolist())
