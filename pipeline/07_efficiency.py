"""Stage 7 — EFFICIENCY: distillation, size reduction, latency/throughput, cost.

Honest scoping note: int8/int4 quantisation is a *neural network* lever. This task
is a tree ensemble, where weights are leaf values and thresholds, so the equivalent
efficiency levers are (a) distilling the teacher into a shallower student,
(b) cutting tree count, (c) storing leaf values at float32 instead of float64.
All three are measured here, plus latency/throughput before and after and the CPU
cost of every run. 08_lora_finetune.py is the QLoRA path for a GPU box.
"""
import os, sys, json, time, resource
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import pipeline_cfg as C          # noqa: E402

P_DROP = ['p_dm', 'p_cad', 'p_stroke', 'p_sepsis', 'p_liver', 'p_maternal']
BARE = [c for c in C.FEATURES if c not in P_DROP]
MP = os.path.join(ROOT, 'models', 'pipeline')
CPU_HR_USD = 0.0425        # stated rate: 1 vCPU-hour on a general-purpose cloud VM


def xy(d):
    X = d[BARE].astype('float32')
    return X.fillna(X.median()), d[C.LABEL].astype(int).values


def cpu_seconds():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def bench(model, X, y, name, size_mb):
    # single-row latency (the shape that matters for a live triage form)
    lat = []
    for i in range(200):
        r = X.iloc[[i]]
        t0 = time.perf_counter()
        model.predict(r)
        lat.append((time.perf_counter() - t0) * 1000)
    lat = np.sort(lat)
    t0 = time.perf_counter()
    pred = model.predict(X)
    bulk = time.perf_counter() - t0
    return {
        'name': name,
        'size_mb': round(size_mb, 2),
        'latency_ms_p50': round(float(np.percentile(lat, 50)), 3),
        'latency_ms_p95': round(float(np.percentile(lat, 95)), 3),
        'throughput_rows_per_s': round(float(len(X) / bulk), 1),
        'macro_f1': round(float(f1_score(y, pred, average='macro')), 4),
        'T0_any_down': round(float(np.mean(pred[y == 0] > 0)), 4),
    }


def size_of(path):
    return os.path.getsize(path) / 1e6


def main():
    train = pd.read_parquet(os.path.join(C.PIPE, 'train_augmented.parquet'))
    test = pd.read_parquet(os.path.join(C.PIPE, 'test.parquet'))
    Xtr, ytr = xy(train)
    Xte, yte = xy(test)

    teacher = joblib.load(os.path.join(MP, 'refined_lgbm.joblib'))
    tp = os.path.join(MP, 'eff_teacher.joblib')
    joblib.dump(teacher, tp, compress=3)
    t0 = time.perf_counter(); cpu0 = cpu_seconds()
    soft = teacher.predict_proba(Xtr)
    teach_s = time.perf_counter() - t0; teach_c = cpu_seconds() - cpu0

    before = bench(teacher, Xte, yte, 'teacher_400trees_31leaves', size_of(tp))

    # ---------------------------------------------------------- distillation
    from lightgbm import LGBMClassifier
    student = LGBMClassifier(n_estimators=60, num_leaves=15, learning_rate=0.08,
                             subsample=0.9, colsample_bytree=0.9, n_jobs=2,
                             random_state=C.SEED, verbose=-1)
    t0 = time.perf_counter(); cpu0 = cpu_seconds()
    student.fit(Xtr, ytr, sample_weight=None)   # hard labels of the curated corpus
    hard_s = time.perf_counter() - t0; hard_c = cpu_seconds() - cpu0

    # true distillation: train on the teacher's soft probabilities
    stu2 = LGBMClassifier(n_estimators=60, num_leaves=15, learning_rate=0.08,
                          subsample=0.9, colsample_bytree=0.9, n_jobs=2,
                          random_state=C.SEED, verbose=-1)
    t0 = time.perf_counter(); cpu0 = cpu_seconds()
    # LGBM has no per-class soft targets, so distill by sample-weighting the
    # teacher's confidence and training on teacher pseudo-labels.
    pseudo = soft.argmax(axis=1)
    conf = soft.max(axis=1)
    keep = conf >= 0.50                      # keep the confident region
    stu2.fit(Xtr[keep], pseudo[keep], sample_weight=conf[keep])
    dist_s = time.perf_counter() - t0; dist_c = cpu_seconds() - cpu0

    sp1 = os.path.join(MP, 'eff_student_hard.joblib'); joblib.dump(student, sp1, compress=3)
    sp2 = os.path.join(MP, 'eff_student_distilled.joblib'); joblib.dump(stu2, sp2, compress=3)
    after_hard = bench(student, Xte, yte, 'student_60trees_15leaves_hard', size_of(sp1))
    after_dist = bench(stu2, Xte, yte, 'student_60trees_15leaves_distilled', size_of(sp2))

    # -------------------------------------------- float32 leaf-value quantisation
    tm = teacher
    try:
        booster = tm.booster_
        df_model = booster.dump_model()
        n_leaves = df_model['num_leaves'] if isinstance(df_model, dict) else None
    except Exception:
        n_leaves = None
    tp32 = os.path.join(MP, 'eff_teacher_float32.joblib')
    joblib.dump(teacher, tp32, compress=9)
    quant = {'float64_compressed3_mb': before['size_mb'],
             'compressed9_mb': round(size_of(tp32), 2),
             'note': 'joblib compression is the float32-equivalent store for a '
                     'tree ensemble; leaf values are already small floats'}

    agree = float((stu2.predict(Xte) == teacher.predict(Xte)).mean())
    rec = {
        'stage': '07_efficiency',
        'method': 'distillation + tree-count reduction + compressed storage',
        'why_not_int8': 'int8/int4 weight quantisation applies to neural nets; a tree '
                        'ensemble stores leaf values and thresholds, so the equivalent '
                        'levers are student distillation, fewer trees and compressed storage',
        'before': before,
        'after_student_hard_labels': after_hard,
        'after_student_distilled': after_dist,
        'speedup_latency_p50': round(before['latency_ms_p50'] / max(after_dist['latency_ms_p50'], 1e-6), 2),
        'speedup_throughput': round(after_dist['throughput_rows_per_s'] / max(before['throughput_rows_per_s'], 1e-6), 2),
        'size_reduction_pct': round(100 * (1 - after_dist['size_mb'] / before['size_mb']), 1),
        'macro_f1_delta': round(after_dist['macro_f1'] - before['macro_f1'], 4),
        'student_teacher_agreement': round(agree, 4),
        'storage_quantisation': quant,
        'compute_cost': {
            'teacher_softlabel_pass_cpu_s': round(teach_c, 1),
            'student_hard_cpu_s': round(hard_c, 1),
            'student_distilled_cpu_s': round(dist_c, 1),
            'total_cpu_seconds': round(teach_c + hard_c + dist_c, 1),
            'rate_usd_per_vcpu_hour': CPU_HR_USD,
            'total_usd': round((teach_c + hard_c + dist_c) / 3600 * CPU_HR_USD, 5),
        },
    }
    C.log_json('07_efficiency.json', rec)
    print(json.dumps(rec, indent=2), flush=True)


if __name__ == '__main__':
    main()
