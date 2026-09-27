"""04 — HIGH-ARTILLERY LEARNERS (grouped + stacked + calibrated).
EVOLVEX IT SOLUTIONS PVT. LTD. — round 2: size-aware calibration, scaled search.

Arsenal per the brief:
  · XGBoost + LightGBM (gradient boosting)
  · Random Forest (bagged sklearn grouping)
  · MLP 'neuro' (sklearn neural net)
  · Logistic regression (calibration anchor)
  · StackingClassifier ('skill-like grouping' of the above)
  · CalibratedClassifierCV → trustworthy probabilities for RISK RATIO
    (sigmoid for small sets — isotonic overfits below ~2.5k rows)
  · De-clustered MoE: per-cluster experts blended by cluster posterior
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import config as C


def calib_method(n):
    return 'isotonic' if n >= 2500 else 'sigmoid'


def calibrated(model, cv=3, n=None, method=None):
    if method is None:
        method = calib_method(n if n is not None else 10**9)
    return CalibratedClassifierCV(model, method=method, cv=cv)


def base_learners(task, n_classes=2):
    xgb = XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.9,
        colsample_bytree=0.9, min_child_weight=2, reg_lambda=1.5,
        objective='binary:logistic' if n_classes == 2 else 'multi:softprob',
        num_class=n_classes if n_classes > 2 else None,
        tree_method='hist', random_state=C.SEED, n_jobs=2, verbosity=0)
    lgbm = LGBMClassifier(
        n_estimators=400, num_leaves=31, learning_rate=0.05, subsample=0.9,
        colsample_bytree=0.9, min_child_samples=10, reg_lambda=1.2,
        random_state=C.SEED, n_jobs=2, verbose=-1)
    rf = RandomForestClassifier(
        n_estimators=450, max_depth=9, min_samples_leaf=3,
        class_weight='balanced_subsample', random_state=C.SEED, n_jobs=2)
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', alpha=1e-3,
                        batch_size=64, learning_rate_init=1e-3, max_iter=450,
                        early_stopping=True, random_state=C.SEED)
    lr = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=C.SEED)
    return {'xgb': xgb, 'lgbm': lgbm, 'rf': rf, 'neuro': mlp, 'lr': lr}


def stack(task, n_classes=2):
    """'Skill-like grouping': heterogeneous stack with logistic meta-learner."""
    b = base_learners(task, n_classes)
    ests = [(k, b[k]) for k in ('xgb', 'lgbm', 'rf', 'neuro')]
    return StackingClassifier(
        estimators=ests, final_estimator=LogisticRegression(max_iter=2000, class_weight='balanced'),
        stack_method='predict_proba', cv=3, n_jobs=1)


def soft_vote(task, n_classes=2):
    b = base_learners(task, n_classes)
    return VotingClassifier(
        estimators=[(k, b[k]) for k in ('xgb', 'lgbm', 'rf')], voting='soft', n_jobs=1)


def fast_search_xgb(X, y, n_classes=2):
    """Randomized search — search budget scales with dataset size."""
    n = len(X)
    n_iter = 40 if n < 20000 else 14
    cv = StratifiedKFold(n_splits=5 if n >= 3000 else 4, shuffle=True, random_state=C.SEED)
    base = XGBClassifier(
        objective='binary:logistic' if n_classes == 2 else 'multi:softprob',
        num_class=n_classes if n_classes > 2 else None,
        tree_method='hist', random_state=C.SEED, n_jobs=2, verbosity=0)
    dist = dict(n_estimators=[300, 500, 700, 900], max_depth=[3, 5, 7, 9],
                learning_rate=[0.02, 0.04, 0.06, 0.08], subsample=[0.8, 0.9, 1.0],
                colsample_bytree=[0.7, 0.85, 1.0], min_child_weight=[1, 2, 5],
                reg_lambda=[1.0, 1.5, 3.0])
    rs = RandomizedSearchCV(base, dist, n_iter=n_iter,
                            scoring='roc_auc' if n_classes == 2 else 'neg_log_loss',
                            cv=cv, random_state=C.SEED, n_jobs=2, refit=True)
    rs.fit(X, y)
    return rs.best_estimator_, rs.best_params_


class MoEDecluster:
    """De-clustering mixture-of-experts: one specialist per KMeans phenotype,
    gated by GMM posterior (soft). Falls back to the global model for tiny
    clusters. This is the 'de clustering' requested: the mix is decomposed into
    cluster-specialized experts and re-composed by gating."""

    def __init__(self, km, gmm, scaler, task, n_classes=2):
        self.km, self.gmm, self.scaler = km, gmm, scaler
        self.task, self.n_classes = task, n_classes
        self.experts = {}
        self.global_model = None

    def fit(self, X, y, X_aug):
        from cluster import cluster_indices
        if len(y) >= 15000:   # memory-safe global anchor on large sets
            self.global_model = calibrated(base_learners(self.task, self.n_classes)['xgb'], n=len(y))
        else:
            self.global_model = calibrated(stack(self.task, self.n_classes), n=len(y))
        self.global_model.fit(X_aug, y)
        idx_map = cluster_indices(X, self.km, self.scaler)
        y = np.asarray(y)
        for c, idx in idx_map.items():
            if len(idx) < max(60, 5 * self.n_classes):
                continue
            ys = y[idx]
            classes = np.asarray(np.unique(ys), dtype=int)
            remap = {int(v): i for i, v in enumerate(classes)}
            yr = np.array([remap[int(v)] for v in ys], dtype=int)
            n_local = len(classes)
            if n_local == 1:   # pure phenotype — constant expert, no model needed
                self.experts[int(c)] = ('constant', classes)
                continue
            xgb = XGBClassifier(
                n_estimators=350, max_depth=5, learning_rate=0.05,
                objective='binary:logistic' if n_local == 2 else 'multi:softprob',
                num_class=n_local if n_local > 2 else None,
                tree_method='hist', random_state=C.SEED, n_jobs=2, verbosity=0)
            minc = int(np.bincount(yr).min())
            m = None
            if minc >= 3:
                try:
                    m = calibrated(xgb, cv=3, n=len(idx))
                    m.fit(X_aug.iloc[idx], yr)
                except ValueError:
                    m = None
            if m is None:  # tiny/absent class in this cluster → uncalibrated expert
                m = xgb.fit(X_aug.iloc[idx], yr)
            self.experts[int(c)] = (m, classes)  # classes = global label order of cols
        return self

    def predict_proba(self, X, X_aug):
        from cluster import scale_with
        Z = scale_with(self.scaler, X)
        post = self.gmm.predict_proba(Z)
        hard = self.km.predict(Z)
        gp = self.global_model.predict_proba(X_aug)
        out = gp.copy()
        for c, (e, classes) in self.experts.items():   # batched per-cluster blend
            idx = np.where(hard == c)[0]
            if len(idx) == 0:
                continue
            ep = np.zeros((len(idx), self.n_classes))
            if isinstance(e, str) and e == 'constant':
                ep[:, classes[0]] = 1.0
            else:
                ep[:, classes] = e.predict_proba(X_aug.iloc[idx])  # scatter local→global cols
            w = post[idx, c][:, None]              # gate = cluster posterior
            blend = w * ep + (1 - w) * gp[idx]
            out[idx] = 0.45 * gp[idx] + 0.55 * blend
        out = out / np.clip(out.sum(axis=1, keepdims=True), 1e-9, None)  # proper distribution
        return out
