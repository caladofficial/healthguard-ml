# HealthGuard — Triage Risk Engine (ML)

**EVOLVEX IT SOLUTIONS PVT. LTD.** · High-artillery clinical risk-ratio + T0–T4 triage model.

> AI is assistive only. Triage = workflow urgency, **not** diagnosis.

## Pipeline (`src/run_all.py`)
1. `collect.py` — UCI/OpenML public clinical datasets → `data/raw/` + `MANIFEST.json` (license/provenance)
2. `datasets.py` — harmonized sets → `data/processed/`
3. `cluster.py` — **KMeans + GMM + DBSCAN** phenotyping, silhouette-k, purity report
4. `train_specialists.py` — per condition: de-clustered **MoE** (per-cluster calibrated XGB experts, GMM-posterior gating) over a **Stacking group** (XGB+LGBM+RF+Neuro MLP) → isotonic-calibrated **risk ratio** (p ÷ baseline, tiered)
5. `train_triage.py` — synthetic 45k ED cohort (deterministic ESI/ATS protocol labels, reason codes) → fused T0–T4 with safety floors + **red-flag rule overrides** + `requires_human_review`
6. `evaluate.py` — golden 20% holdout: ROC-AUC / PR-AUC / Brier / **ECE** / sens@95spec / **under-triage rates** → `reports/metrics.json` + `MODEL_CARD.md`

## Use
```bash
python3 src/run_all.py                      # full train
python3 src/infer.py case_example.json      # CLI inference
python3 src/api.py 8088                     # HTTP API  POST /triage
```

Outputs `risk.risk_ratio` + `triage.class` (T0 emergency → T4 follow-up) with reason codes and confidence.


---

## Where this lives on GitHub

<https://github.com/caladofficial/healthguard-ml>

The platform itself lives in a sibling repo, <https://github.com/caladofficial/healthguard>.

**What is committed here:** all pipeline code, every stage report (`reports/pipeline/*.json`,
`reports/gpu/*.log`), and the model checkpoints that cost real compute to produce —
`models/lora_adapter/lora_weights.pt` (7.4 MB, trained on Kaggle), the fine-tuned
`lgbm_curated_lgbm.joblib`, the refined model, and the distilled student.

**What is deliberately not committed** (~79 MB, all documented in `.gitignore` with restore
instructions): the intermediate parquets and raw UCI CSVs, plus the 35.8 MB
`models/deck_checkpoints_archive.tar.gz`, which is the only local copy of the 14 deck
checkpoints the live site loads — keep it, do not delete it.

Run `bash restore_pipeline.sh` to regenerate the excluded data from the committed code.
Every number those files produced is permanently recorded in `reports/` and
`PIPELINE_REPORT.md`, both of which *are* tracked.

**Gotcha:** `pipeline/01_collect.py` reads `data/*.csv` from disk (lines 143-146); it does
not re-download them. `data/uci_296.csv` (19.5 MB) must be kept, or re-fetched manually from
the UCI flat CSV endpoint.
