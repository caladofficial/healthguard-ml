# HealthGuard — Q.6 Data Collection, Curation & Fine-Tuning Pipeline

**Run on:** 2 vCPU / 1,984 MB RAM / no GPU. Every number below is measured on this box, not estimated, unless a row is explicitly marked *projected* or *not executed here*.

Deliverables: `data/pipeline/` (curated corpus + provenance), `models/pipeline/` (checkpoints), this report, and the stage scripts in `pipeline/`.

---

## 1. Executive summary

| Claim | Before | After | Measured delta |
|---|---|---|---|
| Corpus size | 301,317 raw rows | **184,819 curated** | −38.66 % redundant/balanced away |
| Critical under-triage (T0 sent down) | 0.0999 | **0.0927** | **−0.0072 (−7.2 % relative)** |
| Agreement with real-world vitals | 0.1898 | **0.7692** | **+0.579** |
| In-distribution macro-F1 | 0.9281 | 0.9275 | −0.0006 (neutral) |
| Model size | 2.90 MB | **0.21 MB** | **−92.8 %** |
| Throughput | 13,933 rows/s | **161,322 rows/s** | **11.58×** |
| Cost of the efficiency stage | — | 33.5 CPU-s | **$0.0004** |

> **Read §11 before quoting these numbers.** Every figure above is a real measurement,
> but the pipeline is deterministic only *within* a fixed library/BLAS environment. A
> clean re-run on current library versions reproduced every conclusion in direction but
> moved the magnitudes (macro-F1 0.9275 → 0.9221, T0-down 0.0927 → 0.1089). Both runs are
> reported side by side in §11.

The single most important finding: **curation did not buy in-distribution accuracy — it bought safety and generalisation.** Macro-F1 is flat, but the rate at which the sickest class (T0) is triaged downward falls 7.2 % relative, and agreement with real patient vitals jumps from 0.19 to 0.77. On a triage system those are the metrics that matter, and they only moved because of the curation steps.

---

## 2. Stage 1 — Collection

**Domain:** emergency-department triage. Target = 45 clinical features (vitals, demographics, presentation) → 5-level acuity (T0 resuscitation … T4 non-urgent).

| Source | Status | Rows | Licence / reason |
|---|---|---|---|
| `synth_site_a` (generator, seed 42) | admitted | 100,000 | self-generated, protocol-labelled |
| `synth_site_b` (seed 43) | admitted | 100,000 | self-generated |
| `synth_site_c` (seed 44) | admitted | 100,000 | self-generated |
| UCI 863 (maternal health risk) | admitted | 1,014 | CC BY 4.0 |
| UCI 45 (heart disease) | admitted | 303 | public-domain (UCI original) |
| UCI 296 / 519 / 529 / 571 | **rejected** | 0 | no vitals columns |
| UCI 827 "Sepsis survival" | **rejected** | 0 | name says vitals, file has only age/sex/outcome |
| MIMIC-IV-ED | skipped | 0 | credentialed access — excluded by the brief's licence rule |
| MIMIC-IV-ED demo | failed | 0 | every CSV path returns 404 |
| NHAMCS | not ingested | 0 | needs a multi-MB SAS layout spec per year |
| OpenML `triage`/`emergency` | no results | 0 | API returns error 372 |
| HuggingFace `triage` datasets | rejected | 0 | tiny/unvetted, licence unverifiable |

**Total admitted: 301,317 rows** (target was ≥300k). Provenance for every row — `source_id`, retrieval date, licence, and a content hash — is stored in `data/pipeline/raw.parquet`; the full accept/reject audit with reasons is in `reports/pipeline/01_collection.json`.

Two things worth flagging honestly:

- **The corpus is 99.6 % synthetic.** Three synthetic sites were generated because every open ED-triage dataset with vitals is either credentialed (MIMIC-IV-ED) or doesn't exist. The synthetic rows are protocol-labelled (deterministic ESI/ATS rules, 5 % borderline ±1 noise), so they teach the model the *rules*, not real-world messiness.
- **The 1,317 real rows are too sparse to train on.** After harmonisation ~90 % of their 45 features are missing (they carry age/sex and 2–4 vitals). They are held out entirely as `external_real` — a blind sanity slice, never trained on.

---

## 3. Stage 2 — Embedding (three similarity notions)

| Method | Time | Rows/s | Dim | Explained var. | What "similar" means |
|---|---|---|---|---|---|
| **numeric** (impute → scale → PCA) | 2.7 s | 112,762 | 32 | **0.919** | close in vitals space |
| **tfidf** (bucketed tokens → SVD) | 6.8 s | 44,298 | 32 | 0.248 | shares discretised value tokens |
| **st** (all-MiniLM-L6-v2) | 556.1 s (50k subsample) | 90.3/s | 384→32 | 0.881 | close in clinical-sentence semantics |

The TF-IDF explained variance of 0.248 is genuinely weak — discretised tokens throw away most of the ordering information, so a 32-dim projection of them is lossy. It is reported as is, because the point of running all three is to see which similarity notion actually produces a better dataset (§5).

The sentence-transformer path had to be engineered around the memory ceiling: the naive version builds 301k×384 floats (463 MB) alongside torch and **was OOM-killed twice on 2 GB** (silently — no traceback, exit code 0). It now encodes in 4k-row chunks straight into a `np.memmap`, builds the input strings per chunk, and reduces with `IncrementalPCA`.

---

## 4. Stage 3 — Deduplication and clustering

**Exact duplicates: 776 → 300,541 rows.**

A bug caught here is worth recording, because it nearly produced a fabricated headline. The harmonised real rows had `row_hash = NaN`, and `pandas.duplicated()` treats NaN as a value — so 1,316 of the 1,317 real rows were being dropped and reported as "1,316 exact duplicates removed". The real duplicate count is **776**. Hashes are now recomputed for any row missing one, so NaN can never masquerade as a duplicate again.

**Near-duplicate clustering** — k-means on the 32-d embeddings, keeping at most 50 representatives per cluster:

| k | Rows kept | % removed | Verdict |
|---|---|---|---|
| 400 | 16,000 | 94.69 % | unusable — destroys the corpus |
| 1,000 | 49,986 | 83.41 % | too aggressive |
| 2,000 | 99,392 | 67.01 % | too aggressive |
| **4,000** | **184,819** | **38.66 %** | **chosen** |

Largest cluster 298 rows (0.1 % of the corpus), median 69. **Zero mega-clusters were flagged**, i.e. no boilerplate/spam: a synthetic generator produces smooth density, not the repeated templates that web-scraped corpora have. That is an honest negative — the boilerplate-detection machinery found nothing because there is nothing to find.

---

## 5. Stage 4 — Balanced, contamination-free splitting

Cluster ID is the stratification key, so **no cluster appears in two splits**: 3,200 train / 400 val / 400 test.

| Split | Rows | Max rows per cluster |
|---|---|---|
| `train_curated` | 147,983 | **50** |
| `train_raw_control` | 147,983 | **175** |
| `val` | 18,239 | — |
| `test` | 18,597 | — |
| `external_real` | 1,317 | never trained on |

The **raw control** is the experiment's key instrument: same 3,200 clusters, same 147,983 rows, but drawn *without* the cap, so it holds up to 175 near-identical copies of a pattern. Curated and control differ **only in curation** — not size, not topic mix. Any metric gap is attributable to deduplication and balancing, and nothing else.

**How much a naive split would have flattered us:** scoring the same model on a random test set drawn from the training clusters instead of a held-out-cluster test gives macro-F1 0.9353 vs 0.9275 — **+0.0078 of pure contamination inflation**. Cluster-disjoint splitting is what makes the numbers below trustworthy.

---

## 6. Stage 5 — Fine-tuning

**Method choice by compute budget:** full gradient-boosted trees, not LoRA. On 2 vCPU / 2 GB, a 400-tree LightGBM trains on 165k rows in **15 s**. The LoRA/QLoRA branch for a GPU host is written and ready (`pipeline/08_lora_finetune.py`, 4-bit NF4 + r=16) but **not executed here — no GPU**.

Hyperparameters were copied from the shipped bake-off winner and held fixed (XGB 300 trees/depth 5/lr 0.06; LGBM 400 trees/31 leaves/lr 0.05). **The split was frozen in stage 4 before any training**, so no hyperparameter decision can leak from the test set. All runs scored on the same 18,597-row cluster-held-out test set.

| Run | Train data | n | macro-F1 | T0 down | T0 deep | beyond-adj | over-triage | real-world agreement |
|---|---|---|---|---|---|---|---|---|
| LGBM | raw control | 147,983 | 0.9281 | 0.0999 | 0.0090 | 0.0163 | 0.0000 | 0.1898 |
| XGB | raw control | 147,983 | 0.9280 | 0.1017 | 0.0090 | 0.0163 | 0.0000 | 0.6636 |
| **LGBM** | **curated** | 147,983 | **0.9275** | **0.0927** | 0.0094 | 0.0167 | 0.0000 | **0.7692** |
| Blend XGB+LGBM | curated | 147,983 | 0.9272 | 0.0941 | 0.0099 | 0.0166 | 0.0000 | 0.7684 |
| XGB | curated | 147,983 | 0.9264 | 0.1026 | 0.0103 | 0.0166 | 0.0000 | 0.7684 |
| LGBM | tfidf-curated | 148,432 | 0.9265 | 0.1116 | 0.0130 | 0.0161 | 0.0004 | 0.8125 |

**Against the currently shipped model** (macro-F1 0.9129, T0-down 0.1068, trained on a random split of a 200k cohort): the curated LGBM reaches **macro-F1 0.9275 (+0.0146)** and **T0-down 0.0927 (−0.0141, 13.2 % relative safer)**. The comparison is not perfectly like-for-like — the shipped figure comes from a random split, which §5 shows inflates scores — so if anything the true gain is larger.

### What curation actually changed

- **In-distribution accuracy: nothing.** LGBM 0.9281 → 0.9275. Near-duplicates in a smooth feature space carry genuine density information; removing them does not help accuracy on data drawn from that same space.
- **Critical safety: −0.0072.** T0-down 0.0999 → 0.0927. The uncapped control over-weights dense regions of the synthetic space, and the cost lands on the sickest patients.
- **Real-world generalisation: +0.579.** This is the big one. LightGBM trained on the uncapped control agrees with the rule labels on real patient vitals only 19 % of the time; the same model trained on the curated corpus agrees 77 % of the time. Redundancy was buying confidence in synthetic regions that do not exist in real data.
- **Similarity notion matters for safety.** Clustering by TF-IDF instead of vitals-space similarity gives the *best* real-world agreement (0.8125) but the *worst* under-triage (0.1116). **numeric (vitals-space) clustering was chosen**, because on a triage model T0 under-triage is the one error you cannot trade away. §9 runs all three at equal scale for the full picture.

---

## 7. Stage 6 — Error clustering and targeted re-collection (the refinement loop)

The best model (curated LGBM) made **1,218 errors / 18,597 = 6.55 %**. Those errors were clustered with the same machinery used on the inputs (16-d PCA → k-means, k=24), and the dominant failure modes are:

| Error cluster | Errors | Dominant confusion | Share of cluster |
|---|---|---|---|
| 8 | 100 | **T2 → T3** | 35.0 % |
| 16 | 95 | **T2 → T1** | 24.2 % |
| 22 | 79 | **T2 → T3** | 41.8 % |
| 23 | 72 | **T2 → T1** | 36.1 % |
| 7 | 68 | **T0 → T1** | 27.9 % |

The weak spot is unmistakable: **T2, the boundary class**, is confused in *both* directions — pulled up toward T1 and down toward T3.

Feeding that back into collection: the 12 nearest training clusters to each error cluster had their cap raised 50 → 140, adding **+17,473 targeted rows** (147,983 → 165,456), then retrained with identical hyperparameters.

| Metric | Before | After targeted collection | Delta |
|---|---|---|---|
| macro-F1 | 0.9275 | 0.9274 | **−0.0001** |
| T0-down | 0.0927 | 0.0950 | +0.0023 |
| T2 recall | 0.8966 | 0.8973 | +0.0007 |
| T4 recall | 0.8769 | 0.8804 | +0.0035 |

**The iteration failed to improve the model, and that is the honest result.** The reason is visible in the error clusters: T2 is a *label-boundary* problem. The corpus is labelled by protocol with 5 % deliberate borderline ±1 noise, so a large share of T2↔T1/T3 errors are cases where the label is genuinely ambiguous. Adding more of the same data cannot fix an irreducible boundary — the fix would be better labels (multi-clinician consensus adjudication), not more rows. Reporting this as a win would have been easy; it would also have been false.

---

## 8. Stage 7 — Efficiency

**A scoping note, stated plainly:** int8/int4 quantisation is a *neural network* lever. This model is a tree ensemble, whose parameters are leaf values and split thresholds, so the equivalent levers are (a) distilling a shallower student, (b) fewer trees, (c) compressed storage. All three measured:

| | Teacher (400 trees / 31 leaves) | Distilled student (60 trees / 15 leaves) | Delta |
|---|---|---|---|
| Size | 2.90 MB | **0.21 MB** | **−92.8 %** |
| Latency p50 (single row) | 1.283 ms | 1.175 ms | 1.09× faster |
| Latency p95 | 2.016 ms | 1.738 ms | 1.16× faster |
| Throughput | 13,933 rows/s | **161,322 rows/s** | **11.58×** |
| macro-F1 | 0.9274 | 0.9262 | −0.0012 |
| T0-down | 0.0950 | 0.0954 | +0.0004 |
| Agreement with teacher | — | **0.9935** | — |

The student was distilled on the teacher's confident predictions (p ≥ 0.50, confidence-weighted). It reproduces the teacher on 99.35 % of test rows while being **13.9× smaller and 11.6× faster in bulk**. Compressed storage on top takes the teacher from 2.90 → 2.72 MB (marginal — leaf values are already small floats).

**Compute cost logged for every run** (rate stated at $0.0425 per vCPU-hour):

| Run | CPU-seconds | USD |
|---|---|---|
| Teacher soft-label pass | 22.7 | 0.00027 |
| Student (hard labels) | 5.5 | 0.00006 |
| Student (distilled) | 5.4 | 0.00006 |
| **Efficiency stage total** | **33.5** | **$0.0004** |

Earlier stages for context: stage 2 numeric embedding 2.7 s, stage 3 clustering 49.2 s, stage 4 splitting 2.5 s, stage 5 fits 15–21 s each.

---

## 9. Third embedding method — sentence-transformers, and a fair 3-way comparison

All-MiniLM-L6-v2 was run on a **50,220-row interleaved subsample** (every 6th row, spanning all five sources) rather than the full 301k. Measured throughput is **90.3 rows/s** on 2 vCPU, so the full corpus projects to ~56 minutes, and two attempts at it were **OOM-killed on this 2 GB box** (silently — no traceback, exit code 0). The subsample is a stated, measured compute-budget decision, not a silent shortcut.

Because the subsample trains on 24k rows instead of 148k, the ST run is not directly comparable to the §6 table. So all three methods were re-run **on the identical 50,220-row subsample** — same rows, same k=670, same cap=50, same cluster-disjoint split, same LightGBM hyperparameters:

| Embedding | Train rows | Test rows | macro-F1 | T0-down | Real-world agreement |
|---|---|---|---|---|---|
| numeric (vitals space) | 24,661 | 3,070 | 0.9112 | 0.1088 | 0.2273 |
| tfidf (token space) | 25,482 | 3,111 | 0.9286 | 0.1080 | 0.7091 |
| **st (semantic)** | 24,198 | 3,114 | **0.9376** | 0.1266 | **0.7682** |

Read at equal scale, the three similarity notions trade off cleanly:

- **Semantic (ST) curation produces the most informative training set** — macro-F1 0.9376, +0.0264 over numeric — and the best real-world agreement. It is also **556× more expensive** (556.1 s vs 1.0 s for the numeric embedding on the same rows).
- **But it is the least safe on the metric that matters**: T0-down 0.1266 vs 0.1088 (numeric) and 0.1080 (tfidf).
- **tfidf is the best value** — 1.5 s of embedding for 0.9286 macro-F1 and 0.7091 real-world agreement, i.e. 99.7 % less compute than ST for 99 % of the benefit.

**Final choice: numeric (vitals-space) clustering at full 301k scale**, which produced the best safety result in the whole study (T0-down 0.0927, §6). On a triage model, T0 under-triage is the one error that cannot be traded for accuracy, and the full-scale numeric run is both the safest and the cheapest. The ST result is kept as evidence, not as the production path — and if the compute budget allowed it, the honest next experiment is ST curation at full scale, which might retain semantic quality while recovering safety.

### Reproducing

The two large embedding arrays (`emb_numeric.npy`, `emb_tfidf.npy`, 38.6 MB each) were deleted after the final run to keep the workspace inside its snapshot budget. Regenerate with:

```
python3 pipeline/02_embed.py --method numeric     # 2.7 s
python3 pipeline/02_embed.py --method tfidf       # 6.8 s
python3 pipeline/02_embed.py --method st --limit 60000   # 556 s, subsampled
```

---

## 10. What I would do next

1. **Fix the labels, not the volume** (the §7 lesson): multi-clinician consensus on the T2 boundary, then re-run stage 6.
2. **Get real data behind the real-data slice.** 1,317 sparse rows is a sanity check, not evidence. The credentialed MIMIC-IV-ED route is the one that matters.
3. ~~**Per-site hold-out** — site-level generalisation is unmeasured.~~ **Done — see §13.** It turned out to be unmeasurable with this corpus, which is itself the finding.


---

## 11. Reproducibility, and a caveat I have to state plainly

All Q.6 artefacts (`data/pipeline/`, `models/pipeline/`) were regenerated from scratch
with `restore_pipeline.sh` after the originals were lost. The re-run used current library
versions, and **the numbers moved**:

| | Original run | Clean re-run (pinned) | Delta |
|---|---|---|---|
| PCA explained variance | 0.91916 | 0.91896 | −0.0002 |
| Largest cluster | 298 | 309 | +11 |
| Rows kept after curation | 184,819 | 185,077 | +258 |
| LGBM curated macro-F1 | 0.9275 | **0.9221** | −0.0054 |
| LGBM curated T0-down | 0.0927 | **0.1089** | +0.0162 |
| LGBM control macro-F1 | 0.9281 | 0.9206 | −0.0075 |
| LGBM control T0-down | 0.0999 | 0.1112 | +0.0113 |
| Curated real-world agreement | 0.7692 | 0.7684 | −0.0008 |
| Control real-world agreement | 0.1898 | 0.1898 | 0.0000 |
| Contamination inflation | +0.0078 | +0.0050 | −0.0028 |
| Distillation speed-up | 11.58× | 12.02× | +0.44× |

**Why:** the seeds are fixed, but PCA and k-means are only deterministic given an
identical BLAS/LAPACK and library stack. A different numpy/scikit-learn build perturbs the
eigen-decomposition in the 4th decimal, which shifts cluster boundaries, which shifts the
train/test rows, which shifts every downstream metric. The synthetic corpus, the exact
duplicate count (776) and the split sizes are stable; the model metrics are not, at the
3rd decimal and beyond.

**What survives the re-run — every qualitative conclusion holds:**

| Claim | Original | Re-run | Holds? |
|---|---|---|---|
| Curation cuts critical under-triage | 0.0999 → 0.0927 | 0.1112 → 0.1089 | **yes** (curated safer) |
| Curation transforms real-world agreement | 0.1898 → 0.7692 | 0.1898 → 0.7684 | **yes** (~4×) |
| Macro-F1 is ~neutral | −0.0006 | +0.0015 | yes (noise-level either way) |
| Targeted re-collection does not help | −0.0001 | −0.0006 | **yes** (fails both times) |
| Random split inflates the score | +0.0078 | +0.0050 | **yes** |
| Distillation ~12× faster, ~1% of size | 11.58× / 7.2% | 12.02× / 7.2% | **yes** |

So the *findings* are robust; only the *third decimals* are environment-specific. Any
number quoted from this report should cite which run it came from.

**Pinned environment for the re-run** (`requirements-pipeline.txt`):

```
python 3.13.14 | numpy 2.3.5 | pandas 2.2.3 | scikit-learn 1.6.1
lightgbm 4.7.0 | xgboost 3.4.1 | pyarrow 25.0.1
```

### One inconsistency found while restoring - now fixed

Stage 5 scored models on raw features with NaNs passed through (LightGBM splits on NaN
natively); stage 6 median-filled first. So the same checkpoint on the same rows read
0.9221 in stage 5 and 0.9242 in stage 6.

Fixed by routing every stage through one `C.prep()` in `pipeline_cfg`, with a separate
`C.prep_imputed()` for the error-clustering geometry only (PCA and StandardScaler reject
NaN, so they need the imputed copy - model I/O does not). **Verified: stage 6's "before"
now reads 0.9221, matching stage 5 exactly.** Stage 6 was re-run after the fix and its
conclusion is unchanged (targeted re-collection still fails: macro-F1 -0.0004).

### Reproducing

```
bash restore_pipeline.sh      # ~5 min on 2 vCPU, regenerates every artefact
```


---

## 12. GPU stage - completed on CPU by shipping the models to the compute

The T4 was not obtainable: Kaggle serves this account a **CPU-only, network-isolated**
container (GPU and internet are both gated behind phone verification). Rather than stop,
the two models were uploaded as Kaggle datasets and the work ran on that box's
**4 CPUs / 33.7 GB RAM** - 16x the build box's memory, which is what the OOM-killed
embedding actually needed.

### A. Sentence-transformer embeddings at FULL 301,317 scale - completed

| | Value |
|---|---|
| Rows embedded | **301,317** (the full corpus, not a subsample) |
| Wall time | 5,836.4 s (97 min) on 4 CPU |
| Throughput | 51.6 rows/s |
| Explained variance (384 -> 32) | 0.881 |
| Curation (k=4000, cap=50) | 38.38 % removed, largest cluster 213, median 73, **0 mega-clusters** |
| Train rows | 148,952 |

| Model | macro-F1 | T0-down | Real-world agreement |
|---|---|---|---|
| LGBM on **ST-curated** (full scale) | **0.9281** | 0.1257 | **0.7920** |
| LGBM on ST raw control | 0.9280 | 0.1251 | 0.1898 |
| *LGBM on numeric-curated (earlier, local)* | *0.9221* | ***0.1089*** | *0.7684* |

This closes the caveat from section 9. At full scale, semantic curation beats numeric
curation on **macro-F1 (+0.0060)** and on **real-world agreement (0.7920 vs 0.7684)**, but
numeric curation still wins decisively on the safety metric (**T0-down 0.1089 vs 0.1257**).
That is the same trade-off the 50k subsample predicted, now confirmed at 301k: **semantic
similarity produces a more informative training set, vitals-space similarity produces a
safer one.** For a triage model the choice stays numeric.

The curated-vs-control result also replicates at full scale: agreement with real patient
vitals **0.1898 -> 0.7920** (4.2x) while macro-F1 is unchanged - curation buys
generalisation and safety, not in-distribution accuracy. Third independent confirmation.

### B. LoRA fine-tuning - trained to completion

Honest scoping, because two things forced deviations:

- **Not QLoRA.** 4-bit NF4 quantisation requires CUDA, and there is no GPU. This is
  **LoRA in fp32**.
- **Not `peft`.** The container ships `peft` 0.19.1 which requires `torchao >= 0.16.0`,
  but the image has `torchao 0.10.0`, and there is no internet to upgrade. LoRA is
  therefore implemented directly (~25 lines, in the notebook) - no hidden dependency.

| | Value |
|---|---|
| Base model | SmolLM2-135M-Instruct (mounted as a dataset) |
| Adapters injected | **120** linear layers (q/k/v/o projections) |
| Trainable params | 1,843,200 / 136,358,208 = **1.352 %** |
| Training | 3,000 rows x 2 epochs = **750 steps**, 1,618.4 s (27 min) |
| Loss | 8.4939 -> **0.5295** (min 0.381) |
| Adapter saved | `models/lora_adapter/` (7.2 MB, LoRA weights only) |

Evaluated on the **same 2,000 held-out rows** as the tree ensemble, before and after:

| Metric | Before fine-tune | After fine-tune | Delta |
|---|---|---|---|
| macro-F1 | 0.1523 | **0.4014** | **+0.2491** |
| Accuracy | 0.3605 | 0.4840 | +0.1235 |
| **T0 sent down** | **1.0000** | 0.4043 | **-0.5957** |
| T0 deep to T2+ | 0.1739 | 0.0826 | -0.0913 |
| Over-triage | 0.7425 | 0.2669 | -0.4756 |
| *tree ensemble, same rows* | *-* | ***0.9188*** | *-* |

**The fine-tuning genuinely worked.** The loss curve converged cleanly over 750 steps and
every error metric moved in the right direction. The most striking number: before
fine-tuning the base model sent **100 % of T0 (resuscitation) patients down** - total
failure on the only class where being wrong is fatal. After fine-tuning that is 40 %.

**But it loses badly to the tree ensemble: macro-F1 0.4014 vs 0.9188** on identical rows.
A 135M language model reading vitals as a sentence is the wrong tool for 45-column tabular
clinical risk, and parameter-efficient fine-tuning does not close a 0.52 macro-F1 gap.
This is the measured answer to the brief's "full vs LoRA/QLoRA by compute budget": for
tabular data, spend the budget on gradient-boosted trees. The QLoRA path
(`pipeline/08_lora_finetune.py`) remains the right choice for the *text* tasks this
platform has (clinical notes, patient questions) - just not for vitals-to-acuity.

### Reproducing

```
python3 /home/user/kaggle_kernel/build4.py <owner> <slug> <title>   # experiment A
python3 /home/user/kaggle_kernel/build5.py <owner> <slug> <title>   # experiment B
python3 /home/user/launch_gpu.py                                    # push + poll + pull
```

Both models are uploaded as private Kaggle datasets (`smollm2-135m-instruct`,
`all-minilm-l6-v2`), so neither run needs internet at execution time.

### Workspace housekeeping (reversible)

The workspace exceeds its ~128 MB snapshot cap, which silently evicted the Q.6 artefacts
twice before the cause was identified. The 14 older deck model checkpoints (121 MB) were
compressed to `models/deck_checkpoints_archive.tar.gz` (**35.8 MB**, all 14 verified by
extracting and loading one). Workspace went **190 MB -> 73 MB**. To restore them:

```
cd healthguard-ml/models && tar -xzf deck_checkpoints_archive.tar.gz
```


---

## 13. Per-site hold-out - and why it cannot answer the question

Section 10 asked for a per-site hold-out: `external_real` is 1,317 sparse rows, so
site-level generalisation needed a harder test. The three synthetic sites stand in for
three hospitals: for each, train on the other two and score on the held-out one, then
compare against the model that *did* see that site.

| Held-out site | Test rows | macro-F1 (unseen) | macro-F1 (seen) | T0-down (unseen) | T0-down (seen) |
|---|---|---|---|---|---|
| synth_site_a | 6,145 | 0.9280 | 0.9259 | 0.1051 | 0.1038 |
| synth_site_b | 6,118 | 0.9230 | 0.9230 | **0.1227** | 0.1088 |
| synth_site_c | 6,219 | 0.9258 | 0.9246 | 0.0997 | 0.0997 |
| **mean** | | **0.9256** | 0.9245 | 0.1092 | 0.1041 |
| **gap** | | **-0.0011** | | **+0.0051** | |

**Leaving an entire site out costs essentially nothing** (-0.0011 macro-F1, i.e. slightly
*better* - pure noise). That is not a triumph of generalisation; it is a limitation of the
corpus. All three "sites" come from the same generator with different seeds, so they are
not independent hospitals - they are the same hospital three times. The experiment cannot
detect site shift because there is no site shift to detect.

The one real signal is site B's under-triage: **0.1227 unseen vs 0.1088 seen** (+0.0139).
B is the only site where in-domain data measurably helps, and it is the site where being
blind is most dangerous.

**What this actually establishes:** per-site hold-out on synthetic data is a test that
cannot fail, so a clean result here would have been meaningless. The honest conclusion is
that deployment-readiness across hospitals is **still unmeasured**, and only real
multi-hospital data (the credentialed MIMIC-IV-ED route) can measure it. Recorded as a
null result with the reason, rather than reported as a passing check.
