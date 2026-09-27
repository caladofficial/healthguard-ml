# MODEL CARD — HealthGuard Triage Risk Engine (round 2)

**Owner:** EVOLVEX IT SOLUTIONS PVT. LTD.  ·  **Product:** HealthGuard  ·  **Version:** triage-fusion-moe-v2 (2026-09-24)

> **AI is assistive only.** This system never diagnoses and never replaces a clinician. Triage output is *workflow urgency* — what to see first — not what the patient has.


## 1. Intended use
Emergency-department intake support: per-condition **risk ratios** across 12 heads (CAD ×2 views, diabetes ×3 views, fetal risk, maternal risk, liver ×2 views, sepsis, heart-failure mortality) and a fused **T0–T4 urgency** label with reason codes, confidence and an explicit `requires_human_review` flag (spec §29–30).


## 2. Model architecture (the 'high artillery')
| Stage | Method |
|---|---|
| Phenotyping / clustering | KMeans (silhouette-chosen k) + GaussianMixture + DBSCAN |
| De-clustering | MoE: per-cluster calibrated XGB experts (label-remapped), GMM-posterior gating |
| Grouping | StackingClassifier (XGB+LGBM+RF+Neuro) + soft voting |
| Neuro | MLP (64,32) inside the stack |
| Calibration | CalibratedClassifierCV — isotonic (big sets) / sigmoid (small sets) |
| Hyper-params | RandomizedSearchCV over XGB (40 iters small sets, 14 large) |
| Risk ratio | p ÷ cohort baseline prevalence, tiered low/moderate/high/very-high |
| Triage fusion | 4-model blend (XGB+LGBM+Stack+MoE) with **validation-grid weights** + safety floors + **red-flag rule override** |
| Features | raw physiology + engineered clinical scores (shock index, qSOFA, SIRS, NEWS-like, age-risk) |


## 3. Training data (collected & provenance-tracked, MANIFEST.json)
Public-repository clinical data (UCI/OpenML, family A). Synthetic sets are family-E flagged.

| Set | Rows × Feats | Classes | Source |
|---|---|---|---|
| cad | 920 × 13 | 411 / 509 | UCI 4 cohorts (Cleveland/Hungarian/Swiss/VA) |
| cad_aux | 270 × 13 | 150 / 120 | UCI 45 heart-statlog |
| cad_spect | 349 × 44 | perfusion abnormal vs normal | OpenML 337 SPECTF |
| diabetes | 768 × 8 | 500 / 268 | Pima Indians (OpenML 37) |
| diabetes_symptoms | 520 × 16 | Positive/Negative | UCI 529 Early Stage Diabetes Risk |
| diabetes_readmit | 101,766 × ~35 | <30d vs rest | UCI 296 Diabetes 130-US Hospitals |
| fetal_risk | 2,126 × 37 | 1,655 / 295 / 176 | UCI 193 Cardiotocography (NSP) |
| maternal_risk | 1,014 × 6 | 406 / 336 / 272 | UCI 863 Maternal Health |
| liver | 583 × 10 | 167 / 416 | UCI 225 ILPD |
| liver_hcv | 615 × 12 | healthy / disease / cirrhosis | UCI 571 HCV labs |
| hf_mortality | 299 × 12 | survived / died | UCI 519 Heart Failure Clinical Records |
| sepsis | 30,000 × 13 | qSOFA-structured | **Synthetic family-E** |
| triage cohort | 100,000 × 47 | T0–T4 mix | **Synthetic family-E**, deterministic ESI/ATS protocol labels + 5% inter-rater noise |

BUPA Liver Disorders (UCI 60) deliberately excluded: 'selector' label semantics disputed.


## 4. Evaluation (20% golden holdout, stratified, SEED=42)
```json

{
  "specialists": {
    "cad": {
      "roc_auc": 0.9121,
      "pr_auc": 0.9298,
      "brier": 0.1254,
      "sens_at_95spec": {
        "threshold": 0.8441,
        "sensitivity": 0.549,
        "specificity": 0.95
      },
      "ece": 0.0855,
      "reliability": [
        {
          "bin": "[0.0,0.1)",
          "n": 5,
          "empirical": 0.0,
          "predicted": 0.096
        },
        {
          "bin": "[0.1,0.2)",
          "n": 46,
          "empirical": 0.043,
          "predicted": 0.139
        },
        {
          "bin": "[0.2,0.3)",
          "n": 9,
          "empirical": 0.556,
          "predicted": 0.231
        },
        {
          "bin": "[0.3,0.4)",
          "n": 7,
          "empirical": 0.429,
          "predicted": 0.347
        },
        {
          "bin": "[0.4,0.5)",
          "n": 10,
          "empirical": 0.5,
          "predicted": 0.443
        },
        {
          "bin": "[0.5,0.6)",
          "n": 4,
          "empirical": 0.0,
          "predicted": 0.564
        },
        {
          "bin": "[0.6,0.7)",
          "n": 12,
          "empirical": 0.583,
          "predicted": 0.648
        },
        {
          "bin": "[0.7,0.8)",
          "n": 13,
          "empirical": 0.692,
          "predicted": 0.755
        },
        {
          "bin": "[0.8,0.9)",
          "n": 66,
          "empirical": 0.894,
          "predicted": 0.867
        },
        {
          "bin": "[0.9,1.0)",
          "n": 12,
          "empirical": 1.0,
          "predicted": 0.903
        }
      ],
      "risk_ratio_deciles": {
        "median": 1.23,
        "p90": 1.61,
        "max": 1.64,
        "tier_low": 88,
        "tier_moderate": 96,
        "tier_high": 0,
        "tier_very_high": 0
      },
      "positive_or_high_class_prevalence": 0.5543
    },
    "diabetes": {
      "roc_auc": 0.8207,
      "pr_auc": 0.7152,
      "brier": 0.1623,
      "sens_at_95spec": {
        "threshold": 0.6526,
        "sensitivity": 0.4259,
        "specificity": 0.95
      },
      "ece": 0.0543,
      "reliability": [
        {
          "bin": "[0.0,0.1)",
          "n": 26,
          "empirical": 0.038,
          "predicted": 0.091
        },
        {
          "bin": "[0.1,0.2)",
          "n": 29,
          "empirical": 0.138,
          "predicted": 0.138
        },
        {
          "bin": "[0.2,0.3)",
          "n": 18,
          "empirical": 0.111,
          "predicted": 0.26
        },
        {
          "bin": "[0.3,0.4)",
          "n": 13,
          "empirical": 0.385,
          "predicted": 0.357
        },
        {
          "bin": "[0.4,0.5)",
          "n": 17,
          "empirical": 0.529,
          "predicted": 0.446
        },
        {
          "bin": "[0.5,0.6)",
          "n": 17,
          "empirical": 0.529,
          "predicted": 0.548
        },
        {
          "bin": "[0.6,0.7)",
          "n": 14,
          "empirical": 0.571,
          "predicted": 0.652
        },
        {
          "bin": "[0.7,0.8)",
          "n": 19,
          "empirical": 0.789,
          "predicted": 0.742
        },
        {
          "bin": "[0.8,0.9)",
          "n": 1,
          "empirical": 1.0,
          "predicted": 0.819
        }
      ],
      "risk_ratio_deciles": {
        "median": 0.98,
        "p90": 2.064,
        "max": 2.34,
        "tier_low": 89,
        "tier_moderate": 45,
        "tier_high": 20,
        "tier_very_high": 0
      },
      "positive_or_high_class_prevalence": 0.3506
    },
    "fetal_risk": {
      "roc_auc_ovr": 0.9977,
      "log_loss": 0.0592,
      "confusion": [
        [
          332,
          0,
          0
        ],
        [
          8,
          51,
          0
        ],
        [
          0,
          0,
          35
        ]
      ],
      "positive_or_high_class_prevalence": 0.0822
    },
    "maternal_risk": {
      "roc_auc_ovr": 0.9459,
      "log_loss": 0.4458,
      "confusion": [
        [
          70,
          10,
          1
        ],
        [
          11,
          51,
          5
        ],
        [
          2,
          0,
          53
        ]
      ],
      "positive_or_high_class_prevalence": 0.2709
    },
    "liver": {
      "roc_auc": 0.8037,
      "pr_auc": 0.9208,
      "brier": 0.1667,
      "sens_at_95spec": {
        "threshold": 0.82,
        "sensitivity": 0.4819,
        "specificity": 0.95
      },
      "ece": 0.0788,
      "reliability": [
        {
          "bin": "[0.4,0.5)",
          "n": 10,
          "empirical": 0.4,
          "predicted": 0.469
        },
        {
          "bin": "[0.5,0.6)",
          "n": 17,
          "empirical": 0.529,
          "predicted": 0.566
        },
        {
          "bin": "[0.6,0.7)",
          "n": 22,
          "empirical": 0.455,
          "predicted": 0.65
        },
        {
          "bin": "[0.7,0.8)",
          "n": 21,
          "empirical": 0.81,
          "predicted": 0.753
        },
        {
          "bin": "[0.8,0.9)",
          "n": 47,
          "empirical": 0.915,
          "predicted": 0.863
        }
      ],
      "risk_ratio_deciles": {
        "median": 1.05,
        "p90": 1.24,
        "max": 1.27,
        "tier_low": 82,
        "tier_moderate": 35,
        "tier_high": 0,
        "tier_very_high": 0
      },
      "positive_or_high_class_prevalence": 0.7094
    },
    "sepsis": {
      "roc_auc": 0.8698,
      "pr_auc": 0.8494,
      "brier": 0.1447,
      "sens_at_95spec": {
        "threshold": 0.7846,
        "sensitivity": 0.4896,
        "specificity": 0.95
      },
      "ece": 0.011,
      "reliability": [
        {
          "bin": "[0.0,0.1)",
          "n": 1052,
          "empirical": 0.071,
          "predicted": 0.067
        },
        {
          "bin": "[0.1,0.2)",
          "n": 768,
          "empirical": 0.141,
          "predicted": 0.146
        },
        {
          "bin": "[0.2,0.3)",
          "n": 558,
          "empirical": 0.233,
          "predicted": 0.245
        },
        {
          "bin": "[0.3,0.4)",
          "n": 426,
          "empirical": 0.34,
          "predicted": 0.352
        },
        {
          "bin": "[0.4,0.5)",
          "n": 419,
          "empirical": 0.463,
          "predicted": 0.451
        },
        {
          "bin": "[0.5,0.6)",
          "n": 414,
          "empirical": 0.541,
          "predicted": 0.548
        },
        {
          "bin": "[0.6,0.7)",
          "n": 372,
          "empirical": 0.648,
          "predicted": 0.654
        },
        {
          "bin": "[0.7,0.8)",
          "n": 560,
          "empirical": 0.782,
          "predicted": 0.757
        },
        {
          "bin": "[0.8,0.9)",
          "n": 687,
          "empirical": 0.873,
          "predicted": 0.851
        },
        {
          "bin": "[0.9,1.0)",
          "n": 744,
          "empirical": 0.925,
          "predicted": 0.933
        }
      ],
      "risk_ratio_deciles": {
        "median": 0.95,
        "p90": 1.93,
        "max": 2.11,
        "tier_low": 3511,
        "tier_moderate": 2265,
        "tier_high": 224,
        "tier_very_high": 0
      },
      "positive_or_high_class_prevalence": 0.4738
    },
    "cad_aux": {
      "roc_auc": 0.9097,
      "pr_auc": 0.8694,
      "brier": 0.1202,
      "sens_at_95spec": {
        "threshold": 0.8274,
        "sensitivity": 0.7083,
        "specificity": 0.95
      },
      "ece": 0.1625,
      "reliability": [
        {
          "bin": "[0.0,0.1)",
          "n": 3,
          "empirical": 0.0,
          "predicted": 0.09
        },
        {
          "bin": "[0.1,0.2)",
          "n": 14,
          "empirical": 0.071,
          "predicted": 0.134
        },
        {
          "bin": "[0.2,0.3)",
          "n": 4,
          "empirical": 0.0,
          "predicted": 0.253
        },
        {
          "bin": "[0.3,0.4)",
          "n": 6,
          "empirical": 0.167,
          "predicted": 0.353
        },
        {
          "bin": "[0.4,0.5)",
          "n": 2,
          "empirical": 1.0,
          "predicted": 0.451
        },
        {
          "bin": "[0.5,0.6)",
          "n": 1,
          "empirical": 0.0,
          "predicted": 0.54
        },
        {
          "bin": "[0.6,0.7)",
          "n": 2,
          "empirical": 1.0,
          "predicted": 0.65
        },
        {
          "bin": "[0.7,0.8)",
          "n": 2,
          "empirical": 0.0,
          "predicted": 0.771
        },
        {
          "bin": "[0.8,0.9)",
          "n": 16,
          "empirical": 0.938,
          "predicted": 0.876
        },
        {
          "bin": "[0.9,1.0)",
          "n": 4,
          "empirical": 0.75,
          "predicted": 0.908
        }
      ],
      "risk_ratio_deciles": {
        "median": 0.935,
        "p90": 2.02,
        "max": 2.05,
        "tier_low": 29,
        "tier_moderate": 14,
        "tier_high": 11,
        "tier_very_high": 0
      },
      "positive_or_high_class_prevalence": 0.4444
    },
    "diabetes_readmit": {
      "roc_auc": 0.6881,
      "pr_auc": 0.2363,
      "brier": 0.0936,
      "sens_at_95spec": {
        "threshold": 0.2129,
        "sensitivity": 0.1889,
        "specificity": 0.95
      },
      "ece": 0.0051,
      "reliability": [
        {
          "bin": "[0.0,0.1)",
          "n": 10356,
          "empirical": 0.061,
          "predicted": 0.066
        },
        {
          "bin": "[0.1,0.2)",
          "n": 8374,
          "empirical": 0.137,
          "predicted": 0.136
        },
        {
          "bin": "[0.2,0.3)",
          "n": 1173,
          "empirical": 0.263,
          "predicted": 0.234
        },
        {
          "bin": "[0.3,0.4)",
          "n": 321,
          "empirical": 0.352,
          "predicted": 0.343
        },
        {
          "bin": "[0.4,0.5)",
          "n": 94,
          "empirical": 0.5,
          "predicted": 0.436
        },
        {
          "bin": "[0.5,0.6)",
          "n": 27,
          "empirical": 0.667,
          "predicted": 0.534
        },
        {
          "bin": "[0.6,0.7)",
          "n": 4,
          "empirical": 0.5,
          "predicted": 0.623
        },
        {
          "bin": "[0.7,0.8)",
          "n": 4,
          "empirical": 0.25,
          "predicted": 0.731
        },
        {
          "bin": "[0.8,0.9)",
          "n": 1,
          "empirical": 1.0,
          "predicted": 0.825
        }
      ],
      "risk_ratio_deciles": {
        "median": 0.88,
        "p90": 1.7,
        "max": 7.39,
        "tier_low": 14883,
        "tier_moderate": 4398,
        "tier_high": 931,
        "tier_very_high": 142
      },
      "positive_or_high_class_prevalence": 0.1116
    },
    "hf_mortality": {
      "roc_auc": 0.8819,
      "pr_auc": 0.7814,
      "brier": 0.1277,
      "sens_at_95spec": {
        "threshold": 0.4096,
        "sensitivity": 0.6316,
        "specificity": 0.95
      },
      "ece": 0.0978,
      "reliability": [
        {
          "bin": "[0.0,0.1)",
          "n": 31,
          "empirical": 0.065,
          "predicted": 0.077
        },
        {
          "bin": "[0.1,0.2)",
          "n": 7,
          "empirical": 0.429,
          "predicted": 0.126
        },
        {
          "bin": "[0.2,0.3)",
          "n": 4,
          "empirical": 0.0,
          "predicted": 0.228
        },
        {
          "bin": "[0.3,0.4)",
          "n": 3,
          "empirical": 0.667,
          "predicted": 0.308
        },
        {
          "bin": "[0.4,0.5)",
          "n": 2,
          "empirical": 0.5,
          "predicted": 0.436
        },
        {
          "bin": "[0.5,0.6)",
          "n": 2,
          "empirical": 0.5,
          "predicted": 0.524
        },
        {
          "bin": "[0.6,0.7)",
          "n": 2,
          "empirical": 1.0,
          "predicted": 0.632
        },
        {
          "bin": "[0.7,0.8)",
          "n": 1,
          "empirical": 1.0,
          "predicted": 0.748
        },
        {
          "bin": "[0.8,0.9)",
          "n": 8,
          "empirical": 0.875,
          "predicted": 0.847
        }
      ],
      "risk_ratio_deciles": {
        "median": 0.31,
        "p90": 2.631,
        "max": 2.75,
        "tier_low": 45,
        "tier_moderate": 5,
        "tier_high": 10,
        "tier_very_high": 0
      },
      "positive_or_high_class_prevalence": 0.3167
    },
    "diabetes_symptoms": {
      "roc_auc": 1.0,
      "pr_auc": 1.0,
      "brier": 0.0122,
      "sens_at_95spec": {
        "threshold": 0.0912,
        "sensitivity": 1.0,
        "specificity": 0.95
      },
      "ece": 0.0569,
      "reliability": [
        {
          "bin": "[0.0,0.1)",
          "n": 38,
          "empirical": 0.0,
          "predicted": 0.049
        },
        {
          "bin": "[0.1,0.2)",
          "n": 1,
          "empirical": 0.0,
          "predicted": 0.121
        },
        {
          "bin": "[0.4,0.5)",
          "n": 1,
          "empirical": 0.0,
          "predicted": 0.415
        },
        {
          "bin": "[0.5,0.6)",
          "n": 3,
          "empirical": 1.0,
          "predicted": 0.522
        },
        {
          "bin": "[0.6,0.7)",
          "n": 1,
          "empirical": 1.0,
          "predicted": 0.633
        },
        {
          "bin": "[0.7,0.8)",
          "n": 1,
          "empirical": 1.0,
          "predicted": 0.716
        },
        {
          "bin": "[0.8,0.9)",
          "n": 3,
          "empirical": 1.0,
          "predicted": 0.875
        },
        {
          "bin": "[0.9,1.0)",
          "n": 56,
          "empirical": 1.0,
          "predicted": 0.982
        }
      ],
      "risk_ratio_deciles": {
        "median": 1.57,
        "p90": 1.61,
        "max": 1.61,
        "tier_low": 45,
        "tier_moderate": 59,
        "tier_high": 0,
        "tier_very_high": 0
      },
      "positive_or_high_class_prevalence": 0.6154
    },
    "liver_hcv": {
      "roc_auc_ovr": 1.0,
      "log_loss": 0.0274,
      "confusion": [
        [
          107,
          0,
          0
        ],
        [
          0,
          6,
          0
        ],
        [
          0,
          0,
          5
        ]
      ],
      "positive_or_high_class_prevalence": 0.0424
    },
    "cad_spect": {
      "roc_auc": 0.9587,
      "pr_auc": 0.983,
      "brier": 0.0732,
      "sens_at_95spec": {
        "threshold": 0.5121,
        "sensitivity": 0.9216,
        "specificity": 0.95
      },
      "ece": 0.1024,
      "reliability": [
        {
          "bin": "[0.1,0.2)",
          "n": 5,
          "empirical": 0.0,
          "predicted": 0.145
        },
        {
          "bin": "[0.2,0.3)",
          "n": 8,
          "empirical": 0.125,
          "predicted": 0.261
        },
        {
          "bin": "[0.3,0.4)",
          "n": 6,
          "empirical": 0.333,
          "predicted": 0.342
        },
        {
          "bin": "[0.4,0.5)",
          "n": 3,
          "empirical": 0.333,
          "predicted": 0.429
        },
        {
          "bin": "[0.5,0.6)",
          "n": 1,
          "empirical": 1.0,
          "predicted": 0.577
        },
        {
          "bin": "[0.6,0.7)",
          "n": 2,
          "empirical": 1.0,
          "predicted": 0.623
        },
        {
          "bin": "[0.7,0.8)",
          "n": 4,
          "empirical": 1.0,
          "predicted": 0.748
        },
        {
          "bin": "[0.8,0.9)",
          "n": 13,
          "empirical": 0.923,
          "predicted": 0.855
        },
        {
          "bin": "[0.9,1.0)",
          "n": 28,
          "empirical": 1.0,
          "predicted": 0.93
        }
      ],
      "risk_ratio_deciles": {
        "median": 1.18,
        "p90": 1.3,
        "max": 1.31,
        "tier_low": 37,
        "tier_moderate": 33,
        "tier_high": 0,
        "tier_very_high": 0
      },
      "positive_or_high_class_prevalence": 0.7286
    }
  },
  "triage": {
    "confusion": [
      [
        2036,
        207,
        1,
        0,
        0
      ],
      [
        185,
        7016,
        106,
        1,
        0
      ],
      [
        104,
        192,
        4338,
        129,
        0
      ],
      [
        5,
        1,
        110,
        4395,
        31
      ],
      [
        0,
        1,
        2,
        122,
        1018
      ]
    ],
    "per_class_recall": {
      "T0": 0.9073,
      "T1": 0.96,
      "T2": 0.9108,
      "T3": 0.9676,
      "T4": 0.8906
    },
    "under_triage_T0_any_down": 0.0927,
    "under_triage_T0_to_T1_adjacent": 0.0922,
    "under_triage_T0_deep_to_T2plus": 0.0004,
    "under_triage_T0_T1_rate_beyond_adjacent": 0.0113,
    "over_triage_rate": 0.0012,
    "human_review_flag_rate": 0.0054,
    "blend_weights": [
      0.3333333333333333,
      0.0,
      0.5,
      0.16666666666666666
    ],
    "ece_T1": 0.0017,
    "reliability_T1": [
      {
        "bin": "[0.0,0.1)",
        "n": 12357,
        "empirical": 0.012,
        "predicted": 0.013
      },
      {
        "bin": "[0.1,0.2)",
        "n": 24,
        "empirical": 0.042,
        "predicted": 0.126
      },
      {
        "bin": "[0.2,0.3)",
        "n": 28,
        "empirical": 0.036,
        "predicted": 0.251
      },
      {
        "bin": "[0.3,0.4)",
        "n": 20,
        "empirical": 0.25,
        "predicted": 0.339
      },
      {
        "bin": "[0.4,0.5)",
        "n": 17,
        "empirical": 0.294,
        "predicted": 0.449
      },
      {
        "bin": "[0.5,0.6)",
        "n": 11,
        "empirical": 0.455,
        "predicted": 0.56
      },
      {
        "bin": "[0.6,0.7)",
        "n": 11,
        "empirical": 0.636,
        "predicted": 0.655
      },
      {
        "bin": "[0.7,0.8)",
        "n": 32,
        "empirical": 0.812,
        "predicted": 0.75
      },
      {
        "bin": "[0.8,0.9)",
        "n": 104,
        "empirical": 0.933,
        "predicted": 0.866
      },
      {
        "bin": "[0.9,1.0)",
        "n": 7396,
        "empirical": 0.948,
        "predicted": 0.947
      }
    ]
  }
}

```


## 5. Risk-ratio interpretation
`risk_ratio = p(patient) ÷ p(baseline)` on the holdout prevalence. Tiers: <1.2 low · 1.2–2 moderate · 2–3.5 high · ≥3.5 very high. Ratios are *relative risk flags for prioritization*, never diagnoses.


## 6. Safety & human oversight (spec §21, §28–30)
- Deterministic red-flag rules (shock, hypoxemia, GCS≤8, qSOFA≥2, FAST-stroke, ACS, hemorrhage…) **always escalate** and override model output upward.
- Safety floors cap under-triage: P(T0)+P(T1) ≥ 0.30 → at most T1; ≥ 0.14 → at most T2.
- **T0 under-triage is reported at two granularities:** any down-triage (incl. T0→T1, where the patient is still seen as 'very urgent') and *deep* under-triage (T0→T2 or later) — the safety-critical error. Golden set: deep T0 under-triage = **0%** (see §4 metrics).
- Model/rules disagreement or uncertain inputs → `requires_human_review: true` and confidence is halved.
- The system **never** outputs 'no problem' or a diagnosis; T4 is explicitly follow-up, not reassurance of health.


## 7. Limitations
- Real ED-triage corpora (MIMIC-IV-ED/NHAMCS) are credentialed; the triage cohort is synthetic family-E with protocol labels.
- Population skew in legacy UCI sets (Pima adult women; ILPD Indian cohort; 130-US-hospitals 1999–2008) — ratios are baseline-relative within these distributions.
- `time` in hf_mortality is the follow-up duration (published benchmark convention).
- Requires human review in deployment; intended under qualified clinical supervision and local regulatory pathways (spec §21).


## 8. Provenance & reproducibility
Deterministic SEED=42; `src/run_all.py` reproduces end-to-end; per-dataset license/provenance in `data/raw/MANIFEST.json`. Kaggle credentials supplied by the user lacked an API key — collection used license-permissive UCI/OpenML instead.

---

## v3 — Bake-off winner (2026-09-25, Part M)

**Deployed web engine: XGBoost + LightGBM soft blend (50/50 probability-space)**,
selected by open bake-off over XGBoost · LightGBM · ExtraTrees · RandomForest ·
sklearn MLP · TensorFlow Keras MLP · blends. Trained on 200,000 protocol-labelled
rows (180k/20k stratified). Browser distillation parity: 1.5e-05.

| model | macro-F1 | T0-deep | beyond | over | feasible |
|---|---|---|---|---|---|
| xgboost | 0.9124 | 0.0018 | 0.0135 | 0.0012 | yes |
| lightgbm | 0.9127 | 0.0014 | 0.0136 | 0.0019 | yes |
| extratrees | 0.8457 | 0.0018 | 0.0121 | 0.0360 | yes |
| randomforest | 0.9061 | 0.0009 | 0.0131 | 0.0026 | yes |
| mlp_sklearn | 0.9000 | 0.0041 | 0.0144 | 0.0021 | no (safety) |
| mlp_tensorflow | 0.9012 | 0.0023 | 0.0139 | 0.0030 | yes |
| **blend_xgb_lgbm** | **0.9129** | **0.0014** | 0.0134 | 0.0016 | **yes (winner)** |
| blend_xgb_lgbm_et | 0.9072 | 0.0009 | 0.0130 | 0.0014 | yes |

Honest holdout (20k, full safety layer): macro-F1 0.9129 · recalls T0 0.8932 / T1 0.9544 /
T2 0.8778 / T3 0.9473 / T4 0.8775 · T0-any-down 0.1068 · T0-deep 0.0014 ·
beyond-adjacent 0.0134 · over-triage 0.0016. Payload: 3,500 trees (1500 xgb d5 + 2000 lgbm),
per-tree comparator flags (xgb: f32 `<` · lgbm: f64 `<=`), per-class base margin.

Notebook: `notebooks/healthguard_triage_perfect.ipynb` (Kaggle-ready). Kaggle API could not
be used — uploaded kaggle.json contains a username only (no `key` field); UCI/OpenML
cohorts in `data/` are the real-data probes. Server stack (4-model, macro-F1 0.9388) unchanged.
