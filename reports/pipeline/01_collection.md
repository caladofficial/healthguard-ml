# Stage 1 — collection

Raw corpus: **301,317 rows** (target 300,000 → MET)

| source | status | rows in | admitted | licence | notes |
|---|---|---|---|---|---|
| synth_site_a | admitted | 100,000 | 100,000 | self-generated synthetic; no | deterministic ESI/ATS-shaped protocol labellin |
| synth_site_b | admitted | 100,000 | 100,000 | self-generated synthetic; no | deterministic ESI/ATS-shaped protocol labellin |
| synth_site_c | admitted | 100,000 | 100,000 | self-generated synthetic; no | deterministic ESI/ATS-shaped protocol labellin |
| uci_863 | admitted | 1,014 | 1,014 | CC BY 4.0 | harmonised 7/51 triage columns; rest missing b |
| uci_45 | admitted | 303 | 303 | CC BY 4.0 | harmonised 4/51 triage columns; rest missing b |
| uci_296 | rejected | 101,766 | 0 | n/a - not used | no vital signs - demographics/utilisation only |
| uci_519 | rejected | 299 | 0 | n/a - not used | no vital signs - labs + echocardiography only |
| uci_529 | rejected | 520 | 0 | n/a - not used | no vital signs - symptom questionnaire |
| uci_571 | rejected | 615 | 0 | n/a - not used | no vital signs - laboratory panel |
| uci_827 | rejected | 110,341 | 0 | n/a - not used | fetched live: only age/sex/episode/outcome, no |
| mimic_iv_ed | failed | 0 | 0 | unknown - not ingested | REAL ED triage vitals + acuity - but full extr |
| mimic_iv_ed_demo | failed | 0 | 0 | unknown - not ingested | open-access demo - file paths returned 404 in  |
| nhamcs | failed | 0 | 0 | unknown - not ingested | reachable, but fixed-width layout needs a mult |
| kaggle | failed | 0 | 0 | unknown - not ingested | kaggle.json holds a username-only key - API au |

Class balance: {"T0": 33370, "T1": 110950, "T2": 71535, "T3": 68488, "T4": 16974}

Label provenance: {"protocol": 300000, "rule_from_partial_vitals": 1317}
