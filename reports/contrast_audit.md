# Contrast audit v2 — HealthGuard (Part N)

FAIL: 24 · RISK (dark bg, no text set): 76 · unresolved-var skips: 2

## FAIL

| file | theme | selector | color | ratio | need |
|---|---|---|---|---|---|
| main.css | light | `.float-chip` | #10251e | **1.11** | 4.5 |
| main.css | light | `.nav-trigger:hover, .nav-family.open .nav-trigger` | #10251e | **1.19** | 4.5 |
| main.css | light | `.btn-ghost` | #10251e | **1.19** | 4.5 |
| main.css | light | `.flow-step` | #10251e | **1.19** | 4.5 |
| main.css | light | `.pipe-nodes li` | #10251e | **1.19** | 4.5 |
| main.css | light | `.queue-list li` | #10251e | **1.19** | 4.5 |
| main.css | light | `.chip` | #33453d | **1.32** | 4.5 |
| main.css | light | `.note` | #33453d | **1.32** | 4.5 |
| main.css | light | `.checks li` | #33453d | **1.32** | 4.5 |
| main.css | light | `.hero-art-cap` | #33453d | **1.46** | 4.5 |
| main.css | light | `.site-footer` | #33453d | **1.51** | 4.5 |
| main.css | light | `.skip-link` | #fffdf7 | **1.74** | 4.5 |
| main.css | dark | `.pill-ok, .pill-verified` | #087255 | **2.01** | 4.5 |
| main.css | dark | `.card-icon` | #087255 | **2.04** | 4.5 |
| main.css | light | `.spec-table thead th` | #087255 | **2.27** | 4.5 |
| main.css | dark | `.spec-table thead th` | #087255 | **2.27** | 4.5 |
| main.css | light | `.pill-t1` | #d97706 | **2.59** | 4.5 |
| main.css | dark | `.pill-t0, .pill-danger` | #d7574c | **3.04** | 4.5 |
| main.css | light | `.pill-t0, .pill-danger` | #d7574c | **3.17** | 4.5 |
| main.css | dark | `.pill-t1` | #d97706 | **3.46** | 4.5 |
| app.css | light | `.s-cancelled` | #77807a | **3.6** | 4.5 |
| app.css | dark | `.s-cancelled` | #77807a | **3.6** | 4.5 |
| app.css | light | `.pill-t0` | #fff | **3.92** | 4.5 |
| app.css | dark | `.pill-t0` | #fff | **3.92** | 4.5 |

## RISK

| file | theme | selector | bg |
|---|---|---|---|
| app.css | dark | `.an-block` | #0b352c |
| app.css | dark | `.auth-card` | #062d25) |
| app.css | dark | `.board-row` | #0b352c |
| app.css | dark | `.chat-wrap` | #0b352c |
| app.css | dark | `.ck:hover` | rgba(201, 162, 39, .08) |
| app.css | dark | `.kv` | #0b352c |
| app.css | dark | `.pill-t1` | #d7574c |
| app.css | dark | `.pill-t2` | #087255 |
| app.css | dark | `.pill-t3` | #0b8174 |
| app.css | dark | `.result-card` | #062d25) |
| app.css | dark | `.role-card` | #0b352c |
| app.css | dark | `.ticket` | color-mix(in srgb, #087255 7%, #0b352c) |
| app.css | dark | `.video-stage` | #0a3a2f |
| app.css | light | `.pill-t1` | #d7574c |
| app.css | light | `.pill-t2` | #087255 |
| app.css | light | `.pill-t3` | #0b8174 |
| app.css | light | `.video-stage` | #0a3a2f |
| main.css | dark | `.acc-item` | #0b352c |
| main.css | dark | `.bento-big` | color-mix(in srgb, var(--cell, var(--acc-1)) 7%, #0b352c) |
| main.css | dark | `.card` | #0b352c |
| main.css | dark | `.card::before` | radial-gradient(20rem 13rem at var(--gx, 50%) var(--gy, 50%) |
| main.css | dark | `.chip::before` | #087255 |
| main.css | dark | `.code-block` | #04241d |
| main.css | dark | `.conf` | #0b352c |
| main.css | dark | `.conf-fill` | linear-gradient(90deg, color-mix(in srgb, #087255 75%, #fff) |
| main.css | dark | `.conf-track` | rgba(238, 245, 240, 0.09) |
| main.css | dark | `.cursor-glow` | radial-gradient(circle, color-mix(in srgb, #087255 6%, trans |
| main.css | dark | `.drawer-scrim` | rgba(4, 30, 24, 0.5) |
| main.css | dark | `.drawer-sheet` | linear-gradient(180deg, #0a3a2f, #062d25) |
| main.css | dark | `.eyebrow::before` | #087255 |
| main.css | dark | `.float-chip .fc-dot` | #087255 |
| main.css | dark | `.flow-accent .flow-step:nth-child(-n+2)` | color-mix(in srgb, #f5b914 10%, transparent) |
| main.css | dark | `.hero-copy::before` | radial-gradient(closest-side, rgba(6, 45, 37, 0.55), rgba(6, |
| main.css | dark | `.hero-spot` | radial-gradient(36rem 20rem at var(--spot-x, 30%) var(--spot |
| main.css | dark | `.marquee` | #0b352c |
| main.css | dark | `.nav-card:hover` | rgba(238, 245, 240, 0.07) |
| main.css | dark | `.nav-panel` | #062d25 |
| main.css | dark | `.nav-trigger::after` | #087255 |
| main.css | dark | `.note-warn` | color-mix(in srgb, #f5b914 8%, transparent) |
| main.css | dark | `.orb-1` | radial-gradient(circle at 35% 30%, rgba(216, 239, 116, 0.18) |
| main.css | dark | `.orb-2` | radial-gradient(circle at 35% 30%, rgba(86, 201, 155, 0.16), |
| main.css | dark | `.orb-3` | radial-gradient(circle at 35% 30%, rgba(255, 253, 247, 0.22) |
| main.css | dark | `.orb::after` | radial-gradient(circle at 32% 28%, rgba(238, 245, 240, 0.3), |
| main.css | dark | `.pf-wash` | radial-gradient(circle at var(--px, 50%) var(--py, 50%), col |
| main.css | dark | `.pipe` | #0b352c |
| main.css | dark | `.pipe-dot` | #087255 |
| main.css | dark | `.queue` | #0b352c |
| main.css | dark | `.site-header` | rgba(6, 45, 37, 0.97) |
| main.css | dark | `.spec-table tbody th` | #0b352c |
| main.css | dark | `.ss-1` | radial-gradient(circle at 40% 40%, rgba(216, 239, 116, .5),  |
| main.css | dark | `.ss-2` | radial-gradient(circle at 50% 50%, rgba(86, 201, 155, .42),  |
| main.css | dark | `.ss-3` | radial-gradient(circle at 50% 50%, rgba(255, 253, 247, .85), |
| main.css | dark | `.ss-4` | radial-gradient(circle at 50% 50%, rgba(245, 185, 20, .28),  |
| main.css | dark | `.ss-streak` | linear-gradient(100deg, transparent, rgba(255, 255, 255, .3) |
| main.css | dark | `.stat` | #0b352c |
| main.css | dark | `.table-wrap` | #0b352c |
| main.css | dark | `.theme-toggle` | #0b352c |
| main.css | dark | `16%` | #0b352c |
| main.css | dark | `6%` | color-mix(in srgb, #087255 7%, transparent) |
| main.css | dark | `html[data-theme="dark"] .ss-3` | radial-gradient(circle at 50% 50%, rgba(223, 246, 232, .16), |
| main.css | dark | `html[data-theme="dark"] .ss-streak` | linear-gradient(100deg, transparent, rgba(223, 246, 232, .07 |
| main.css | dark | `img, .hero-art-frame, .video-stage` | #0b352c |
| main.css | light | `.burger span` | #10251e |
| main.css | light | `.chip::before` | #087255 |
| main.css | light | `.code-block` | #04241d |
| main.css | light | `.conf-fill` | linear-gradient(90deg, color-mix(in srgb, #087255 75%, #fff) |
| main.css | light | `.eyebrow::before` | #087255 |
| main.css | light | `.float-chip .fc-dot` | #087255 |
| main.css | light | `.marquee` | #0b352c |
| main.css | light | `.nav-trigger::after` | #087255 |
| main.css | light | `.pipe-dot` | #087255 |
| main.css | light | `.site-header` | rgba(6, 45, 37, 0.97) |
| main.css | light | `.spec-table tbody th` | #0b352c |
| main.css | light | `.theme-toggle` | #0b352c |
| main.css | light | `16%` | #0b352c |
| main.css | light | `html[data-theme="light"] .burger span` | #10251e |

## Skipped (unresolved var)

- `.fld-i` → var(--ink, #eef5f0)
- `.fld-i` → var(--ink, #eef5f0)
