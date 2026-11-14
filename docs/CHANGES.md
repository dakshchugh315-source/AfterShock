# Dashboard changes — 2026-11-12

## 2026-11-14 (evening): UI upgrade on Daksh's laptop

31. **Hero banner**: pure-CSS animated header (gradient, rotating 3D globe with an orbiting satellite,
    course-topic pills). No JavaScript added, so nothing extra runs in the browser.
32. **3D damage map** (default view in Map & hotspots; the 2D satellite map is still one click away):
    pydeck `ColumnLayer`, one column per building, height and colour = predicted damage (scaled by
    confidence); DBSCAN hotspots as violet discs. The view opens on the biggest hotspot, or on the
    ~1 km grid cell with the most major/destroyed predictions (the plain mean can land on an empty field).
33. **3D probability space** (Probabilistic DB tab): each test building plotted by its stored
    p_minor / p_major / p_destroyed, marker size = MC-Dropout uncertainty. Read straight from `predictions`.
34. Glass KPI tiles tilt in 3D on hover; tab content fades in; gradient tab underline;
    `prefers-reduced-motion` turns all animation off. Plotly legends no longer overlap titles.
    Per-disaster table shows "3,403" instead of "3403.000000".
36. **2D satellite view marks the damage on the roofs**: every building's real footprint polygon
    (`buildings.footprint_wkt`, lon/lat) is outlined and filled in its predicted-damage colour
    (one GeoJSON layer, hover shows class / confidence / uncertainty). Both map views now open on the
    biggest hotspot or the most-damaged ~1 km cell, instead of the plain mean (which landed on a school
    with no labelled buildings). The "no hotspots" message no longer suggests the disaster already selected.
37. **Dark "mission control" theme**: `.streamlit/config.toml` (dark base, orange/violet accents, Inter +
    Space Grotesk + JetBrains Mono fonts, minimal toolbar without the Deploy button). CSS: grid + glow
    background, branded sidebar (sections "Model & area", "Filters"), tabs as a pill bar with shorter
    labels, glass panels for metrics/tables/charts/alerts/code, gradient buttons. Plotly text/grid colours
    tuned for dark.
38. **Bug fix: review-queue confidence showed 0%/1%.** Confidence is stored as a probability (0-1) but the
    column format `%.0f%%` printed the raw number, so 0.28 became "0%". It is now converted to a percentage
    and drawn as a bar. Only the display was wrong; stored values and metrics were always correct.
39. Grad-CAM tab: queue size + share metrics; a **verdict card** for the selected building (predicted class,
    confidence, uncertainty, match/mismatch with the xBD label, "human review" vs "model decides") and a bar
    chart of its four stored probabilities. Header: one-line question + pipeline flow + live COUNT(*) stats
    (buildings, predictions, pending review_queue rows, hotspots).
40. **3D city view** (`app/city3d.py`, replaces the pydeck columns): every building's real footprint polygon is
    extruded on Esri satellite imagery with deck.gl 8.9 (lit, shaded 3D blocks; height + colour = predicted
    damage). Buildings rise from the ground on load, the camera orbits until the user touches the map,
    destroyed buildings pulse red and DBSCAN hotspots are violet rings. HUD: legend, live counts and
    buttons (orbit, replay, top view, tilt). `prefers-reduced-motion` disables the animation. The map tab is
    now full width. Data drawn = the same `predictions` rows; nothing about the model changes.
41. **Colour scheme**: UI accents are now cool (cyan #22d3ee / indigo #818cf8: tabs, sliders, buttons, links,
    hero) so warm colours only ever mean damage. Damage palette brightened for the dark background and kept
    identical in every view (2D map, 3D city, charts, verdict card): no damage #34d399, minor #facc15,
    major #fb923c, destroyed #f43f5e.
42. Verdict card no longer headlines an unsure arg-max as an answer: above the review threshold it says
    "Unsure · leaning <top class> X% vs <second class> Y%" in neutral grey, with a one-line reason.
43. **Fullscreen tables fixed.** The tab fade-in kept a filling CSS animation on the tab panel, which creates a
    stacking context, so a table opened fullscreen was trapped under the sidebar with its first column hidden.
    Animations are now opacity-only with no fill-mode, and tables keep Streamlit's own styling. Schema row
    counts use thousands separators.
44. **Accuracy without retraining** (`scripts/improve_model.py`): test-time augmentation (flips/rotation, same
    transform on pre and post), ensembling, and per-class log-probability offsets fitted on VALIDATION to
    maximise macro-F1. Every candidate is chosen on validation; test is evaluated once. Result stored as a new
    model_version with runs/<name>/metrics.json, so the dashboard picks it automatically if its validation
    macro-F1 is best. Pilot DB (64 px pilot models): best single test macro-F1 0.616 -> 0.637, accuracy
    0.621 -> 0.654 (`runs/pilot_ensemble_tuned/candidates.json`); destroyed recall drops 0.736 -> 0.657,
    reported as a trade-off. Grad-CAM and ai_predict() fall back to the best single network for an ensemble.
45. **Real-world check against CAL FIRE field inspections** (`scripts/fetch_dins.py`,
    `src/aftershock/fieldcheck.py`): 5,759 public DINS inspections of the 2017 Tubbs fire (damage + location
    only) loaded into `field_inspections` (CHECK on damage class, geo index). One SQL statement matches each
    building to its nearest inspection (bounding-box join on the index + distance + ROW_NUMBER() OVER
    (PARTITION BY building)) into `field_matches`; views `v_field_compare`, `v_field_vs_model`.
    Radius sweep (pilot DB): 8 m 2,847 matches / model 96.1% / xBD 98.1%; 12 m 3,174 / 96.0% / 98.0%;
    25 m 3,547 / 92.2% / 96.5%. Larger radii start matching a neighbour's inspection point, so 12 m is the
    default. On the same 371 test buildings, destroyed-vs-not accuracy: model 96.0%, xBD human labels 96.5%.
    Light damage (Affected/Minor 1-25%) is mostly invisible from above even for the annotators.
    Pre-2018 DINS lists only damaged structures: unmatched buildings are reported separately, never scored.
46. Dashboard tab **Real-world check**: KPIs and both confusion matrices on the SAME buildings, a comparison
    table, plain-language findings, twin synced 3D maps (model | inspectors, one camera, "highlight
    differences"), a building-by-building table (disagreements or all) and the SQL behind it.
47. Per-disaster and hotspot tables sized to their rows (no empty filler rows); accuracy shown as bars.
48. **Teacher-style audit fixes.** (a) The disaster summary block belonged to no tab, so it showed under EVERY tab:
    now Overview only. (b) Header numbers added up rows of several models (15,760 "predictions" for 7,880 test
    buildings): now for the final model only, and the model is named. (c) The leaderboard's #1 (`post_baseline`) had
    no stored predictions, so the site's "final" disagreed with the leaderboard: its 7,880 test predictions are now
    stored. (d) Grad-CAM came from a different network than the selected model without saying so: caption now names
    the network and explains why. (e) "128px" and "20 MC passes" were hard-coded: now read from the run files.
    (f) The probabilistic query opened with 0 results: it now centres on the densest confident severe predictions.
    (g) "Roughly half are false alarms" was hard-coded and contradicted 76% precision: computed now. (h) Pilot
    architecture group added to Overview; tables sized to their rows; shorter sidebar labels; "agreement" renamed
    "test accuracy" with its definition.
49. **Accuracy script, honest result on the strongest pilot model.** With post_baseline included, the candidate chosen
    on validation is post_baseline + TTA + bias: val 0.795 (vs 0.767) but test 0.6656 vs 0.6658, accuracy 0.688 vs
    0.681. The validation gain came from fitting the offsets on validation (mild overfitting) and does not carry to
    test. Stored as `post_baseline_tta_bias`; its `arch` now names the chosen candidate. Hotspots mined for it (45).
    Against CAL FIRE inspectors on the same 371 buildings: model 97.3%, xBD human labels 96.5% (destroyed vs not).
50. **PostgreSQL 18 copy of the database** (`sql/postgres_schema.sql`, `scripts/setup_postgres.ps1`,
    `scripts/load_postgres.py`): schema `aftershock` with PK/FK/CHECK/NOT NULL (incl. a CHECK that the 4
    probabilities sum to 1), PL/pgSQL triggers (review queue built by the trigger on load: 12,408 rows, identical
    to SQLite), SQL functions, all views, and an earthdistance GiST spatial index (PostGIS used automatically when
    installed). `sql/demo_queries.sql`: 13 viva queries, all verified (constraint violations rejected, index ~1 ms
    vs seq scan ~12 ms, trigger demo in a rolled-back transaction, LATERAL nearest-inspection join).
51. **Image quality + "What changed?" panel** (display only): larger, contrast-stretched context crops, and a
    brightness-normalised before/after difference heatmap with a change score inside the building vs its
    surroundings (destroyed test houses ~1.6-3x their surroundings, undamaged ~0.8-1.4x). The caption warns that
    misalignment between dates and shadows also show as change.
35. Verified headless (AppTest: no exceptions on any tab/button) and by screenshots in Edge on the
    pilot DB. 9/9 tests pass. No model/metric/SQL code was changed.

## 2026-11-14: SQL/DBMS pass before the demo (teacher is a DBMS/SQL teacher)

24. **Security:** removed the "Advanced (local database path)" sidebar
    expander and its text box. The database now comes **only** from the
    `AFTERSHOCK_DB` environment variable (set by `scripts/start_demo.ps1`),
    and the dashboard never displays the path. Without the variable it shows
    "Database not configured" and stops, without falling back to the old
    3-disaster `aftershock.db`.
25. **KPI cards no longer truncate** ("107,…", "sia…"). Hero tiles wrap and
    scale with the viewport (`clamp()`), and use short display names
    ("Siamese-FM"), with the run id in the label. `st.metric` labels are
    shorter and allowed to wrap.
26. **New tab "🗄️ Database & SQL"**, read-only (`mode=ro` connection, fixed
    parameterised presets, no free-text SQL). It contains:
    - the schema of all 10 tables with live `COUNT(*)` and PK/FK/CHECK/NOT NULL;
    - six presets, each showing its SQL, its result and the view definition:
      leaderboard (`RANK()`), expected vs hard destroyed, the P(destroyed) ≥ 0.8
      threshold query, the review queue, calibration, and selective accuracy
      (`ROW_NUMBER()`);
    - `EXPLAIN QUERY PLAN` + timing, index vs `NOT INDEXED`;
    - `ai_predict()` / `ai_confidence()` running the final model inside a
      `SELECT`;
    - the `trg_queue_uncertain` trigger text and `review_queue` counts.
27. The Probabilistic query panel now shows the exact SQL and parameters it
    ran. It also states that SQLite applies the radius filter in Python,
    where PostGIS would use `ST_DWithin`.
28. `src/aftershock/sqlai.py` and the new `test_sql_ai_layer` test were added
    from `AfterShock_sql_ai_update.zip` (9 tests pass). The layer is installed
    on `aftershock_full.db`.
    **`post_baseline` was removed from `model_runs`:** it was evaluated on
    the old 3-disaster DB, and its val 0.766 would wrongly top the
    leaderboard. Remove it the same way after running `--install` on another
    machine.
29. The sidebar model now defaults to the final model, the best by
    validation macro-F1 (marked "(final)").
30. Deprecated `use_container_width` on Plotly charts was replaced by
    `width="stretch"`. The whole app is verified headless with Streamlit
    `AppTest`: every tab, every preset and both buttons, with no exceptions.

## Final pass (2026-11-12)

21. **`palu-tsunami` showed as silently empty** in every tab if selected — its
    979 buildings all landed in the `train` split by the deterministic
    per-tile hash (none in `val`/`test`), so `predict` (which only scores the
    test split) never produced predictions for it. Not a bug, just how few
    tiles that disaster has — but it looked broken. The dashboard now shows
    an explicit, honest explanation instead of several vague "no data" states
    scattered across tabs.
22. **192px retrain (final architecture).** The first attempts were killed
    five times by the low-memory killer for background shells.
    The last attempt ran as a standalone console process
    (`runs/_train192_standalone.ps1`), 12 epochs, batch 64. Result:
    `siamese_fm_focal_192` got val 0.643 and test 0.616, against 0.647 and
    0.632 at 128px, so **the 128px model stays final**, selected by
    validation (report Section 10.5). post/siamese at 192px are bonus runs.
23. Public tunnel URL rotates each time `cloudflared` restarts (quick tunnels
    expire after some hours) — check the current link in chat rather than a
    bookmark before a demo.

Follow-up to a review of the live dashboard. Every number below comes from the
already-stored `runs/*/metrics.json` and the `predictions`/`hotspots` tables —
nothing here is invented, and the 192px retrain running in the background
(`post_focal_192` etc.) was not touched by any of this.

## A. Honesty / correctness fixes

1. **Model selection now uses VALIDATION macro-F1, not test.** Picking the
   "best" model by its test score is the model leaking into its own selection
   criterion — a real methodology bug an examiner would catch. Fixed in the
   Overview tab. For the current 128px trio this happens to pick the same
   winner (`siamese_fm_focal`, val 0.647 vs test-selected 0.632) — the result
   doesn't change, but the method now would hold up in viva.
2. **"+5%" was the relative lift; the point difference is +3.2.**
   `0.632 − 0.600 = 0.032` macro-F1, i.e. **+3.2 percentage points**, which is
   `+5.3%` *relative*. Both numbers are now shown, labelled separately.
3. **Surfaced the hidden recall trade-off.** `post_focal` has better recall on
   the `destroyed` class (0.716) than the overall-best `siamese_fm_focal`
   (0.661) — for rescue triage, missing a destroyed building is worse than a
   false alarm, so this is arguably the more important number and the
   dashboard previously hid it. Now called out explicitly as a warning box
   in the Overview tab whenever the recall-best model isn't the overall-best.
4. **Destroyed-class precision (~0.51 for the best model, ~0.41 for
   post-only) is now shown**, with a note that this is exactly the gap
   MC-Dropout uncertainty routing is meant to cover (send unsure "destroyed"
   calls to a human instead of trusting the label).
5. **"Agreement with labels" was computed over every split, including
   training buildings the model has memorised — inflating it.** Now filtered
   to `buildings.split = 'test'` only, and relabelled "Test-set agreement".
6. **95% confidence intervals added** for macro-F1, by bootstrapping each
   model's own recorded test confusion matrix (500 resamples from the
   confusion matrix's own distribution — not invented data, just the honest
   amount of "how much would this move on a re-run" a single training run
   per architecture can answer without retraining 3x).
7. **Per-disaster breakdown table added** (Map tab) — test-split accuracy per
   disaster for the selected model. Worth noting for the report: accuracy
   ranges from 46% (hurricane-matthew) to 99% (mexico-earthquake) — the
   headline macro-F1 hides a lot of per-disaster variance.
8. **Calibration / reliability diagram added** (Probabilistic DB tab) —
   observed accuracy vs. mean predicted confidence, bucketed. Shows where the
   model is over-confident, which is exactly what MC-Dropout's uncertainty
   score is supposed to catch downstream.

## B. Course topics made visible (previously present in the DB/pipeline but
   not surfaced in the UI)

9. **Probabilistic DB tab** — a live, parameterised version of the
   confidence + spatial query from the project README (`buildings_near`):
   pick a building, a radius, a damage class and a minimum confidence, see
   the matching buildings update live. Defaults to the biggest hotspot's
   neighbourhood so it isn't empty on first load.
10. **Hotspots are now listed in a table next to the map**, not just drawn as
    faint circles — the "37 hotspots found" summary line had nothing to point
    at on-screen before. Also fixed: the default disaster (`guatemala-volcano`,
    168 buildings / 7 damaged, no visible hotspots) is now
    `hurricane-harvey` if available — busier, has hotspots, a stronger first
    impression.
11. **Grad-CAM now computes on the fly** for any building that wasn't in the
    precomputed 30-most-uncertain batch, instead of dead-ending on a
    "run this command yourself" message in front of an examiner.

## C. Interface fixes

12. KPI tile value text used a fixed near-black colour — unreadable in
    Streamlit's dark theme. Now uses the tile's own accent colour, which has
    enough contrast in both themes.
13. Long model names (`siamese_fm_focal`) no longer wrap mid-word in the hero
    tiles (`white-space: nowrap` + ellipsis + full name on hover).
14. Chart title/legend overlap fixed (more top margin, legend pushed down).
15. The local database path is no longer shown directly in the sidebar — it's
    behind a collapsed "⚙️ Advanced" expander.
16. Map no longer hijacks the page scroll wheel (`scrollWheelZoom=False`);
    a caption explains you click the map first to zoom it.
17. The review queue showed full raw UUIDs and the confidence column was cut
    off. UIDs are now truncated with the full id visible on selection;
    confidence/uncertainty columns have explicit widths via
    `st.column_config`.
18. Before/after and Grad-CAM crops are upscaled further (384px → 480px for
    crops, 900px → 960px for the Grad-CAM panel) with the same
    Lanczos + unsharp-mask treatment as before.

## D. 192px results

19. The Overview tab now groups runs by resolution (`128px, 15 epochs` /
    `192px, 25 epochs`) and auto-detects whichever `_192` run folders exist —
    no code change needed once the background retrain finishes; the new
    results will just appear.

## E. Portability

20. A zip of the changed files is produced for moving this back to another
    machine — see the chat for the path.

## Not done (flagged, not silently skipped)

- **True multi-run variance** (training each architecture 3x) wasn't done —
  that needs real retraining, not just a dashboard fix, and would roughly
  triple the already-long training time. The bootstrap CI (item 6) is the
  honest substitute available without that.
- **The `trycloudflare.com` tunnel is a public, unauthenticated link** —
  anyone with it can open the dashboard while it's running. Stop
  `cloudflared.exe` when not actively demoing.
