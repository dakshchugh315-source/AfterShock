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
