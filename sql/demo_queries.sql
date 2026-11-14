-- AfterShock: live database demo for the viva (PostgreSQL 18, database "aftershock", schema "aftershock").
-- Open pgAdmin -> aftershock -> Query Tool, paste this file, select ONE block and press F5 (Execute selection).
SET search_path = aftershock, public;

-- 1. What is in the database: every table with its row count
SELECT relname AS table_name, n_live_tup AS approx_rows
FROM pg_stat_user_tables WHERE schemaname = 'aftershock' ORDER BY relname;

-- 2. Keys and constraints, straight from the system catalog
SELECT conrelid::regclass AS table_name, conname AS constraint_name,
       CASE contype WHEN 'p' THEN 'PRIMARY KEY' WHEN 'f' THEN 'FOREIGN KEY' WHEN 'c' THEN 'CHECK' WHEN 'u' THEN 'UNIQUE' END AS type,
       pg_get_constraintdef(oid) AS definition
FROM pg_constraint WHERE connamespace = 'aftershock'::regnamespace ORDER BY 1, 3;

-- 3. Constraints at work: these INSERTs are REJECTED by the database (run them one at a time)
INSERT INTO predictions (uid, model_version, pred_idx, p_no, p_minor, p_major, p_destroyed, confidence, uncertainty)
VALUES ('no-such-building', 'demo', 0, 1, 0, 0, 0, 1, 0);                  -- foreign key violation
INSERT INTO human_reviews (uid, model_version, reviewer, label_idx)
SELECT uid, 'demo', 'teacher', 7 FROM buildings LIMIT 1;                    -- CHECK label_idx BETWEEN 0 AND 3

-- 4. Join: buildings + their probabilistic prediction (first 10 test buildings)
SELECT b.uid, b.disaster, damage_label(b.label_idx) AS truth, p.model_version,
       damage_label(p.pred_idx) AS predicted, ROUND(p.confidence::NUMERIC, 2) AS confidence,
       ROUND(p.uncertainty::NUMERIC, 2) AS uncertainty
FROM buildings b JOIN predictions p ON p.uid = b.uid
WHERE b.split = 'test'
  AND p.model_version = (SELECT model_version FROM v_model_leaderboard WHERE rank_by_val = 1)  -- final model only
ORDER BY b.uid LIMIT 10;

-- 5. Probabilistic DB: threshold query - buildings the model is >= 80% sure are destroyed, and how many really are
SELECT model_version, COUNT(*) AS called_destroyed, COUNT(*) FILTER (WHERE true_idx = 3) AS really_destroyed,
       ROUND(AVG((true_idx = 3)::INT), 3) AS precision
FROM v_predictions WHERE split = 'test' AND p_destroyed >= 0.8 GROUP BY model_version;

-- 6. Probabilistic DB: expected count (sum of probabilities) next to the hard count, per disaster
SELECT * FROM v_disaster_damage ORDER BY model_version, disaster;

-- 7. Window functions: accuracy if the system answers only its surest X% (ROW_NUMBER over uncertainty)
SELECT model_version, cov AS coverage,
       ROUND(AVG(correct::INT) FILTER (WHERE rnk <= cov * total), 3) AS accuracy
FROM v_uncertainty_rank CROSS JOIN (VALUES (0.5), (0.8), (1.0)) AS c(cov)
GROUP BY model_version, cov ORDER BY model_version, cov;

-- 8. Model leaderboard: RANK() over validation macro-F1 (selection is by validation, never by test)
SELECT * FROM v_model_leaderboard;

-- 9. Index vs full scan: the same query, with and without the B-tree index
EXPLAIN ANALYZE SELECT COUNT(*) FROM buildings WHERE split = 'test';
SET enable_indexscan = off; SET enable_bitmapscan = off; SET enable_indexonlyscan = off;
EXPLAIN ANALYZE SELECT COUNT(*) FROM buildings WHERE split = 'test';
RESET enable_indexscan; RESET enable_bitmapscan; RESET enable_indexonlyscan;

-- 10. Trigger demo (in a transaction, rolled back so the data stays as it was)
BEGIN;
  SELECT COUNT(*) AS pending_before FROM review_queue WHERE status = 'pending';
  -- a new, very uncertain prediction for one building: trg_queue_uncertain queues it by itself
  INSERT INTO predictions (uid, model_version, pred_idx, p_no, p_minor, p_major, p_destroyed, confidence, uncertainty)
  SELECT uid, 'live_demo', 2, 0.25, 0.25, 0.26, 0.24, 0.26, 0.99 FROM buildings WHERE split = 'test' LIMIT 1;
  SELECT uid, model_version, reason, status FROM review_queue WHERE model_version = 'live_demo';
  -- an expert answers: trg_review_done closes the item
  INSERT INTO human_reviews (uid, model_version, reviewer, label_idx)
  SELECT uid, 'live_demo', 'teacher', 2 FROM review_queue WHERE model_version = 'live_demo';
  SELECT uid, model_version, status FROM review_queue WHERE model_version = 'live_demo';
ROLLBACK;

-- 11. Spatial query (earthdistance + GiST index): destroyed predictions within 500 m of a building
SELECT b2.uid, damage_label(p.pred_idx) AS predicted, ROUND(p.confidence::NUMERIC, 2) AS confidence,
       ROUND(earth_distance(ll_to_earth(b1.lat, b1.lon), ll_to_earth(b2.lat, b2.lon))::NUMERIC) AS metres
FROM (SELECT lat, lon FROM buildings WHERE disaster = 'santa-rosa-wildfire' AND split = 'test' LIMIT 1) b1
JOIN buildings b2 ON earth_box(ll_to_earth(b1.lat, b1.lon), 500) @> ll_to_earth(b2.lat, b2.lon)
                 AND earth_distance(ll_to_earth(b1.lat, b1.lon), ll_to_earth(b2.lat, b2.lon)) <= 500
JOIN predictions p ON p.uid = b2.uid
WHERE p.pred_idx = 3
  AND p.model_version = (SELECT model_version FROM v_model_leaderboard WHERE rank_by_val = 1)
ORDER BY metres LIMIT 20;
-- (with PostGIS installed the same filter is: ST_DWithin(b2.geom::geography, b1.geom::geography, 500))

-- 12. Real-world check: nearest CAL FIRE inspection for each building, in pure SQL (LATERAL + ORDER BY distance)
SELECT b.uid, damage_label(b.label_idx) AS xbd_label, f.damage_raw AS inspectors_recorded,
       ROUND(earth_distance(ll_to_earth(b.lat, b.lon), ll_to_earth(f.lat, f.lon))::NUMERIC, 1) AS metres
FROM (SELECT * FROM buildings WHERE disaster = 'santa-rosa-wildfire' AND split = 'test' ORDER BY uid LIMIT 15) b
CROSS JOIN LATERAL (
    SELECT fi.* FROM field_inspections fi
    WHERE earth_box(ll_to_earth(b.lat, b.lon), 12) @> ll_to_earth(fi.lat, fi.lon)
    ORDER BY earth_distance(ll_to_earth(b.lat, b.lon), ll_to_earth(fi.lat, fi.lon)) LIMIT 1) f;

-- 13. Model vs field inspectors, destroyed vs not (same numbers as the dashboard's Real-world tab)
SELECT model_version, COUNT(*) AS buildings,
       ROUND(AVG(((pred_idx = 3) = (field_idx = 3))::INT), 3) AS model_agrees,
       ROUND(AVG(((xbd_idx = 3) = (field_idx = 3))::INT), 3)  AS xbd_labels_agree
FROM v_field_vs_model WHERE field_idx IS NOT NULL AND split = 'test' GROUP BY model_version;
