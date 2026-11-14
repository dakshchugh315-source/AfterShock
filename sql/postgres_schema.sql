-- AfterShock on PostgreSQL: full schema + AI layer (views, functions, triggers), loaded by scripts/load_postgres.py.
-- Same tables as the SQLite experiment database, written in PostgreSQL dialect with explicit keys and constraints.
-- Spatial: PostGIS when installed (geometry + GiST); otherwise the bundled earthdistance extension (GiST on ll_to_earth).

DROP SCHEMA IF EXISTS aftershock CASCADE;
CREATE SCHEMA aftershock;
SET search_path = aftershock, public;

-- ------------------------------------------------------------------ core tables
CREATE TABLE disasters (
    name           TEXT PRIMARY KEY,
    disaster_type  TEXT
);

CREATE TABLE images (
    image_id  TEXT PRIMARY KEY,
    disaster  TEXT NOT NULL REFERENCES disasters (name),
    split     TEXT NOT NULL CHECK (split IN ('train', 'val', 'test')),
    width     INT  CHECK (width > 0),
    height    INT  CHECK (height > 0)
);

CREATE TABLE buildings (
    uid            TEXT PRIMARY KEY,
    image_id       TEXT NOT NULL REFERENCES images (image_id),
    disaster       TEXT NOT NULL REFERENCES disasters (name),  -- denormalised on purpose (report 6.2)
    split          TEXT NOT NULL CHECK (split IN ('train', 'val', 'test')),
    label          TEXT CHECK (label IN ('no-damage', 'minor-damage', 'major-damage', 'destroyed')),
    label_idx      INT  CHECK (label_idx BETWEEN 0 AND 3),
    footprint_wkt  TEXT,
    lon            DOUBLE PRECISION CHECK (lon BETWEEN -180 AND 180),
    lat            DOUBLE PRECISION CHECK (lat BETWEEN -90 AND 90)
);
CREATE INDEX buildings_split_idx    ON buildings (split);
CREATE INDEX buildings_disaster_idx ON buildings (disaster, split);

-- image bytes kept apart from the metadata (vertical partitioning); 1:1 with buildings
CREATE TABLE crops (
    uid       TEXT PRIMARY KEY REFERENCES buildings (uid) ON DELETE CASCADE,
    pre_png   BYTEA NOT NULL,
    post_png  BYTEA NOT NULL
);

-- probabilistic output: one row per (building, model)
CREATE TABLE predictions (
    uid            TEXT NOT NULL REFERENCES buildings (uid) ON DELETE CASCADE,
    model_version  TEXT NOT NULL,
    pred_idx       INT  NOT NULL CHECK (pred_idx BETWEEN 0 AND 3),
    p_no           REAL NOT NULL CHECK (p_no BETWEEN 0 AND 1),
    p_minor        REAL NOT NULL CHECK (p_minor BETWEEN 0 AND 1),
    p_major        REAL NOT NULL CHECK (p_major BETWEEN 0 AND 1),
    p_destroyed    REAL NOT NULL CHECK (p_destroyed BETWEEN 0 AND 1),
    confidence     REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    uncertainty    REAL NOT NULL CHECK (uncertainty BETWEEN 0 AND 1),
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (uid, model_version),
    CHECK (abs(p_no + p_minor + p_major + p_destroyed - 1) < 0.01)    -- a probability distribution
);
CREATE INDEX predictions_unc_idx ON predictions (model_version, uncertainty);

CREATE TABLE facilities (
    id     SERIAL PRIMARY KEY,
    name   TEXT NOT NULL,
    ftype  TEXT NOT NULL,
    lon    DOUBLE PRECISION NOT NULL,
    lat    DOUBLE PRECISION NOT NULL
);

CREATE TABLE hotspots (
    model_version    TEXT NOT NULL,
    cluster_id       INT  NOT NULL,
    disaster         TEXT NOT NULL REFERENCES disasters (name),
    n_buildings      INT  NOT NULL CHECK (n_buildings > 0),
    lon              DOUBLE PRECISION NOT NULL,
    lat              DOUBLE PRECISION NOT NULL,
    mean_confidence  REAL CHECK (mean_confidence BETWEEN 0 AND 1),
    PRIMARY KEY (model_version, cluster_id)
);

-- ------------------------------------------------------------------ AI layer
CREATE TABLE model_runs (
    model_version  TEXT PRIMARY KEY,
    arch           TEXT NOT NULL,
    loss           TEXT,
    img_size       INT  CHECK (img_size > 0),
    val_macro_f1   REAL CHECK (val_macro_f1 BETWEEN 0 AND 1),
    test_macro_f1  REAL CHECK (test_macro_f1 BETWEEN 0 AND 1),
    test_accuracy  REAL CHECK (test_accuracy BETWEEN 0 AND 1),
    ckpt_path      TEXT,
    registered_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE review_queue (
    uid            TEXT NOT NULL,
    model_version  TEXT NOT NULL,
    reason         TEXT NOT NULL,
    uncertainty    REAL,
    status         TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'done')),
    queued_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (uid, model_version),
    FOREIGN KEY (uid, model_version) REFERENCES predictions (uid, model_version) ON DELETE CASCADE
);

CREATE TABLE human_reviews (
    review_id      SERIAL PRIMARY KEY,
    uid            TEXT NOT NULL REFERENCES buildings (uid),
    model_version  TEXT NOT NULL,
    reviewer       TEXT NOT NULL,
    label_idx      INT  NOT NULL CHECK (label_idx BETWEEN 0 AND 3),
    reviewed_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Trigger 1: every new prediction the model is unsure about queues itself for a person.
CREATE FUNCTION queue_uncertain() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.uncertainty >= 0.5 OR NEW.confidence < 0.6 THEN
        INSERT INTO review_queue (uid, model_version, reason, uncertainty)
        VALUES (NEW.uid, NEW.model_version,
                CASE WHEN NEW.uncertainty >= 0.5 THEN 'high uncertainty' ELSE 'low confidence' END,
                NEW.uncertainty)
        ON CONFLICT (uid, model_version) DO UPDATE
            SET reason = EXCLUDED.reason, uncertainty = EXCLUDED.uncertainty, status = 'pending', queued_at = now();
    END IF;
    RETURN NEW;
END $$;
CREATE TRIGGER trg_queue_uncertain AFTER INSERT ON predictions
    FOR EACH ROW EXECUTE FUNCTION queue_uncertain();

-- Trigger 2: an expert's answer closes the queue item.
CREATE FUNCTION review_done() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    UPDATE review_queue SET status = 'done' WHERE uid = NEW.uid AND model_version = NEW.model_version;
    RETURN NEW;
END $$;
CREATE TRIGGER trg_review_done AFTER INSERT ON human_reviews
    FOR EACH ROW EXECUTE FUNCTION review_done();

-- SQL helper functions on stored probabilities
CREATE FUNCTION damage_label(idx INT) RETURNS TEXT LANGUAGE sql IMMUTABLE AS $$
    SELECT (ARRAY['no-damage', 'minor-damage', 'major-damage', 'destroyed'])[idx + 1] $$;
CREATE FUNCTION entropy4(a REAL, b REAL, c REAL, d REAL) RETURNS DOUBLE PRECISION LANGUAGE sql IMMUTABLE AS $$
    SELECT -(  CASE WHEN a > 0 THEN a * ln(a) ELSE 0 END + CASE WHEN b > 0 THEN b * ln(b) ELSE 0 END
             + CASE WHEN c > 0 THEN c * ln(c) ELSE 0 END + CASE WHEN d > 0 THEN d * ln(d) ELSE 0 END) / ln(4) $$;
CREATE FUNCTION expected_severity(a REAL, b REAL, c REAL, d REAL) RETURNS DOUBLE PRECISION LANGUAGE sql IMMUTABLE AS $$
    SELECT (0 * a + 1 * b + 2 * c + 3 * d)::DOUBLE PRECISION $$;

-- Views
CREATE VIEW v_predictions AS
SELECT b.uid, b.disaster, b.split, b.label_idx AS true_idx, p.model_version, p.pred_idx,
       p.p_no, p.p_minor, p.p_major, p.p_destroyed, p.confidence, p.uncertainty,
       (p.pred_idx = b.label_idx) AS correct
FROM buildings b JOIN predictions p ON p.uid = b.uid;

CREATE VIEW v_disaster_damage AS
SELECT model_version, disaster, COUNT(*) AS n_buildings,
       COUNT(*) FILTER (WHERE pred_idx = 3)          AS hard_destroyed,
       ROUND(SUM(p_destroyed)::NUMERIC, 1)           AS expected_destroyed,
       COUNT(*) FILTER (WHERE true_idx = 3)          AS true_destroyed,
       ROUND(SUM(p_major + p_destroyed)::NUMERIC, 1) AS expected_severe,
       ROUND(AVG(uncertainty)::NUMERIC, 3)           AS mean_uncertainty,
       ROUND(AVG(correct::INT), 3)                   AS accuracy
FROM v_predictions WHERE split = 'test'
GROUP BY model_version, disaster;

CREATE VIEW v_calibration AS
SELECT model_version, LEAST(FLOOR(confidence * 10), 9) / 10.0 AS conf_bin, COUNT(*) AS n,
       ROUND(AVG(confidence)::NUMERIC, 3) AS mean_confidence, ROUND(AVG(correct::INT), 3) AS accuracy
FROM v_predictions WHERE split = 'test'
GROUP BY model_version, conf_bin;

CREATE VIEW v_uncertainty_rank AS
SELECT model_version, uid, uncertainty, correct,
       ROW_NUMBER() OVER (PARTITION BY model_version ORDER BY uncertainty) AS rnk,
       COUNT(*)     OVER (PARTITION BY model_version)                      AS total
FROM v_predictions WHERE split = 'test';

CREATE VIEW v_review_queue AS
SELECT q.uid, q.model_version, q.reason, ROUND(q.uncertainty::NUMERIC, 3) AS uncertainty, b.disaster, b.lon, b.lat
FROM review_queue q JOIN buildings b ON b.uid = q.uid
WHERE q.status = 'pending'
ORDER BY q.uncertainty DESC;

CREATE VIEW v_model_leaderboard AS
SELECT model_version, arch, loss, img_size, ROUND(val_macro_f1::NUMERIC, 3) AS val_macro_f1,
       ROUND(test_macro_f1::NUMERIC, 3) AS test_macro_f1,
       RANK() OVER (ORDER BY val_macro_f1 DESC) AS rank_by_val
FROM model_runs;

CREATE VIEW v_model_vs_human AS
SELECT h.model_version, COUNT(*) AS reviewed, ROUND(AVG((p.pred_idx = h.label_idx)::INT), 3) AS model_agrees_with_human
FROM human_reviews h JOIN predictions p ON p.uid = h.uid AND p.model_version = h.model_version
GROUP BY h.model_version;

-- ------------------------------------------------------------------ real-world check (CAL FIRE field inspections)
CREATE TABLE field_inspections (
    id          INT  PRIMARY KEY,
    source      TEXT NOT NULL DEFAULT 'CAL FIRE DINS',
    incident    TEXT NOT NULL,
    disaster    TEXT NOT NULL REFERENCES disasters (name),
    damage_raw  TEXT NOT NULL,
    damage_idx  INT  NOT NULL CHECK (damage_idx BETWEEN 0 AND 3),
    structure   TEXT,
    lon         DOUBLE PRECISION NOT NULL,
    lat         DOUBLE PRECISION NOT NULL
);
CREATE TABLE field_matches (
    uid       TEXT PRIMARY KEY REFERENCES buildings (uid),
    field_id  INT  NOT NULL REFERENCES field_inspections (id),
    dist_m    REAL NOT NULL CHECK (dist_m >= 0),
    radius_m  REAL NOT NULL
);
CREATE VIEW v_field_compare AS
SELECT b.uid, b.disaster, b.split, b.lon, b.lat, b.label_idx AS xbd_idx,
       f.damage_idx AS field_idx, f.damage_raw AS field_raw, m.dist_m
FROM buildings b
LEFT JOIN field_matches m     ON m.uid = b.uid
LEFT JOIN field_inspections f ON f.id = m.field_id
WHERE b.disaster IN (SELECT DISTINCT disaster FROM field_inspections);
CREATE VIEW v_field_vs_model AS
SELECT c.*, p.model_version, p.pred_idx, p.confidence, p.uncertainty, p.p_destroyed
FROM v_field_compare c JOIN predictions p ON p.uid = c.uid;
