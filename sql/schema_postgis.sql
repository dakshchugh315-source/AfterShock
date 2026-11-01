-- AfterShock schema for PostgreSQL + PostGIS (run automatically by aftershock.db.DB.init_schema)
CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS disasters (
    name           TEXT PRIMARY KEY,
    disaster_type  TEXT
);

CREATE TABLE IF NOT EXISTS images (
    image_id   TEXT PRIMARY KEY,               -- e.g. hurricane-michael_00000123
    disaster   TEXT REFERENCES disasters(name),
    split      TEXT CHECK (split IN ('train', 'val', 'test')),
    width      INT,
    height     INT
);

CREATE TABLE IF NOT EXISTS buildings (
    uid            TEXT PRIMARY KEY,
    image_id       TEXT REFERENCES images(image_id),
    disaster       TEXT,
    split          TEXT,
    label          TEXT,                       -- Joint Damage Scale class name
    label_idx      INT,
    footprint_wkt  TEXT,                       -- lon/lat polygon as WKT
    lon            DOUBLE PRECISION,           -- footprint centroid
    lat            DOUBLE PRECISION,
    footprint      geometry(Polygon, 4326),
    geom           geometry(Point, 4326) GENERATED ALWAYS AS (ST_SetSRID(ST_MakePoint(lon, lat), 4326)) STORED
);
CREATE INDEX IF NOT EXISTS buildings_geom_gix ON buildings USING GIST (geom);
CREATE INDEX IF NOT EXISTS buildings_split_idx ON buildings (split);

-- Pre/post image crops live in the database so training streams straight from it.
CREATE TABLE IF NOT EXISTS crops (
    uid       TEXT PRIMARY KEY REFERENCES buildings(uid),
    pre_png   BYTEA,
    post_png  BYTEA
);

-- Uncertain values: class probabilities and model uncertainty per building.
CREATE TABLE IF NOT EXISTS predictions (
    uid            TEXT REFERENCES buildings(uid),
    model_version  TEXT,
    pred_idx       INT,
    p_no           REAL,
    p_minor        REAL,
    p_major        REAL,
    p_destroyed    REAL,
    confidence     REAL,                       -- max mean probability over MC passes
    uncertainty    REAL,                       -- normalised predictive entropy, 0..1
    created_at     TIMESTAMP DEFAULT now(),
    PRIMARY KEY (uid, model_version)
);

CREATE TABLE IF NOT EXISTS facilities (
    id     SERIAL PRIMARY KEY,
    name   TEXT,
    ftype  TEXT,                               -- hospital, school, shelter ...
    lon    DOUBLE PRECISION,
    lat    DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS hotspots (
    model_version    TEXT,
    cluster_id       INT,
    disaster         TEXT,
    n_buildings      INT,
    lon              DOUBLE PRECISION,
    lat              DOUBLE PRECISION,
    mean_confidence  REAL,
    PRIMARY KEY (model_version, cluster_id)
);
