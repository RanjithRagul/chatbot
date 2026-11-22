CREATE TABLE IF NOT EXISTS monthly_usage (
    month_utc TEXT PRIMARY KEY,
    calls INTEGER NOT NULL DEFAULT 0 CHECK (calls >= 0)
);
