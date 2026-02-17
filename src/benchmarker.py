"""
KNN Peer Benchmarker for NSF HERD institutions.

Why KNN instead of KMeans or resource parity (+-20%)?
- KMeans produces wildly uneven clusters on skewed funding data.
- Resource parity (+-20% of total R&D) fails for outliers. Johns Hopkins
  at $4.1B has zero institutions within 20% of its total R&D. Tiny schools
  at $200K have maybe one. KNN always returns exactly n_peers regardless
  of where the institution sits in the distribution.

The pipeline:
  1. Log-transform all funding columns (compresses the skew so $1M vs $2M
     gets similar weight as $1B vs $2B).
  2. StandardScaler to zero-mean, unit-variance.
  3. Fit a NearestNeighbors model in that space.
  4. For any institution, return the k closest peers.

Note on features: we include total_rd alongside the 6 funding source
columns. Since total_rd = sum of all sources, this intentionally
double-weights overall size. The effect is that KNN primarily matches
institutions of similar size, then uses funding mix as a secondary
differentiator. This is the behavior we want -- a $300M school should
never be compared to a $3B school just because their federal percentage
is similar.
"""

import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


# ---------------------------------------------------------------------------
# SQL: pulls one row per institution from the most recent survey year.
# Edit column/table names here if the HERD schema changes.
# ---------------------------------------------------------------------------
FEATURES_QUERY = """
SELECT
    inst_id,
    name,
    state,
    total_rd,
    federal,
    state_local,
    business,
    nonprofit,
    institutional,
    other_sources
FROM institutions
WHERE year = (SELECT MAX(year) FROM institutions)
ORDER BY name;
"""

# These columns get log-transformed and scaled before feeding into KNN.
NUMERIC_COLS = [
    "total_rd",
    "federal",
    "state_local",
    "business",
    "nonprofit",
    "institutional",
    "other_sources",
]


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------
def fetch_university_features(db_path: str) -> pd.DataFrame:
    """Load institution-level funding features from the HERD database.

    Returns one row per institution for the latest survey year.
    Drops rows where all numeric features are null (no useful signal).
    Fills remaining nulls with 0 (an institution with no business
    funding has 0 business funding, not unknown).
    """
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql(FEATURES_QUERY, conn)
    finally:
        conn.close()

    df = df.dropna(subset=NUMERIC_COLS, how="all")
    df[NUMERIC_COLS] = df[NUMERIC_COLS].fillna(0)

    return df


# ---------------------------------------------------------------------------
# Benchmarker
# ---------------------------------------------------------------------------
class AutoBenchmarker:
    """KNN-based peer finder for HERD institutions.

    Usage:
        bench = AutoBenchmarker(n_peers=10)
        bench.fit(df)
        peers = bench.get_peers("003594")  # returns list of institution names
        gaps  = bench.analyze_gap("003594")
    """

    def __init__(self, n_peers: int = 10):
        self.n_peers = n_peers

        # These are set by fit().
        self.scaler = None
        self.nn_model = None
        self._data = None       # internal DataFrame, not exposed directly
        self._scaled = None     # log-scaled + standardized feature matrix

    # ------------------------------------------------------------------
    # Public read-only access to the fitted data.
    # Returns a copy so callers can't accidentally mutate our state.
    # This matters because Streamlit's @st.cache_resource shares one
    # benchmarker instance across all user sessions.
    # ------------------------------------------------------------------
    @property
    def data(self) -> pd.DataFrame:
        if self._data is None:
            return pd.DataFrame()
        return self._data.copy()

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "AutoBenchmarker":
        """Log-transform, normalize, and build the KNN index.

        Steps:
          1. np.log1p on every numeric column to compress skew.
          2. StandardScaler to zero-mean, unit-variance.
          3. Fit a NearestNeighbors model for fast lookups.
        """
        self._data = df.copy().reset_index(drop=True)

        log_features = np.log1p(self._data[NUMERIC_COLS])

        self.scaler = StandardScaler()
        self._scaled = self.scaler.fit_transform(log_features)

        # n_neighbors = n_peers + 1 because the query point is returned
        # as its own nearest neighbor (distance=0) and we strip it out.
        self.nn_model = NearestNeighbors(
            n_neighbors=min(self.n_peers + 1, len(self._data)),
            metric="euclidean",
            algorithm="auto",
        )
        self.nn_model.fit(self._scaled)

        return self

    # ------------------------------------------------------------------
    # get_peers
    # ------------------------------------------------------------------
    def get_peers(self, target_inst_id: str) -> list[str]:
        """Return the n_peers nearest institution names for a given inst_id.

        Results are sorted by proximity (closest first).
        """
        self._check_fitted()
        peer_indices = self._peer_indices(target_inst_id)
        return self._data.loc[peer_indices, "name"].tolist()

    def get_peer_inst_ids(self, target_inst_id: str) -> list[str]:
        """Return the n_peers nearest institution inst_ids.

        Used by the Research Portfolio and Federal Landscape tabs to JOIN
        peer inst_ids against field_expenditures and agency_funding tables.
        Same peers as get_peers(), just returning IDs instead of names.
        """
        self._check_fitted()
        peer_indices = self._peer_indices(target_inst_id)
        return self._data.loc[peer_indices, "inst_id"].tolist()

    # ------------------------------------------------------------------
    # analyze_gap
    # ------------------------------------------------------------------
    def analyze_gap(self, target_inst_id: str) -> list[dict]:
        """Compare a target institution against its peer-group average.

        Returns a list of dicts, one per funding metric:
          {"metric": "federal", "my_val": 500000, "peer_avg": 450000, "gap": 50000}
        """
        self._check_fitted()
        target_row = self._find_target(target_inst_id)
        peer_indices = self._peer_indices(target_inst_id)
        peer_avg = self._data.loc[peer_indices, NUMERIC_COLS].mean()

        gaps = []
        for col in NUMERIC_COLS:
            my_val = float(target_row[col].values[0])
            avg_val = float(peer_avg[col])
            gaps.append({
                "metric": col,
                "my_val": round(my_val, 2),
                "peer_avg": round(avg_val, 2),
                "gap": round(my_val - avg_val, 2),
            })

        return gaps

    # ------------------------------------------------------------------
    # get_peer_trend
    # ------------------------------------------------------------------
    def get_peer_trend(
        self,
        target_inst_id: str,
        db_path: str,
        start_year: int = 2019,
        end_year: int = 2024,
    ) -> tuple[pd.DataFrame, dict]:
        """Historical R&D trend for the target and its KNN peers.

        IMPORTANT: This joins on inst_id, not name. Many institutions
        have changed names over the years (e.g. "University of Arizona"
        became "The University of Arizona"). Joining on name would silently
        drop historical rows for those institutions.

        Returns:
          trend_df: Columns [name, year, total_rd, is_target]
          stats: {target_cagr, peer_avg_cagr, growth_rank, total_in_group}
        """
        self._check_fitted()
        target_row = self._find_target(target_inst_id)
        target_name = target_row["name"].values[0]

        # Get inst_ids for all peers (not names -- names change over time).
        peer_indices = self._peer_indices(target_inst_id)
        peer_inst_ids = self._data.loc[peer_indices, "inst_id"].tolist()
        all_inst_ids = [target_inst_id] + peer_inst_ids

        # Pull historical data by inst_id.
        # We use the most recent name for display so the chart labels
        # are consistent even if an institution changed names mid-series.
        placeholders = ",".join(["?"] * len(all_inst_ids))
        sql = f"""
            SELECT
                i.inst_id,
                i.year,
                i.total_rd,
                COALESCE(latest.name, i.name) as name
            FROM institutions i
            LEFT JOIN (
                SELECT inst_id, name
                FROM institutions
                WHERE year = (SELECT MAX(year) FROM institutions)
            ) latest ON i.inst_id = latest.inst_id
            WHERE i.inst_id IN ({placeholders})
              AND i.year BETWEEN ? AND ?
            ORDER BY i.inst_id, i.year
        """
        conn = sqlite3.connect(db_path)
        try:
            trend_df = pd.read_sql(
                sql, conn,
                params=all_inst_ids + [start_year, end_year]
            )
        finally:
            conn.close()

        trend_df["is_target"] = trend_df["inst_id"] == target_inst_id

        # Compute CAGR for each institution.
        # Use the actual years available, not the requested window,
        # so institutions that entered the dataset mid-window get
        # an accurate annualized growth rate instead of a diluted one.
        cagrs = {}
        for inst_id in all_inst_ids:
            inst = trend_df[trend_df["inst_id"] == inst_id].sort_values("year")
            if len(inst) < 2:
                continue

            first_row = inst.iloc[0]
            last_row = inst.iloc[-1]
            s = float(first_row["total_rd"])
            e = float(last_row["total_rd"])
            actual_years = int(last_row["year"]) - int(first_row["year"])

            if s > 0 and actual_years > 0:
                display_name = inst["name"].iloc[0]
                cagrs[display_name] = round(
                    ((e / s) ** (1 / actual_years) - 1) * 100, 1
                )

        target_cagr = cagrs.get(target_name, 0.0)
        peer_cagrs = [v for k, v in cagrs.items() if k != target_name]
        peer_avg_cagr = (
            round(sum(peer_cagrs) / len(peer_cagrs), 1) if peer_cagrs else 0.0
        )
        growth_rank = sum(1 for c in peer_cagrs if c > target_cagr) + 1

        stats = {
            "target_cagr": target_cagr,
            "peer_avg_cagr": peer_avg_cagr,
            "growth_rank": growth_rank,
            "total_in_group": len(cagrs),
        }

        return trend_df, stats

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _check_fitted(self) -> None:
        if self.nn_model is None or self._data is None:
            raise RuntimeError(
                "AutoBenchmarker has not been fitted yet. Call fit() first."
            )

    def _find_target(self, target_inst_id: str) -> pd.DataFrame:
        """Look up the target row by inst_id. Raises KeyError if missing."""
        target_row = self._data[self._data["inst_id"] == target_inst_id]
        if target_row.empty:
            raise KeyError(
                f"Institution '{target_inst_id}' not found in fitted data. "
                f"Available (first 10): {self._data['inst_id'].head(10).tolist()}"
            )
        return target_row

    def _peer_indices(self, target_inst_id: str) -> list[int]:
        """Return DataFrame indices of the n_peers nearest neighbors."""
        target_row = self._find_target(target_inst_id)
        target_idx = target_row.index[0]

        distances, indices = self.nn_model.kneighbors(
            self._scaled[target_idx].reshape(1, -1)
        )

        # The target itself appears in results (distance=0). Remove it.
        neighbor_indices = [int(i) for i in indices[0] if i != target_idx]
        return neighbor_indices[: self.n_peers]
