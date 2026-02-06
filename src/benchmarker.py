"""
VPR Benchmarking Tool — AutoBenchmarker
Connects to the local NSF HERD SQLite database, fetches institution-level
R&D features, and uses K-Nearest-Neighbors to find tight peer groups
(~10 institutions per university).
"""

import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


# ---------------------------------------------------------------------------
# SQL QUERY — edit column / table names here to match the HERD actual DB schema.
# The query pulls one row per institution using the most recent survey year.
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

# Numeric columns that will be scaled and fed into the clustering algorithm.
# Must match the SELECT list above (minus the identifier / name columns).
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
# Data connector
# ---------------------------------------------------------------------------
def fetch_university_features(db_path: str) -> pd.DataFrame:
    """Connect to the HERD SQLite database and return a feature DataFrame.

    Parameters
    ----------
    db_path : str
        Path to the ``herd.db`` file (e.g. ``"data/herd.db"``).

    Returns
    -------
    pd.DataFrame
        One row per institution with identifier, name, and funding columns.

    Raises
    ------
    FileNotFoundError
        If *db_path* does not point to an existing file.
    sqlite3.OperationalError
        If the table or columns referenced in ``FEATURES_QUERY`` are missing.
    """
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql(FEATURES_QUERY, conn)
    finally:
        conn.close()

    # Drop rows where all numeric features are null or zero to remove noise.
    df = df.dropna(subset=NUMERIC_COLS, how="all")
    df[NUMERIC_COLS] = df[NUMERIC_COLS].fillna(0)

    return df


# ---------------------------------------------------------------------------
# Benchmarker class
# ---------------------------------------------------------------------------
class AutoBenchmarker:
    """KNN-based peer-finding engine for NSF HERD institutions.

    I tired KMeans (clustering), but it produced wildly uneven groups on skewed funding data. 
    So here we use KNN to find the *n_peers* nearest neighbors for any given institution
    in a log-scaled, standardized feature space.

    Workflow
    --------
    1. Call :meth:`fit` with a DataFrame from :func:`fetch_university_features`.
    2. Call :meth:`get_peers` to retrieve the closest peer institutions.
    3. Call :meth:`analyze_gap` to compare a university against its peer average.

    Parameters
    ----------
    n_peers : int, optional
        Number of nearest neighbors to return (default ``10``).
    """

    def __init__(self, n_peers: int = 10):
        self.n_peers = n_peers

        # Populated by fit()
        self.scaler: StandardScaler | None = None
        self.nn_model: NearestNeighbors | None = None
        self.data: pd.DataFrame | None = None
        self._scaled: np.ndarray | None = None  # log-scaled feature matrix

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "AutoBenchmarker":
        """Log-transform, normalize, and index the feature space for KNN.

        Steps:
            1. ``np.log1p`` on every numeric column (compresses the skew so
               that $1M vs $2M and $1B vs $2B are weighted equally).
            2. ``StandardScaler`` to zero-mean, unit-variance.
            3. Fit a ``NearestNeighbors`` model for fast neighbor lookups.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns in ``NUMERIC_COLS`` plus ``inst_id``, ``name``.

        Returns
        -------
        AutoBenchmarker
            *self*, for method chaining (``bench.fit(df).get_peers(…)``).
        """
        self.data = df.copy().reset_index(drop=True)

        # --- Log-transform to tame the skew ---
        log_features = np.log1p(self.data[NUMERIC_COLS])

        # --- Standardize ---
        self.scaler = StandardScaler()
        self._scaled = self.scaler.fit_transform(log_features)

        # --- Build KNN index ---
        # n_neighbors is n_peers + 1 because the query point itself is
        # returned as its own nearest neighbor and we strip it out later.
        self.nn_model = NearestNeighbors(
            n_neighbors=min(self.n_peers + 1, len(self.data)),
            metric="euclidean",
            algorithm="auto",
        )
        self.nn_model.fit(self._scaled)

        return self

    # ------------------------------------------------------------------
    # get_peers
    # ------------------------------------------------------------------
    def get_peers(self, target_inst_id: str) -> list[str]:
        """Return the *n_peers* nearest institution names to *target_inst_id*.

        Parameters
        ----------
        target_inst_id : str
            The ``inst_id`` value of the target institution.

        Returns
        -------
        list[str]
            Peer institution names sorted by proximity (closest first).

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called yet.
        KeyError
            If *target_inst_id* is not present in the fitted data.
        """
        self._check_fitted()
        peer_indices = self._peer_indices(target_inst_id)
        return self.data.loc[peer_indices, "name"].tolist()

    # ------------------------------------------------------------------
    # analyze_gap
    # ------------------------------------------------------------------
    def analyze_gap(self, target_inst_id: str) -> list[dict]:
        """Compare the target institution to its peer-group average.

        The peer average is computed over the *n_peers* nearest neighbors
        (excluding the target itself).

        Parameters
        ----------
        target_inst_id : str
            The ``inst_id`` value of the target institution.

        Returns
        -------
        list[dict]
            One dict per numeric metric with keys:
            ``metric``, ``my_val``, ``peer_avg``, ``gap``.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called yet.
        KeyError
            If *target_inst_id* is not present in the fitted data.
        """
        self._check_fitted()
        target_row = self._find_target(target_inst_id)
        peer_indices = self._peer_indices(target_inst_id)
        peer_avg = self.data.loc[peer_indices, NUMERIC_COLS].mean()

        gaps = []
        for col in NUMERIC_COLS:
            my_val = float(target_row[col].values[0])
            avg_val = float(peer_avg[col])
            gaps.append(
                {
                    "metric": col,
                    "my_val": round(my_val, 2),
                    "peer_avg": round(avg_val, 2),
                    "gap": round(my_val - avg_val, 2),
                }
            )

        return gaps

    # ------------------------------------------------------------------
    # analyze_state_context
    # ------------------------------------------------------------------
    def analyze_state_context(self, target_inst_id: str) -> dict:
        """Geographical benchmarking — rank and funding share within the state.

        Useful for VPRs who need to justify budgets to state legislators by
        showing exactly where their institution stands among in-state peers.

        Parameters
        ----------
        target_inst_id : str
            The ``inst_id`` value of the target institution.

        Returns
        -------
        dict
            Keys:

            - **state** (*str*) — two-letter state code.
            - **state_rank** (*int*) — rank by ``total_rd`` within the state
              (1 = highest).
            - **total_in_state** (*int*) — number of institutions in the state.
            - **state_funding_share** (*float*) — target's share of the state's
              total ``state_local`` funding, as a percentage (0-100).
            - **top_competitor** (*str | None*) — name of the #1-ranked
              institution in the state, or ``None`` if the target *is* #1.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called yet.
        KeyError
            If *target_inst_id* is not present in the fitted data.
        ValueError
            If the fitted data does not contain a ``state`` column.
        """
        self._check_fitted()

        if "state" not in self.data.columns:
            raise ValueError(
                "The fitted DataFrame does not contain a 'state' column. "
                "Add 'state' to FEATURES_QUERY and re-run fetch + fit."
            )

        target_row = self._find_target(target_inst_id)

        # --- Identify state ---
        state_code = target_row["state"].values[0]

        # --- Filter to same state, rank by total_rd descending ---
        state_df = (
            self.data[self.data["state"] == state_code]
            .sort_values("total_rd", ascending=False)
            .reset_index(drop=True)
        )

        total_in_state = len(state_df)

        # Rank is 1-indexed position in the sorted list
        state_rank = int(
            state_df[state_df["inst_id"] == target_inst_id].index[0]
        ) + 1

        # --- State funding share (based on state_local column) ---
        state_total_funding = state_df["state_local"].sum()
        target_funding = float(target_row["state_local"].values[0])

        if state_total_funding > 0:
            funding_share = round(
                (target_funding / state_total_funding) * 100, 2
            )
        else:
            funding_share = 0.0

        # --- Top competitor ---
        # If the target IS #1, or is the only school, there is no competitor.
        if state_rank == 1 or total_in_state == 1:
            top_competitor = None
        else:
            top_competitor = state_df.iloc[0]["name"]

        return {
            "state": state_code,
            "state_rank": state_rank,
            "total_in_state": total_in_state,
            "state_funding_share": funding_share,
            "top_competitor": top_competitor,
        }

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _check_fitted(self) -> None:
        """Raise if fit() hasn't been called."""
        if self.nn_model is None or self.data is None:
            raise RuntimeError(
                "AutoBenchmarker has not been fitted yet. Call fit() first."
            )

    def _find_target(self, target_inst_id: str) -> pd.DataFrame:
        """Locate the target row or raise a clear error."""
        target_row = self.data[self.data["inst_id"] == target_inst_id]
        if target_row.empty:
            raise KeyError(
                f"Institution '{target_inst_id}' not found in the fitted data. "
                f"Available inst_ids (first 10): "
                f"{self.data['inst_id'].head(10).tolist()}"
            )
        return target_row

    def _peer_indices(self, target_inst_id: str) -> list[int]:
        """Return DataFrame indices of the *n_peers* nearest neighbors."""
        target_row = self._find_target(target_inst_id)
        target_idx = target_row.index[0]

        # Query the KNN model with the target's scaled features
        distances, indices = self.nn_model.kneighbors(
            self._scaled[target_idx].reshape(1, -1)
        )

        # Strip out the target itself (distance == 0) and return the rest
        neighbor_indices = [
            int(i) for i in indices[0] if i != target_idx
        ]
        return neighbor_indices[: self.n_peers]
