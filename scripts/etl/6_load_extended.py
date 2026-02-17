"""
NSF HERD Extended Data Loader
Loads field_expenditures and agency_funding CSVs into SQLite database.

Run AFTER 3_load.py (which creates the institutions table).
This script adds two new tables to the existing herd.db:
  - field_expenditures: R&D spending by research field (10 parent categories)
  - agency_funding: Federal spending by agency (7 agencies)

Both tables join to `institutions` on (inst_id, year).
"""

import pandas as pd
import sqlite3
from pathlib import Path


class ExtendedLoader:
    def __init__(self, transformed_dir, db_path):
        self.transformed_dir = Path(transformed_dir)
        self.db_path = Path(db_path)

    def load_all(self):
        """Load field_expenditures and agency_funding into the database."""
        print("=" * 80)
        print("NSF HERD EXTENDED DATA LOADER")
        print("=" * 80)
        print()

        if not self.db_path.exists():
            print(f"❌ Database not found: {self.db_path}")
            print(f"   Run 3_load.py first to create the institutions table.")
            return

        conn = sqlite3.connect(self.db_path)

        # Verify institutions table exists
        tables = pd.read_sql(
            "SELECT name FROM sqlite_master WHERE type='table'", conn
        )["name"].tolist()
        if "institutions" not in tables:
            print("❌ institutions table not found. Run 3_load.py first.")
            conn.close()
            return

        inst_count = pd.read_sql("SELECT COUNT(DISTINCT inst_id) as n FROM institutions", conn)["n"][0]
        print(f"✅ Found institutions table ({inst_count} institutions)")
        print()

        # --- Load field_expenditures ---
        self._load_table(
            conn,
            csv_name="field_expenditures.csv",
            table_name="field_expenditures",
            indexes=[
                ("idx_fe_inst_year", "field_expenditures(inst_id, year)"),
                ("idx_fe_field", "field_expenditures(field_code)"),
                ("idx_fe_inst_year_field", "field_expenditures(inst_id, year, field_code)"),
            ],
        )

        # --- Load agency_funding ---
        self._load_table(
            conn,
            csv_name="agency_funding.csv",
            table_name="agency_funding",
            indexes=[
                ("idx_af_inst_year", "agency_funding(inst_id, year)"),
                ("idx_af_agency", "agency_funding(agency_code)"),
                ("idx_af_inst_year_agency", "agency_funding(inst_id, year, agency_code)"),
            ],
        )

        # Compact
        print("   Compacting database...")
        conn.execute("VACUUM")
        print("   ✓ Database compacted")
        conn.close()

        print()
        self._verify()

    def _load_table(self, conn, csv_name, table_name, indexes):
        """Load a single CSV into a table, replacing if it exists."""
        csv_path = self.transformed_dir / csv_name
        if not csv_path.exists():
            print(f"❌ {csv_name} not found in {self.transformed_dir}")
            print(f"   Run the corresponding transform script first.")
            return

        print(f"📄 Loading {csv_name}...")
        df = pd.read_csv(csv_path, dtype={"inst_id": str})
        print(f"   {len(df):,} rows, {df['inst_id'].nunique()} institutions, "
              f"{df['year'].min()}-{df['year'].max()}")

        # Drop and recreate
        conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        df.to_sql(table_name, conn, index=False, if_exists="replace")

        # Create indexes
        for idx_name, idx_def in indexes:
            conn.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {idx_def}")

        print(f"   ✅ {table_name} loaded with {len(indexes)} indexes")
        print()

    def _verify(self):
        """Quick verification of created tables."""
        conn = sqlite3.connect(self.db_path)

        print("=" * 80)
        print("VERIFICATION")
        print("=" * 80)
        print()

        # List all tables
        tables = pd.read_sql(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name", conn
        )["name"].tolist()
        print(f"Tables in database: {tables}")
        print()

        # Verify field_expenditures
        if "field_expenditures" in tables:
            print("field_expenditures:")
            stats = pd.read_sql("""
                SELECT
                    COUNT(*) as rows,
                    COUNT(DISTINCT inst_id) as institutions,
                    COUNT(DISTINCT field_code) as fields,
                    MIN(year) as first_year,
                    MAX(year) as last_year
                FROM field_expenditures
            """, conn)
            for col in stats.columns:
                print(f"  {col}: {stats[col][0]}")

            # UNT check
            unt_fields = pd.read_sql("""
                SELECT field_code, federal, nonfederal, total
                FROM field_expenditures
                WHERE inst_id = '003594' AND year = (SELECT MAX(year) FROM field_expenditures)
                ORDER BY total DESC
            """, conn)
            if not unt_fields.empty:
                print(f"\n  UNT top fields (latest year):")
                for _, row in unt_fields.head(5).iterrows():
                    print(f"    {row['field_code']:20s}  Fed=${row['federal']:>12,}  "
                          f"NonFed=${row['nonfederal']:>12,}  Total=${row['total']:>12,}")
            print()

        # Verify agency_funding
        if "agency_funding" in tables:
            print("agency_funding:")
            stats = pd.read_sql("""
                SELECT
                    COUNT(*) as rows,
                    COUNT(DISTINCT inst_id) as institutions,
                    COUNT(DISTINCT agency_code) as agencies,
                    MIN(year) as first_year,
                    MAX(year) as last_year
                FROM agency_funding
            """, conn)
            for col in stats.columns:
                print(f"  {col}: {stats[col][0]}")

            # UNT check
            unt_agencies = pd.read_sql("""
                SELECT agency_code, agency_name, amount
                FROM agency_funding
                WHERE inst_id = '003594' AND year = (SELECT MAX(year) FROM agency_funding)
                ORDER BY amount DESC
            """, conn)
            if not unt_agencies.empty:
                print(f"\n  UNT agency breakdown (latest year):")
                for _, row in unt_agencies.iterrows():
                    print(f"    {row['agency_name']:20s}  ${row['amount']:>12,}")

            # Cross-check: agency total should equal institutions.federal
            print()
            cross = pd.read_sql("""
                SELECT
                    i.federal as q01_federal,
                    a.agency_total
                FROM institutions i
                LEFT JOIN (
                    SELECT inst_id, year, SUM(amount) as agency_total
                    FROM agency_funding
                    GROUP BY inst_id, year
                ) a ON i.inst_id = a.inst_id AND i.year = a.year
                WHERE i.inst_id = '003594'
                  AND i.year = (SELECT MAX(year) FROM institutions)
            """, conn)
            if not cross.empty:
                q01 = cross["q01_federal"][0]
                agency = cross["agency_total"][0]
                match = "✅" if q01 == agency else "⚠️ MISMATCH"
                print(f"  Cross-check UNT: Q01 federal=${q01:,}  Agency sum=${agency:,}  {match}")

        conn.close()
        print()
        print(f"📁 Database: {self.db_path}")
        print()


if __name__ == "__main__":
    loader = ExtendedLoader(
        transformed_dir="../../data/transformed",
        db_path="../../data/herd.db",
    )
    loader.load_all()
