"""
NSF HERD Agency Funding Transformer
Extracts Q09 federal expenditures by agency (aggregated across all fields)
from the raw long-format microdata.

Design decisions:
  - We extract from Q09 where row = 'All' (the grand total across all fields),
    broken out by agency columns (DOD, DOE, HHS, NASA, NSF, USDA, Other agencies).
  - The agency columns have been completely stable across all 15 survey years
    (2010–2024). No taxonomy mapping needed.
  - We do NOT store the 'Total' column since it equals Q01 federal and is
    already in the `institutions` table. This avoids redundancy.
  - Values in the raw data are in THOUSANDS. We multiply by 1,000 to store
    dollars, matching the existing `institutions` table convention.
  - Rows with 0 or NaN are stored (an institution reporting $0 from DOD is
    meaningful — it means no DOD funding, not missing data).
"""

import pandas as pd
from pathlib import Path
import re


AGENCIES = ["DOD", "DOE", "HHS", "NASA", "NSF", "USDA", "Other agencies"]

# Clean display names for the UI
AGENCY_DISPLAY = {
    "DOD": "Dept of Defense",
    "DOE": "Dept of Energy",
    "HHS": "HHS (incl. NIH)",
    "NASA": "NASA",
    "NSF": "NSF",
    "USDA": "USDA",
    "Other agencies": "Other Federal",
}


class AgencyTransformer:
    def __init__(self, raw_dir, output_dir):
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)

    def transform_year(self, year):
        """Extract agency-level federal funding for one year."""
        csv_files = list(self.raw_dir.glob(f"*{year}*.csv"))
        if not csv_files:
            print(f"   ❌ No CSV for {year}")
            return None

        print(f"   📄 {year}: {csv_files[0].name}...", end=" ")

        try:
            df = pd.read_csv(
                csv_files[0],
                encoding="latin-1",
                dtype={"inst_id": str},
                low_memory=False,
            )
            df.columns = df.columns.str.strip()

            # Q09 where row = 'All' gives total federal by agency across all fields
            q09_all = df[
                (df["question"] == "Federal expenditures by field and agency")
                & (df["row"] == "All")
                & (df["column"].isin(AGENCIES))
            ][["inst_id", "column", "data"]].copy()

            q09_all = q09_all.rename(columns={
                "column": "agency_code",
                "data": "amount",
            })

            # Fill NaN amounts with 0, convert to dollars
            q09_all["amount"] = q09_all["amount"].fillna(0)
            q09_all["amount"] = (q09_all["amount"] * 1000).astype(int)

            # Add year and display name
            q09_all["year"] = year
            q09_all["agency_name"] = q09_all["agency_code"].map(AGENCY_DISPLAY)

            # Select final columns
            result = q09_all[
                ["inst_id", "year", "agency_code", "agency_name", "amount"]
            ].sort_values(["inst_id", "agency_code"]).reset_index(drop=True)

            print(f"✅ {result['inst_id'].nunique()} institutions, {len(result)} rows")
            return result

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def transform_all(self):
        """Transform all years and save."""
        print("=" * 80)
        print("NSF HERD AGENCY FUNDING TRANSFORMATION")
        print("=" * 80)
        print()

        years = set()
        for csv_file in self.raw_dir.glob("*.csv"):
            match = re.search(r"(\d{4})", csv_file.name)
            if match:
                years.add(int(match.group(1)))

        if not years:
            print(f"❌ No CSV files in {self.raw_dir}")
            return

        print(f"🔎 Found {len(years)} years: {sorted(years)}")
        print()

        all_dfs = []
        for year in sorted(years):
            result = self.transform_year(year)
            if result is not None:
                all_dfs.append(result)

        if not all_dfs:
            print("\n❌ No data transformed")
            return

        combined = pd.concat(all_dfs, ignore_index=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_file = self.output_dir / "agency_funding.csv"
        combined.to_csv(output_file, index=False)

        print()
        print("=" * 80)
        print("AGENCY TRANSFORMATION SUMMARY")
        print("=" * 80)
        print(f"  Total rows: {len(combined):,}")
        print(f"  Years: {combined['year'].min()}-{combined['year'].max()}")
        print(f"  Institutions: {combined['inst_id'].nunique():,}")
        print(f"  Agencies: {sorted(combined['agency_code'].unique())}")
        print(f"  📁 Output: {output_file}")
        print()


if __name__ == "__main__":
    transformer = AgencyTransformer(
        raw_dir="../../data/raw",
        output_dir="../../data/transformed",
    )
    transformer.transform_all()
