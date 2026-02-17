"""
NSF HERD Field Expenditure Transformer
Extracts Q09 (federal by field & agency) and Q11 (nonfederal by field & source)
from the raw long-format microdata and produces a clean per-institution,
per-field, per-year CSV with federal and nonfederal breakdowns.

Design decisions:
  - We store BOTH parent-level fields (10 categories ending in ", all") and
    their sub-fields (~37 discipline-level breakdowns). Parent rows drive the
    Research Portfolio tab's top-level view; sub-fields power drill-downs
    and Q&A queries.
  - The `is_parent` flag (1/0) separates the two levels. The `parent_field`
    column links sub-fields to their parent, so the tab can do:
        WHERE parent_field = 'engineering' AND is_parent = 0
    to get Engineering sub-fields, or:
        WHERE is_parent = 1
    to get the top-level portfolio view.
  - We combine Q09 (federal by agency, per field) and Q11 (nonfederal by
    source, per field) into a single output. For each field:
        federal    = Q09 column "Total"
        nonfederal = Q11 column "Total"
        total      = federal + nonfederal
  - Values in the raw data are in THOUSANDS. We multiply by 1,000 to store
    dollars, matching the existing `institutions` table convention.
  - The grand total row (row = "All") is excluded — it duplicates
    institutions.total_rd and institutions.federal.
  - Institutions only report fields where they have activity. Missing fields
    mean $0, not unknown. The UI should fill in $0 for display purposes.

Survey evolution notes:
  - 2010–2015: 32 sub-fields + 10 parents = 42 distinct field rows per institution
  - 2016–2024: 36 sub-fields + 10 parents = 46 distinct field rows per institution
  - The 4 sub-fields added in 2016 (Engineering industrial/manufacturing,
    Life sciences natural resources, Physical sciences materials science,
    Social sciences anthropology) simply don't exist in pre-2016 data.
    No backfill needed — absence is truthful.
  - Parent "all" rollups are unaffected by this transition. Their totals
    already included the activity that later got its own sub-field code.
"""

import pandas as pd
from pathlib import Path
import re


# ---------------------------------------------------------------------------
# Field taxonomy
#
# Maps every NSF row value to a stable (field_code, parent_field, is_parent)
# tuple. This is the single source of truth for field identity.
#
# Why hardcode instead of auto-generate?
#   1. Auto-slugified names are ugly and fragile ("engineering_bioengineering_
#      and_biomedical_engineering")
#   2. We want short, readable codes that work well in SQL and display
#   3. This list has changed exactly once in 15 years (2016, additive only)
#   4. If NSF adds a field, we want to notice and add it deliberately
#
# Parents with no sub-fields (CS, Math, Psychology, Other Sciences) are
# standalone — the survey doesn't break them into sub-disciplines.
# ---------------------------------------------------------------------------
FIELD_TAXONOMY = {
    # ── Computer & Information Sciences (no sub-fields in survey) ──
    "Computer and information sciences, all":
        ("cs", "cs", True),

    # ── Engineering ──
    "Engineering, all":
        ("engineering", "engineering", True),
    "Engineering, aerospace, aeronautical, and astronautical":
        ("eng_aerospace", "engineering", False),
    "Engineering, bioengineering and biomedical engineering":
        ("eng_biomedical", "engineering", False),
    "Engineering, chemical":
        ("eng_chemical", "engineering", False),
    "Engineering, civil":
        ("eng_civil", "engineering", False),
    "Engineering, electrical, electronic, and communications":
        ("eng_electrical", "engineering", False),
    "Engineering, industrial and manufacturing":            # Added 2016
        ("eng_industrial", "engineering", False),
    "Engineering, mechanical":
        ("eng_mechanical", "engineering", False),
    "Engineering, metallurgical and materials":
        ("eng_materials", "engineering", False),
    "Engineering, other":
        ("eng_other", "engineering", False),

    # ── Geosciences ──
    "Geosciences, atmospheric sciences, and ocean sciences, all":
        ("geosciences", "geosciences", True),
    "Geosciences, atmospheric sciences, and ocean sciences, atmospheric science and meteorology":
        ("geo_atmospheric", "geosciences", False),
    "Geosciences, atmospheric sciences, and ocean sciences, geological and earth sciences":
        ("geo_earth", "geosciences", False),
    "Geosciences, atmospheric sciences, and ocean sciences, ocean sciences and marine sciences":
        ("geo_ocean", "geosciences", False),
    "Geosciences, atmospheric sciences, and ocean sciences, other":
        ("geo_other", "geosciences", False),

    # ── Life Sciences ──
    "Life sciences, all":
        ("life_sciences", "life_sciences", True),
    "Life sciences, agricultural sciences":
        ("life_agricultural", "life_sciences", False),
    "Life sciences, biological and biomedical sciences":
        ("life_biomedical", "life_sciences", False),
    "Life sciences, health sciences":
        ("life_health", "life_sciences", False),
    "Life sciences, natural resources and conservation":     # Added 2016
        ("life_natural_resources", "life_sciences", False),
    "Life sciences, other":
        ("life_other", "life_sciences", False),

    # ── Mathematics & Statistics (no sub-fields in survey) ──
    "Mathematics and statistics, all":
        ("math", "math", True),

    # ── Physical Sciences ──
    "Physical sciences, all":
        ("physical_sciences", "physical_sciences", True),
    "Physical sciences, astronomy and astrophysics":
        ("phys_astronomy", "physical_sciences", False),
    "Physical sciences, chemistry":
        ("phys_chemistry", "physical_sciences", False),
    "Physical sciences, materials science":                  # Added 2016
        ("phys_materials", "physical_sciences", False),
    "Physical sciences, physics":
        ("phys_physics", "physical_sciences", False),
    "Physical sciences, other":
        ("phys_other", "physical_sciences", False),

    # ── Psychology (no sub-fields in survey) ──
    "Psychology, all":
        ("psychology", "psychology", True),

    # ── Social Sciences ──
    "Social sciences, all":
        ("social_sciences", "social_sciences", True),
    "Social sciences, anthropology":                         # Added 2016
        ("soc_anthropology", "social_sciences", False),
    "Social sciences, economics":
        ("soc_economics", "social_sciences", False),
    "Social sciences, political science and government":
        ("soc_political", "social_sciences", False),
    "Social sciences, sociology, demography, and population studies":
        ("soc_sociology", "social_sciences", False),
    "Social sciences, other":
        ("soc_other", "social_sciences", False),

    # ── Other Sciences (no sub-fields in survey) ──
    "Other sciences, all":
        ("other_sciences", "other_sciences", True),

    # ── Non-S&E ──
    "Non-S&E, all":
        ("non_se", "non_se", True),
    "Non-S&E, business management and business administration":
        ("nse_business", "non_se", False),
    "Non-S&E, communication and communications technologies":
        ("nse_communication", "non_se", False),
    "Non-S&E, education":
        ("nse_education", "non_se", False),
    "Non-S&E, humanities":
        ("nse_humanities", "non_se", False),
    "Non-S&E, law":
        ("nse_law", "non_se", False),
    "Non-S&E, social work":
        ("nse_social_work", "non_se", False),
    "Non-S&E, visual and performing arts":
        ("nse_arts", "non_se", False),
    "Non-S&E, other":
        ("nse_other", "non_se", False),
}

# Convenience set: all row values we want to extract (everything except "All")
KNOWN_FIELDS = set(FIELD_TAXONOMY.keys())


class FieldTransformer:
    def __init__(self, raw_dir, output_dir):
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)

    def transform_year(self, year):
        """Extract field expenditures (parents + sub-fields) for one year."""
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

            # --- Q09: Federal expenditures by field and agency ---
            # Filter to known fields only (excludes "All" grand total row).
            # Column "Total" gives total federal across all agencies for that field.
            q09 = df[
                (df["question"] == "Federal expenditures by field and agency")
                & (df["row"].isin(KNOWN_FIELDS))
                & (df["column"] == "Total")
            ][["inst_id", "row", "data"]].copy()
            q09 = q09.rename(columns={"row": "field_name", "data": "federal"})

            # --- Q11: Nonfederal expenditures by field and source ---
            q11 = df[
                (df["question"] == "Nonfederal expenditures by field and source")
                & (df["row"].isin(KNOWN_FIELDS))
                & (df["column"] == "Total")
            ][["inst_id", "row", "data"]].copy()
            q11 = q11.rename(columns={"row": "field_name", "data": "nonfederal"})

            # --- Merge federal + nonfederal on (inst_id, field_name) ---
            # Outer join: some fields may appear in Q09 but not Q11 or vice versa.
            # Example: an institution with federal engineering funding but no
            # nonfederal engineering funding.
            merged = pd.merge(
                q09, q11,
                on=["inst_id", "field_name"],
                how="outer",
            )

            merged["federal"] = merged["federal"].fillna(0)
            merged["nonfederal"] = merged["nonfederal"].fillna(0)

            # Convert from thousands to dollars
            merged["federal"] = (merged["federal"] * 1000).astype(int)
            merged["nonfederal"] = (merged["nonfederal"] * 1000).astype(int)
            merged["total"] = merged["federal"] + merged["nonfederal"]

            merged["year"] = year

            # Apply taxonomy: map each field_name → (field_code, parent_field, is_parent)
            taxonomy = merged["field_name"].map(FIELD_TAXONOMY)
            merged["field_code"] = taxonomy.apply(lambda x: x[0] if x else None)
            merged["parent_field"] = taxonomy.apply(lambda x: x[1] if x else None)
            merged["is_parent"] = taxonomy.apply(lambda x: int(x[2]) if x else None)

            # Drop any rows that didn't match taxonomy (shouldn't happen, but
            # guards against NSF adding a field we haven't mapped yet).
            unmapped = merged[merged["field_code"].isna()]
            if not unmapped.empty:
                unknown = unmapped["field_name"].unique().tolist()
                print(f"\n   ⚠️  Unmapped fields in {year}: {unknown}")
            merged = merged.dropna(subset=["field_code"])

            result = merged[
                ["inst_id", "year", "field_code", "parent_field", "is_parent",
                 "field_name", "federal", "nonfederal", "total"]
            ].sort_values(["inst_id", "is_parent", "field_code"]).reset_index(drop=True)

            n_parents = len(result[result["is_parent"] == 1])
            n_subs = len(result[result["is_parent"] == 0])
            n_inst = result["inst_id"].nunique()
            print(f"✅ {n_inst} institutions, {n_parents} parent rows, {n_subs} sub-field rows")
            return result

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def transform_all(self):
        """Transform all years and save."""
        print("=" * 80)
        print("NSF HERD FIELD EXPENDITURE TRANSFORMATION")
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
        output_file = self.output_dir / "field_expenditures.csv"
        combined.to_csv(output_file, index=False)

        print()
        print("=" * 80)
        print("FIELD TRANSFORMATION SUMMARY")
        print("=" * 80)
        print(f"  Total rows:     {len(combined):,}")
        print(f"  Parent rows:    {len(combined[combined['is_parent'] == 1]):,}")
        print(f"  Sub-field rows: {len(combined[combined['is_parent'] == 0]):,}")
        print(f"  Years:          {combined['year'].min()}-{combined['year'].max()}")
        print(f"  Institutions:   {combined['inst_id'].nunique():,}")
        print(f"  Field codes:    {combined['field_code'].nunique()}")
        print(f"  📁 Output:      {output_file}")
        print()


if __name__ == "__main__":
    transformer = FieldTransformer(
        raw_dir="../../data/raw",
        output_dir="../../data/transformed",
    )
    transformer.transform_all()
