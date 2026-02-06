"""
Quick script to export clean CSV from checkpoint data.
Run this anytime to get the current results in the original 6-column format.

Handles both:
- research_ran = "true" -> uses dbs_testable, best_method, confidence (from deep research)
- research_ran = "false" -> uses pre_dbs_testable, pre_likely_method, pre_confidence (from preclassification)
"""

import pandas as pd
import argparse

CHECKPOINT_CSV = "checkpoint_partial.csv"
OUTPUT_CSV = "results_clean.csv"
CONFIDENCE_COL = "Confidence(0-1)"
LEGACY_CONFIDENCE_COL = " Confidence(0-1)"  # legacy: leading space


def get_value(df, i, col):
    """Safely get a string value from dataframe"""
    if col not in df.columns:
        return ""
    val = str(df.loc[i, col]).strip()
    if val in ["", "nan", "None", "NaN"]:
        return ""
    return val


def main():
    parser = argparse.ArgumentParser(
        description="Export a clean 6-column CSV from checkpoint_partial.csv"
    )
    parser.add_argument(
        "--checkpoint",
        default=CHECKPOINT_CSV,
        help="Path to checkpoint CSV (default: checkpoint_partial.csv)",
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_CSV,
        help="Output CSV path (default: results_clean.csv)",
    )
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    df = pd.read_csv(args.checkpoint, dtype=str)
    print(f"Loaded {len(df):,} biomarkers")

    # Normalize legacy confidence column name (leading space)
    if LEGACY_CONFIDENCE_COL in df.columns and CONFIDENCE_COL not in df.columns:
        df = df.rename(columns={LEGACY_CONFIDENCE_COL: CONFIDENCE_COL})

    deep_research_count = 0
    preclassify_count = 0

    # Copy results from working columns to original columns
    for i in range(len(df)):
        research_status = get_value(df, i, "research_ran").lower()

        # Determine which source columns to use
        if research_status == "true":
            # Deep research completed - use deep research results
            dbs_val = get_value(df, i, "dbs_testable")
            method_val = get_value(df, i, "best_method")
            conf_val = get_value(df, i, "confidence")
            if dbs_val:
                deep_research_count += 1
        else:
            # Use preclassification results (for false, failed, or empty)
            dbs_val = get_value(df, i, "pre_dbs_testable")
            method_val = get_value(df, i, "pre_likely_method")
            conf_val = get_value(df, i, "pre_confidence")
            if dbs_val:
                preclassify_count += 1

        # Copy to original columns
        if dbs_val:
            df.loc[i, "Testable by DBS"] = dbs_val
        if method_val:
            df.loc[i, "Best Method"] = method_val
        if conf_val:
            try:
                conf_float = float(conf_val)
                if not pd.isna(conf_float):
                    df.loc[i, CONFIDENCE_COL] = str(conf_float)
            except (ValueError, TypeError):
                pass

    print(f"\nData sources:")
    print(f"  From Deep Research: {deep_research_count:,}")
    print(f"  From Preclassification: {preclassify_count:,}")

    # Create clean output with only the 6 original columns
    original_columns = [
        "S. No.",
        "Identifier",
        "Biomarker Name",
        "Testable by DBS",
        "Best Method",
        CONFIDENCE_COL,
    ]
    for col in original_columns:
        if col not in df.columns:
            df[col] = ""
    df_clean = df[original_columns].copy()
    df_clean.to_csv(args.output, index=False)

    # Stats
    testable_yes = sum(
        1
        for i in range(len(df_clean))
        if str(df_clean.loc[i, "Testable by DBS"]).strip() == "Yes"
    )
    testable_no = sum(
        1
        for i in range(len(df_clean))
        if str(df_clean.loc[i, "Testable by DBS"]).strip() == "No"
    )
    testable_unclear = sum(
        1
        for i in range(len(df_clean))
        if str(df_clean.loc[i, "Testable by DBS"]).strip() == "Unclear"
    )
    empty = len(df_clean) - testable_yes - testable_no - testable_unclear

    print(f"\n✅ Exported to: {args.output}")
    print(f"\nCurrent Results:")
    print(f"  Yes: {testable_yes:,}")
    print(f"  No: {testable_no:,}")
    print(f"  Unclear: {testable_unclear:,}")
    print(f"  Not yet processed: {empty:,}")


if __name__ == "__main__":
    main()
