import os
import re
import time
import json
import argparse
from typing import Any, Dict, Optional, Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from dotenv import load_dotenv

from google import genai

# -----------------------------
# Configuration (edit as needed)
# -----------------------------

# Cheap, fast model for bulk classification
# Using stable GA model for reliability (as of Feb 2026)
FAST_MODEL = "gemini-2.5-flash"  # Fast, stable, cost-effective
# Structured extraction model (must support JSON schema well)
STRUCTURED_MODEL = "gemini-2.5-flash"  # Same model, consistent performance

# Deep Research agent id from official docs (Dec 2025 preview)
DEEP_RESEARCH_AGENT = "deep-research-pro-preview-12-2025"

# Polling settings for Deep Research
POLL_SECONDS = 10
MAX_POLL_MINUTES = (
    90  # 90 minutes timeout for batched research (25 biomarkers can take long)
)

# When to send to Deep Research (Conservative threshold to control costs)
# With 6K biomarkers, 0.50 = ~20-30% deep research, 0.70 = ~40-50%
SEND_TO_RESEARCH_IF_CONFIDENCE_BELOW = 0.55

# Batching configuration for Fast Preclassification
PRECLASSIFY_BATCH_SIZE = 40  # Process 40 biomarkers per preclassify call
MAX_CONCURRENT_PRECLASSIFY = 25  # Run 25 preclassify calls in parallel

# Batching configuration for Deep Research
# IMPORTANT: Deep Research API has a STRICT rate limit of 10 requests per minute!
# Going over this limit causes 429 errors and wastes requests
BIOMARKERS_PER_BATCH = (
    25  # Process 25 biomarkers in one Deep Research call (more efficient)
)
MAX_CONCURRENT_RESEARCH = 8  # ONLY 8 concurrent (API limit is 10 RPM, stay under it)
DEEP_RESEARCH_BATCH_DELAY = 15  # Seconds to wait between starting new batch groups

# Checkpoint files (so you can resume without losing progress)
CHECKPOINT_CSV = "checkpoint_partial.csv"

# Output column names (input datasets sometimes contain legacy names)
CONFIDENCE_COL = "Confidence(0-1)"
LEGACY_CONFIDENCE_COL = " Confidence(0-1)"  # legacy: leading space

# -----------------------------
# Helpers
# -----------------------------


def ensure_api_key():
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        raise RuntimeError(
            "Missing API key. Set GEMINI_API_KEY (preferred) or GOOGLE_API_KEY."
        )


def build_genai_client() -> genai.Client:
    """
    Builds a Gemini client.
    The SDK has supported both env-var based auth and explicit api_key depending on version.
    We support both patterns to stay robust across environments.
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    try:
        return genai.Client(api_key=api_key)  # type: ignore[arg-type]
    except TypeError:
        # Fall back to env-var based auth if this SDK version doesn't accept api_key kwarg.
        return genai.Client()


def normalize_identifier(identifier: Any) -> str:
    """Safely normalize identifier to string."""
    if identifier is None or (isinstance(identifier, float) and pd.isna(identifier)):
        return ""
    return str(identifier).strip()


def normalize_name(name: Any) -> str:
    """Safely normalize name to string."""
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""
    return str(name).strip()


@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    retry=retry_if_exception_type(Exception),
)
def gemini_generate_text(client: genai.Client, model: str, prompt: str) -> str:
    resp = client.models.generate_content(model=model, contents=prompt)
    return resp.text or ""


def is_rate_limit_error(exception):
    """Check if the exception is a rate limit error (429)."""
    error_str = str(exception).lower()
    return "429" in error_str or "rate" in error_str or "quota" in error_str


@retry(
    reraise=True,
    stop=stop_after_attempt(4),  # More retries for rate limit recovery
    wait=wait_exponential(
        multiplier=2, min=30, max=120
    ),  # Longer waits for rate limits
    retry=retry_if_exception_type(Exception),
)
def deep_research_batch_run(
    client: genai.Client, prompt: str, batch_id: int, batch_size: int
) -> str:
    """
    Starts Deep Research in background for a batch of biomarkers.
    Polls until completed/failed.

    Args:
        client: Gemini client
        prompt: Batched research prompt
        batch_id: Batch identifier for logging
        batch_size: Number of biomarkers in this batch
    """
    print(f"\n🔬 Batch {batch_id}: Starting Deep Research for {batch_size} biomarkers")

    interaction = client.interactions.create(
        input=prompt,
        agent=DEEP_RESEARCH_AGENT,
        background=True,
    )

    interaction_id = getattr(interaction, "id", None)
    if not interaction_id:
        raise RuntimeError("Deep Research did not return an interaction id.")

    print(f"   ✓ Batch {batch_id} Interaction ID: {interaction_id}")

    deadline = time.time() + MAX_POLL_MINUTES * 60
    start_time = time.time()
    poll_count = 0

    while time.time() < deadline:
        time.sleep(POLL_SECONDS)
        poll_count += 1
        elapsed = int(time.time() - start_time)

        interaction = client.interactions.get(interaction_id)
        status = getattr(interaction, "status", None)

        # Progress indicator (less frequent for batches)
        if poll_count % 6 == 0:  # Every 60 seconds
            print(f"   ⏳ Batch {batch_id}: {elapsed}s elapsed (status: {status})")

        if status == "completed":
            print(
                f"   ✅ Batch {batch_id}: Completed in {elapsed}s ({batch_size} biomarkers)"
            )
            if interaction.outputs:
                report = interaction.outputs[-1].text or ""
                if len(report) < 500:
                    print(
                        f"   ⚠️  Batch {batch_id}: Short report ({len(report)} chars) - may be incomplete"
                    )
                return report
            print(f"   ⚠️  Batch {batch_id}: No outputs in completed interaction")
            return ""

        if status == "failed":
            err = getattr(interaction, "error", "unknown error")
            print(f"   ❌ Batch {batch_id}: Failed after {elapsed}s: {err}")
            raise RuntimeError(f"Deep Research failed: {err}")

    elapsed = int(time.time() - start_time)
    print(f"   ⏰ Batch {batch_id}: Timeout after {elapsed}s")
    raise TimeoutError(f"Deep Research timed out after {MAX_POLL_MINUTES} minutes.")


def cheap_preclassify_batch_prompt(biomarkers: List[Tuple[str, str, int]]) -> str:
    """
    Batched preclassification prompt for faster processing.
    Process multiple biomarkers in one API call.
    """
    biomarker_list = "\n".join(
        [
            f"{i+1}. Identifier: {bio[0]}, Name: {bio[1]}"
            for i, bio in enumerate(biomarkers)
        ]
    )

    return f"""
You are classifying if biomarkers are realistically testable from dried blood spots (DBS) in humans.

BIOMARKERS TO CLASSIFY ({len(biomarkers)} total):
{biomarker_list}

Return ONLY a valid JSON ARRAY with one object per biomarker:
[
  {{
    "identifier": string (exact match),
    "name": string (exact match),
    "dbs_testable": "Yes" | "No" | "Unclear",
    "confidence": number 0..1,
    "likely_method": "LC-MS/MS" | "immunoassay" | "enzymatic" | "other" | "unknown",
    "short_reason": string (<= 30 words)
  }},
  ... (repeat for all {len(biomarkers)} biomarkers)
]

DBS Testability Criteria:
1. Small molecules/metabolites (CHEBI):
   - YES if: stable at room temp, detectable in blood, MW < 2000 Da → LC-MS/MS
   - NO if: highly volatile, rapidly degraded, or plasma-only

2. Proteins (UniProt):
   - YES if: abundant in blood (>1 µg/mL), stable when dried → immunoassay
   - NO if: very large (>300 kDa), extremely low abundance
   - UNCLEAR if: moderate abundance with no DBS validation

3. Other (RNA/DNA/ENSEMBL): Usually YES if extraction from DBS works

CRITICAL: Return exactly {len(biomarkers)} objects in the array. Use exact identifiers and names from the list above.
""".strip()


def deep_research_batch_prompt(biomarkers: List[Tuple[str, str, int]]) -> str:
    """
    Create a batched Deep Research prompt for multiple biomarkers.
    Enhanced for efficient batch processing.

    Args:
        biomarkers: List of (identifier, name, dataframe_index) tuples
    """
    biomarker_list = "\n".join(
        [
            f"{i+1}. Identifier: {bio[0]}, Name: {bio[1]}"
            for i, bio in enumerate(biomarkers)
        ]
    )

    return f"""
Research task: Determine whether each of the following {len(biomarkers)} biomarkers is testable from dried blood spots (DBS) in humans for clinical/diagnostic purposes.

BIOMARKERS TO ANALYZE:
{biomarker_list}

IMPORTANT: You must provide analysis for ALL {len(biomarkers)} biomarkers listed above.

Required Analysis FOR EACH BIOMARKER:

1) DBS Testability Conclusion: Yes / No / Unclear
   - YES only if: validated DBS methods exist OR strong evidence of stability + detectability
   - NO if: known instability, incompatible with DBS matrix, or technical impossibility
   - UNCLEAR if: theoretically possible but no validation studies found

2) Analytical Method Assessment:
   - Primary detection method (LC-MS/MS, immunoassay, enzymatic, PCR, etc.)
   - Sample preparation requirements (extraction method, volume needed)
   - Sensitivity/LOD requirements and if achievable from DBS

3) Critical DBS-Specific Factors:
   - **Stability**: Does analyte remain stable when dried on filter paper (days to weeks)?
   - **Hematocrit effect**: Is measurement affected by blood cell concentration?
   - **Matrix effects**: Interference from dried blood components?
   - **Blood concentration**: Is analyte present at detectable levels in whole blood?
   - **Recovery**: Can analyte be efficiently extracted from dried spots?

4) Evidence Quality:
   - Cite at least 2-3 peer-reviewed sources (PubMed, journals)
   - Include any FDA/EMA approved DBS methods if applicable
   - Note if evidence is from animal studies vs. human validation
   - If NO direct DBS validation exists, clearly state this and base conclusion on analyte properties

5) Practical Considerations:
   - Volume of blood needed (typical DBS = 15-50 µL)
   - Storage/shipping stability at room temperature
   - Any known commercial DBS tests available

OUTPUT FORMAT:
Structure your report with clear sections for each biomarker:

## BIOMARKER 1: [Identifier] - [Name]
**DBS Testability:** Yes/No/Unclear
**Confidence:** 0.X
**Method:** [method]
[Analysis details...]

## BIOMARKER 2: [Identifier] - [Name]
[...continue for all biomarkers...]

CRITICAL REQUIREMENTS:
- Analyze ALL {len(biomarkers)} biomarkers listed above
- Use clear section headers with biomarker identifiers
- If you cannot find direct DBS validation studies, mark as "Unclear"
- Provide evidence/citations for every major claim
- Be concise but thorough (aim for 200-400 words per biomarker)
""".strip()


def extract_structured_from_batch_report_prompt(
    report_text: str, biomarkers: List[Tuple[str, str, int]]
) -> str:
    """
    Extract structured data for multiple biomarkers from a batched Deep Research report.
    """
    biomarker_list = "\n".join(
        [f"{i+1}. {bio[0]} - {bio[1]}" for i, bio in enumerate(biomarkers)]
    )

    return f"""
Convert the following batched research report into STRICT JSON array format.

BIOMARKERS IN THIS BATCH:
{biomarker_list}

OUTPUT FORMAT - Return a JSON array with one object per biomarker:
[
  {{
    "identifier": string (EXACT match from list above),
    "name": string (EXACT match from list above),
    "dbs_testable": "Yes" | "No" | "Unclear",
    "confidence": number (0..1),
    "best_method": string,
    "sample_prep_notes": string,
    "limitations": [string, ...],
    "evidence": [{{"title": string, "url": string, "note": string}}, ...]
  }},
  ... (repeat for all {len(biomarkers)} biomarkers)
]

CRITICAL RULES:
1. You MUST return exactly {len(biomarkers)} objects in the array
2. Use EXACT identifier and name from the list above
3. If a biomarker wasn't analyzed in the report, use:
   - dbs_testable: "Unclear"
   - confidence: 0.0
   - best_method: "not analyzed"
4. If report lacks direct DBS validation, set dbs_testable="Unclear" and confidence <= 0.6
5. evidence: include 2-5 items per biomarker, with URLs if available
6. Output ONLY the JSON array, no markdown, no explanation

BATCHED RESEARCH REPORT:
{report_text}
""".strip()


def parse_json_loose(text: str) -> Optional[Any]:
    """
    Tries to parse JSON even if model returns extra text.
    Can return dict, list, or other JSON types.
    """
    if not text:
        return None

    # Try to find JSON array first (for batched results)
    array_match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if array_match:
        try:
            return json.loads(array_match.group(0))
        except Exception:
            pass

    # Fall back to finding JSON object
    obj_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if obj_match:
        try:
            return json.loads(obj_match.group(0))
        except Exception:
            pass

    return None


def coerce_preclassify(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalizes preclassify fields.
    """
    out = {
        "pre_dbs_testable": "Unclear",
        "pre_confidence": 0.0,
        "pre_likely_method": "unknown",
        "pre_short_reason": "",
    }
    if not isinstance(d, dict):
        return out

    dbs = str(d.get("dbs_testable", "Unclear")).strip().title()
    if dbs not in ("Yes", "No", "Unclear"):
        dbs = "Unclear"
    out["pre_dbs_testable"] = dbs

    try:
        conf = float(d.get("confidence", 0.0))
        conf = max(0.0, min(1.0, conf))
    except Exception:
        conf = 0.0
    out["pre_confidence"] = conf

    method = str(d.get("likely_method", "unknown")).strip()
    out["pre_likely_method"] = method[:50]

    reason = str(d.get("short_reason", "")).strip()
    out["pre_short_reason"] = reason[:200]

    return out


def coerce_final(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalizes final structured fields.
    """
    out = {
        "dbs_testable": "Unclear",
        "confidence": 0.0,
        "best_method": "",
        "sample_prep_notes": "",
        "limitations": "[]",
        "evidence": "[]",
    }
    if not isinstance(d, dict):
        return out

    dbs = str(d.get("dbs_testable", "Unclear")).strip().title()
    if dbs not in ("Yes", "No", "Unclear"):
        dbs = "Unclear"
    out["dbs_testable"] = dbs

    try:
        conf = float(d.get("confidence", 0.0))
        conf = max(0.0, min(1.0, conf))
    except Exception:
        conf = 0.0
    out["confidence"] = conf

    out["best_method"] = str(d.get("best_method", "")).strip()[:100]
    out["sample_prep_notes"] = str(d.get("sample_prep_notes", "")).strip()[:1000]

    limitations = d.get("limitations", [])
    if not isinstance(limitations, list):
        limitations = []
    out["limitations"] = json.dumps(
        [str(x)[:200] for x in limitations][:20], ensure_ascii=False
    )

    evidence = d.get("evidence", [])
    if not isinstance(evidence, list):
        evidence = []
    # keep only a few, sanitize
    cleaned = []
    for item in evidence[:8]:
        if isinstance(item, dict):
            cleaned.append(
                {
                    "title": str(item.get("title", ""))[:200],
                    "url": str(item.get("url", ""))[:500],
                    "note": str(item.get("note", ""))[:300],
                }
            )
    out["evidence"] = json.dumps(cleaned, ensure_ascii=False)

    return out


def main():
    # Load environment variables from .env file
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Classify biomarkers for DBS testability using Gemini AI"
    )
    parser.add_argument("--input", required=True, help="Input CSV file path")
    parser.add_argument("--output", required=True, help="Output CSV file path")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint_partial.csv if present",
    )
    parser.add_argument(
        "--skip-preclassify",
        action="store_true",
        help="Skip stage A (preclassify) and only run deep research",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show statistics without making API calls",
    )
    args = parser.parse_args()

    # Load
    if args.resume and os.path.exists(CHECKPOINT_CSV):
        print(f"\n📂 Resuming from checkpoint: {CHECKPOINT_CSV}")
        df = pd.read_csv(CHECKPOINT_CSV, dtype=str)
        print(f"   Loaded {len(df):,} biomarkers from checkpoint.")
    else:
        print(f"\n📂 Loading input file: {args.input}")
        df = pd.read_csv(args.input, dtype=str)
        print(f"   Loaded {len(df):,} biomarkers.")

    # Normalize legacy confidence column name (leading space)
    if LEGACY_CONFIDENCE_COL in df.columns and CONFIDENCE_COL not in df.columns:
        df = df.rename(columns={LEGACY_CONFIDENCE_COL: CONFIDENCE_COL})

    # Ensure serial column exists (some inputs may omit it)
    if "S. No." not in df.columns:
        df.insert(0, "S. No.", [str(i + 1) for i in range(len(df))])

    # Ensure required columns exist
    if "Identifier" not in df.columns or "Biomarker Name" not in df.columns:
        raise RuntimeError("Input must contain columns: Identifier, Biomarker Name")

    # Define column data types for new columns
    column_types = {
        "pre_dbs_testable": str,
        "pre_confidence": float,
        "pre_likely_method": str,
        "pre_short_reason": str,
        "research_ran": str,
        "deep_research_report": str,
        "dbs_testable": str,
        "confidence": float,
        "best_method": str,
        "sample_prep_notes": str,
        "limitations": str,
        "evidence": str,
    }

    # Add output columns if missing with proper types
    for col, dtype in column_types.items():
        if col not in df.columns:
            if dtype == float:
                df[col] = pd.Series(dtype=float)
            else:
                df[col] = pd.Series(dtype=str)
        else:
            # Convert existing columns to proper type
            if dtype == float:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                df[col] = df[col].astype(str)

    # Dry run mode: show statistics
    if args.dry_run:
        total = len(df)
        already_preclassified = 0
        for i in range(len(df)):
            pre_dbs = str(df.loc[i, "pre_dbs_testable"]).strip()
            pre_conf = df.loc[i, "pre_confidence"]
            if pre_dbs in ("Yes", "No", "Unclear") and not pd.isna(pre_conf):
                already_preclassified += 1

        already_researched = sum(
            1
            for i in range(len(df))
            if str(df.loc[i, "research_ran"]).strip().lower() == "true"
        )

        remaining_preclassify = total - already_preclassified
        preclassify_batches = (
            remaining_preclassify + PRECLASSIFY_BATCH_SIZE - 1
        ) // PRECLASSIFY_BATCH_SIZE
        estimated_research_pct = 0.30
        estimated_research_biomarkers = int(
            remaining_preclassify * estimated_research_pct
        )
        research_batches = (
            estimated_research_biomarkers + BIOMARKERS_PER_BATCH - 1
        ) // BIOMARKERS_PER_BATCH

        print("\n" + "=" * 60)
        print("DRY RUN - Statistics & Cost Estimation")
        print("=" * 60)
        print(f"Total biomarkers: {total:,}")
        print(f"Already preclassified: {already_preclassified:,}")
        print(f"Remaining for preclassify: {remaining_preclassify:,}")
        print(f"Already deep researched: {already_researched:,}")
        print("\n--- STAGE A: Preclassification ---")
        print(
            f"Batches needed: {preclassify_batches:,} (@ {PRECLASSIFY_BATCH_SIZE} biomarkers/batch)"
        )
        print(f"Parallel execution: {MAX_CONCURRENT_PRECLASSIFY} batches at once")
        print(
            f"Estimated time: ~{preclassify_batches // MAX_CONCURRENT_PRECLASSIFY * 2}-{preclassify_batches // MAX_CONCURRENT_PRECLASSIFY * 4} minutes"
        )
        print(f"Estimated cost: ${preclassify_batches * 0.001:.2f}")
        print("\n--- STAGE B: Deep Research (~30% of total) ---")
        print(
            f"Estimated biomarkers needing research: {estimated_research_biomarkers:,}"
        )
        print(
            f"Research batches: {research_batches:,} (@ {BIOMARKERS_PER_BATCH} biomarkers/batch)"
        )
        print(
            f"Parallel execution: {MAX_CONCURRENT_RESEARCH} batches at once (API limit: 10 RPM)"
        )
        batch_groups = (
            research_batches + MAX_CONCURRENT_RESEARCH - 1
        ) // MAX_CONCURRENT_RESEARCH
        minutes_per_group = 10 + (
            DEEP_RESEARCH_BATCH_DELAY // 60
        )  # ~10 min research + delay
        estimated_hours = (batch_groups * minutes_per_group) / 60
        print(
            f"Batch groups: {batch_groups} (with {DEEP_RESEARCH_BATCH_DELAY}s delay between)"
        )
        print(
            f"Estimated time: ~{estimated_hours:.1f}-{estimated_hours * 1.5:.1f} hours"
        )
        print(f"Estimated cost: ${research_batches * 3:.2f}")
        print("\n--- TOTAL ESTIMATES ---")
        print(
            f"Total estimated cost: ${preclassify_batches * 0.001 + research_batches * 3:.2f}"
        )
        print(
            f"Total estimated time: ~{preclassify_batches // MAX_CONCURRENT_PRECLASSIFY * 3 // 60 + research_batches // MAX_CONCURRENT_RESEARCH * 10}-{preclassify_batches // MAX_CONCURRENT_PRECLASSIFY * 5 // 60 + research_batches // MAX_CONCURRENT_RESEARCH * 15} hours"
        )
        print("=" * 60)
        return

    # Only require API key when we are actually going to call Gemini.
    ensure_api_key()
    client = build_genai_client()

    # Stage A: Batched preclassification
    if not args.skip_preclassify:
        print("\n" + "=" * 60)
        print("STAGE A: Batched Fast Preclassification")
        print("=" * 60)

        # Collect biomarkers that need preclassification
        biomarkers_to_preclassify = []
        for i in range(len(df)):
            pre_dbs_val = str(df.loc[i, "pre_dbs_testable"]).strip()
            pre_conf_val = df.loc[i, "pre_confidence"]

            # Check if already preclassified (has valid dbs_testable and confidence)
            is_preclassified = (
                pre_dbs_val in ("Yes", "No", "Unclear")
                and not pd.isna(pre_conf_val)
                and str(pre_conf_val).strip() != ""
                and str(pre_conf_val).strip() != "nan"
            )

            if is_preclassified:
                continue  # already done

            identifier = normalize_identifier(df.loc[i, "Identifier"])
            name = normalize_name(df.loc[i, "Biomarker Name"])
            biomarkers_to_preclassify.append((identifier, name, i))

        if len(biomarkers_to_preclassify) == 0:
            print("✅ All biomarkers already preclassified! Skipping Stage A.")
            print("   (Using existing checkpoint data)")
        else:
            num_batches = (
                len(biomarkers_to_preclassify) + PRECLASSIFY_BATCH_SIZE - 1
            ) // PRECLASSIFY_BATCH_SIZE
            print(f"Biomarkers to preclassify: {len(biomarkers_to_preclassify)}")
            print(f"Batch size: {PRECLASSIFY_BATCH_SIZE} biomarkers per call")
            print(f"Total batches: {num_batches}")
            print(f"Parallel execution: {MAX_CONCURRENT_PRECLASSIFY} batches at once")
            print(
                f"Estimated time: ~{num_batches // MAX_CONCURRENT_PRECLASSIFY * 2}-{num_batches // MAX_CONCURRENT_PRECLASSIFY * 4} minutes\n"
            )

            # Create batches
            preclassify_batches = []
            for batch_idx in range(
                0, len(biomarkers_to_preclassify), PRECLASSIFY_BATCH_SIZE
            ):
                batch = biomarkers_to_preclassify[
                    batch_idx : batch_idx + PRECLASSIFY_BATCH_SIZE
                ]
                preclassify_batches.append(batch)

            def process_preclassify_batch(batch_info):
                """Process a single preclassify batch"""
                batch_num, batch = batch_info
                batch_id = batch_num + 1

                try:
                    prompt = cheap_preclassify_batch_prompt(batch)
                    text = gemini_generate_text(client, FAST_MODEL, prompt)
                    parsed = parse_json_loose(text)

                    if not isinstance(parsed, list):
                        parsed = [parsed] if isinstance(parsed, dict) else []

                    # Match results to biomarkers
                    results = []
                    for bio_identifier, bio_name, df_idx in batch:
                        matched = None
                        for result in parsed:
                            if isinstance(result, dict):
                                result_id = str(result.get("identifier", "")).strip()
                                if (
                                    bio_identifier in result_id
                                    or result_id in bio_identifier
                                ):
                                    matched = result
                                    break

                        if matched:
                            pre = coerce_preclassify(matched)
                            results.append((df_idx, pre, None))
                        else:
                            # No match found
                            results.append((df_idx, None, f"No match for {bio_name}"))

                    return (batch_id, results, None)

                except Exception as e:
                    # All biomarkers in batch failed
                    failed_results = [(bio[2], None, str(e)) for bio in batch]
                    return (batch_id, failed_results, str(e))

            # Process batches in parallel
            completed = 0
            with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_PRECLASSIFY) as executor:
                future_to_batch = {
                    executor.submit(process_preclassify_batch, (i, batch)): i
                    for i, batch in enumerate(preclassify_batches)
                }

                with tqdm(
                    total=len(preclassify_batches), desc="Stage A: batched preclassify"
                ) as pbar:
                    for future in as_completed(future_to_batch):
                        batch_id, results, error = future.result()
                        completed += 1

                        # Update dataframe
                        for df_idx, pre_data, err_msg in results:
                            if pre_data:
                                df.loc[df_idx, "pre_dbs_testable"] = pre_data[
                                    "pre_dbs_testable"
                                ]
                                df.loc[df_idx, "pre_confidence"] = float(
                                    pre_data["pre_confidence"]
                                )
                                df.loc[df_idx, "pre_likely_method"] = pre_data[
                                    "pre_likely_method"
                                ]
                                df.loc[df_idx, "pre_short_reason"] = pre_data[
                                    "pre_short_reason"
                                ]
                            else:
                                # Error case
                                df.loc[df_idx, "pre_dbs_testable"] = "Unclear"
                                df.loc[df_idx, "pre_confidence"] = 0.0
                                df.loc[df_idx, "pre_likely_method"] = "unknown"
                                df.loc[df_idx, "pre_short_reason"] = (
                                    f"Error: {err_msg[:100] if err_msg else 'unknown'}"
                                )

                        # Checkpoint every 20 batches
                        if completed % 20 == 0:
                            df.to_csv(CHECKPOINT_CSV, index=False)

                        pbar.update(1)

        df.to_csv(CHECKPOINT_CSV, index=False)
        print("\n✅ Stage A completed. Checkpoint saved.")
    else:
        print("\n⏭️  Skipping Stage A (preclassify) as requested.")

    # Stage B: Batched Deep Research for uncertain cases
    print("\n" + "=" * 60)
    print("STAGE B: Batched Deep Research for Uncertain Cases")
    print("=" * 60)

    # First, copy trusted preclassify results to final fields
    print("Copying high-confidence preclassify results to final fields...")
    for i in range(len(df)):
        already = str(df.loc[i, "research_ran"]).strip().lower() == "true"
        if already:
            continue

        pre_dbs = str(df.loc[i, "pre_dbs_testable"]).strip()
        try:
            pre_conf = float(df.loc[i, "pre_confidence"])
            if pd.isna(pre_conf):
                pre_conf = 0.0
        except (ValueError, TypeError):
            pre_conf = 0.0

        needs_research = (pre_dbs == "Unclear") or (
            pre_conf < SEND_TO_RESEARCH_IF_CONFIDENCE_BELOW
        )

        if not needs_research:
            # If we trust preclassify, copy into final fields
            df.loc[i, "dbs_testable"] = pre_dbs
            df.loc[i, "confidence"] = float(pre_conf)
            df.loc[i, "best_method"] = str(df.loc[i, "pre_likely_method"])
            df.loc[i, "sample_prep_notes"] = str(df.loc[i, "pre_short_reason"])
            df.loc[i, "research_ran"] = "false"

    # Collect biomarkers that need deep research
    # Include: empty research_ran, "failed", "partial" status (retry these)
    # Exclude: "true" (completed with valid data) and "false" (high confidence, no research needed)
    biomarkers_needing_research = []
    failed_count = 0
    partial_count = 0
    skipped_completed = 0
    for i in range(len(df)):
        research_status = str(df.loc[i, "research_ran"]).strip().lower()

        # Skip high-confidence (no research needed)
        if research_status == "false":
            continue

        # Skip completed entries ONLY if they have valid data
        if research_status == "true":
            # Verify there's actual data, not just the flag
            dbs_val = str(df.loc[i, "dbs_testable"]).strip()
            if dbs_val in ["Yes", "No", "Unclear"] and dbs_val != "":
                skipped_completed += 1
                continue
            # Otherwise, re-run this one (data might be missing)

        # Count retries
        if research_status == "failed":
            failed_count += 1
        elif research_status == "partial":
            partial_count += 1

        identifier = normalize_identifier(df.loc[i, "Identifier"])
        name = normalize_name(df.loc[i, "Biomarker Name"])
        biomarkers_needing_research.append((identifier, name, i))

    if skipped_completed > 0:
        print(
            f"✅ Skipping {skipped_completed} biomarkers with completed deep research"
        )

    if failed_count > 0 or partial_count > 0:
        print(
            f"ℹ️  Retrying: {failed_count} failed + {partial_count} partial from previous run"
        )

    research_needed = len(biomarkers_needing_research)
    num_batches = (research_needed + BIOMARKERS_PER_BATCH - 1) // BIOMARKERS_PER_BATCH

    print(f"Biomarkers requiring deep research: {research_needed}")
    print(f"Batch size: {BIOMARKERS_PER_BATCH} biomarkers per Deep Research call")
    print(f"Total batches: {num_batches}")
    print(f"Parallel execution: {MAX_CONCURRENT_RESEARCH} batches at once")
    print(
        f"Estimated batches to process: {research_needed}/{BIOMARKERS_PER_BATCH} = {num_batches} batches"
    )
    print(
        f"Estimated cost: ${num_batches * 3:.2f} (much cheaper than individual calls)"
    )
    print(f"Confidence threshold: {SEND_TO_RESEARCH_IF_CONFIDENCE_BELOW}")
    print()

    if research_needed == 0:
        print("✅ No biomarkers need deep research!")
    else:
        # Create batches
        batches = []
        for batch_idx in range(0, research_needed, BIOMARKERS_PER_BATCH):
            batch = biomarkers_needing_research[
                batch_idx : batch_idx + BIOMARKERS_PER_BATCH
            ]
            batches.append(batch)

        print(
            f"Processing {len(batches)} batches with {MAX_CONCURRENT_RESEARCH} concurrent tasks..."
        )
        print(
            f"⚠️  Deep Research API limit: 10 RPM. Using {DEEP_RESEARCH_BATCH_DELAY}s delay between batch groups.\n"
        )

        def process_batch(batch_info):
            """Process a single batch of biomarkers"""
            batch_num, batch = batch_info
            batch_id = batch_num + 1

            try:
                # Create batched prompt
                prompt = deep_research_batch_prompt(batch)

                # Run deep research for the batch
                report = deep_research_batch_run(client, prompt, batch_id, len(batch))

                # Extract structured data for all biomarkers in batch
                extraction_prompt = extract_structured_from_batch_report_prompt(
                    report, batch
                )
                extraction_text = gemini_generate_text(
                    client, STRUCTURED_MODEL, extraction_prompt
                )

                # Parse JSON array
                parsed = parse_json_loose(extraction_text)
                if not isinstance(parsed, list):
                    print(f"   ⚠️  Batch {batch_id}: Expected array, got {type(parsed)}")
                    parsed = [parsed] if isinstance(parsed, dict) else []

                # Match results to biomarkers
                results = []
                for bio_identifier, bio_name, df_idx in batch:
                    # Find matching result
                    matched_result = None
                    for result in parsed:
                        if isinstance(result, dict):
                            result_id = str(result.get("identifier", "")).strip()
                            if (
                                bio_identifier in result_id
                                or result_id in bio_identifier
                            ):
                                matched_result = result
                                break

                    if matched_result:
                        results.append((df_idx, matched_result, report, "true"))
                    else:
                        # Biomarker not found in results - mark as incomplete
                        print(f"   ⚠️  Batch {batch_id}: No result for {bio_name}")
                        results.append(
                            (
                                df_idx,
                                {},
                                f"Partial report (batch {batch_id})",
                                "partial",
                            )
                        )

                return (batch_id, results, None)

            except Exception as e:
                error_msg = str(e)
                # Check if it's a rate limit error
                if "429" in error_msg or "quota" in error_msg.lower():
                    print(
                        f"\n⏳ Batch {batch_id}: Rate limited, will retry after backoff..."
                    )
                else:
                    print(f"\n❌ Batch {batch_id} failed: {e}")
                # Return failure info for all biomarkers in this batch
                failed_results = [
                    (bio[2], {}, f"ERROR: {error_msg}", "failed") for bio in batch
                ]
                return (batch_id, failed_results, error_msg)

        def update_df_with_results(results_list):
            """Update dataframe with batch results"""
            for df_idx, structured_data, report_text, status in results_list:
                df.loc[df_idx, "deep_research_report"] = str(report_text)[:50000]
                df.loc[df_idx, "research_ran"] = status

                if structured_data:
                    final = coerce_final(structured_data)
                    df.loc[df_idx, "dbs_testable"] = final["dbs_testable"]
                    df.loc[df_idx, "confidence"] = float(final["confidence"])
                    df.loc[df_idx, "best_method"] = final["best_method"]
                    df.loc[df_idx, "sample_prep_notes"] = final["sample_prep_notes"]
                    df.loc[df_idx, "limitations"] = final["limitations"]
                    df.loc[df_idx, "evidence"] = final["evidence"]
                else:
                    df.loc[df_idx, "dbs_testable"] = "Unclear"
                    df.loc[df_idx, "confidence"] = 0.0
                    df.loc[df_idx, "best_method"] = "unknown"

        # Process batches in groups to respect rate limits
        # Submit MAX_CONCURRENT_RESEARCH at a time, wait for completion, then next group
        completed_batches = 0
        batch_groups = [
            batches[i : i + MAX_CONCURRENT_RESEARCH]
            for i in range(0, len(batches), MAX_CONCURRENT_RESEARCH)
        ]

        print(
            f"Split into {len(batch_groups)} groups of up to {MAX_CONCURRENT_RESEARCH} batches each.\n"
        )

        with tqdm(total=len(batches), desc="Processing batches") as pbar:
            for group_idx, batch_group in enumerate(batch_groups):
                group_start_idx = group_idx * MAX_CONCURRENT_RESEARCH

                # Add delay between groups (except first one) to avoid rate limits
                if group_idx > 0:
                    print(
                        f"\n⏳ Waiting {DEEP_RESEARCH_BATCH_DELAY}s before next group to respect API rate limits..."
                    )
                    time.sleep(DEEP_RESEARCH_BATCH_DELAY)

                print(
                    f"\n📦 Starting group {group_idx + 1}/{len(batch_groups)} ({len(batch_group)} batches)"
                )

                # Submit this group
                with ThreadPoolExecutor(
                    max_workers=MAX_CONCURRENT_RESEARCH
                ) as executor:
                    future_to_batch = {
                        executor.submit(process_batch, (group_start_idx + i, batch)): i
                        for i, batch in enumerate(batch_group)
                    }

                    # Process as they complete - SAVE IMMEDIATELY after each batch
                    for future in as_completed(future_to_batch):
                        batch_id, results, error = future.result()
                        completed_batches += 1

                        # Update dataframe with results
                        update_df_with_results(results)

                        # SAVE IMMEDIATELY after each batch completes (not waiting for group)
                        df.to_csv(CHECKPOINT_CSV, index=False)
                        print(
                            f"   💾 Batch {batch_id} saved to checkpoint ({completed_batches}/{len(batches)} total)"
                        )

                        pbar.update(1)

        print(f"\n✅ Completed all {len(batches)} batches!")

    # Copy results from working columns to original columns
    print("\n📋 Copying results to original columns...")
    for i in range(len(df)):
        # Copy dbs_testable -> Testable by DBS
        dbs_val = str(df.loc[i, "dbs_testable"]).strip()
        if dbs_val and dbs_val not in ["", "nan", "None"]:
            df.loc[i, "Testable by DBS"] = dbs_val

        # Copy best_method -> Best Method
        method_val = str(df.loc[i, "best_method"]).strip()
        if method_val and method_val not in ["", "nan", "None"]:
            df.loc[i, "Best Method"] = method_val

        # Copy confidence -> Confidence(0-1)
        conf_val = df.loc[i, "confidence"]
        try:
            conf_float = float(conf_val)
            if not pd.isna(conf_float):
                df.loc[i, CONFIDENCE_COL] = conf_float
        except (ValueError, TypeError):
            pass

    # Save full checkpoint with all columns (for potential resume)
    df.to_csv(CHECKPOINT_CSV, index=False)

    # Create clean output with only the 6 original columns
    original_columns = [
        "S. No.",
        "Identifier",
        "Biomarker Name",
        "Testable by DBS",
        "Best Method",
        CONFIDENCE_COL,
    ]
    # Ensure the columns exist before selecting (avoid KeyError if input was missing them)
    for col in original_columns:
        if col not in df.columns:
            df[col] = ""
    df_clean = df[original_columns].copy()
    df_clean.to_csv(args.output, index=False)

    print(f"✅ Clean output saved with {len(original_columns)} columns")

    # Final statistics
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)

    total = len(df)
    testable_yes = sum(
        1 for i in range(len(df)) if str(df.loc[i, "dbs_testable"]).strip() == "Yes"
    )
    testable_no = sum(
        1 for i in range(len(df)) if str(df.loc[i, "dbs_testable"]).strip() == "No"
    )
    testable_unclear = sum(
        1 for i in range(len(df)) if str(df.loc[i, "dbs_testable"]).strip() == "Unclear"
    )
    researched = sum(
        1
        for i in range(len(df))
        if str(df.loc[i, "research_ran"]).strip().lower() == "true"
    )
    failed = sum(
        1
        for i in range(len(df))
        if str(df.loc[i, "research_ran"]).strip().lower() == "failed"
    )

    print(f"Total biomarkers processed: {total:,}")
    print(f"\nTestability Results:")
    print(f"  ✅ Testable (Yes): {testable_yes:,} ({testable_yes/total*100:.1f}%)")
    print(f"  ❌ Not Testable (No): {testable_no:,} ({testable_no/total*100:.1f}%)")
    print(f"  ❓ Unclear: {testable_unclear:,} ({testable_unclear/total*100:.1f}%)")
    print(f"\nDeep Research:")
    print(f"  Completed: {researched:,}")
    if failed > 0:
        print(f"  Failed: {failed:,} (review checkpoint_partial.csv)")
    print(f"\n📄 Clean output saved to: {args.output}")
    print(f"💾 Full checkpoint saved to: {CHECKPOINT_CSV}")
    print("=" * 60)


if __name__ == "__main__":
    main()
