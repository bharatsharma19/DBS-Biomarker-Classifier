# Biomarker DBS Testability Classification

Automated classification system to determine if biomarkers are testable from Dried Blood Spots (DBS) using Google Gemini AI with Deep Research capabilities.

## Overview

This tool processes large biomarker datasets (6K+) and classifies each biomarker for:
- **DBS Testability**: Yes / No / Unclear
- **Best Analytical Method**: LC-MS/MS, immunoassay, enzymatic, etc.
- **Technical Considerations**: Stability, matrix effects, sample prep requirements
- **Evidence Quality**: Citations and validation studies

## Two-Stage Processing

### Stage A: Batched Fast Preclassification ⚡ SUPER FAST
- Uses `gemini-2.5-flash` for rapid initial assessment
- **BATCHED PROCESSING**: 40 biomarkers per API call
- **PARALLEL EXECUTION**: 25 concurrent batches = 1,000 biomarkers simultaneously
- Processes ~6,000 biomarkers in **~5-10 minutes** (100x faster!)
- Cost: ~$0.20 total (batching reduces API calls by 40x)
- Based on identifier type (CHEBI, UniProt, ENSEMBL) and general biomarker properties

### Stage B: Batched Deep Research (for uncertain cases) ⚡ OPTIMIZED
- Uses `deep-research-pro-preview-12-2025` agent
- Triggered when confidence < 0.55 or result is "Unclear"
- **BATCHED PROCESSING**: 25 biomarkers per Deep Research call (default)
- **RATE LIMIT AWARE**: Deep Research has a strict ~10 RPM limit; default concurrency is set conservatively
- ~5-15 minutes per batch (25 biomarkers)
- Cost: ~$3 per batch (25 biomarkers) = **$0.12 per biomarker** (still dramatically cheaper than 1-by-1)
- Produces detailed reports with evidence for all biomarkers in batch

**Speed Improvement**: 
- Old (sequential): 6,000 biomarkers × 10 min = 1,000 hours (42 days!)
- New (batched + parallel): 6,000 ÷ 1,000 × 10 min = **~1 hour** 🚀 (1000x faster!)

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY (preferred) or GOOGLE_API_KEY
```

On Windows PowerShell, use:

```powershell
Copy-Item .env.example .env
```

## Usage

### Basic Run (Full Pipeline)

```bash
python main.py --input biomarkers.csv --output results.csv
```

Tip: `biomarkers.csv` is intentionally ignored (often not shareable). Start by copying `biomarkers.sample.csv`:

```bash
cp biomarkers.sample.csv biomarkers.csv
```

On Windows PowerShell:

```powershell
Copy-Item biomarkers.sample.csv biomarkers.csv
```

### Dry Run (Estimate costs without API calls)

```bash
python main.py --input biomarkers.csv --output results.csv --dry-run
```

### Resume from Checkpoint

```bash
python main.py --input biomarkers.csv --output results.csv --resume
```

### Skip Preclassification (Only run deep research)

```bash
python main.py --input biomarkers.csv --output results.csv --skip-preclassify
```

## Input Format

CSV with columns (minimum required):
- `Identifier`: `CHEBI:XXXXX`, `UniProt:PXXXXX`, `ENSEMBL:ENST...`
- `Biomarker Name`: Full name

Optional (recommended):
- `S. No.`: Serial number

See `biomarkers.sample.csv` for an example template.

## Output Format

Original columns plus:

### Preclassification Results
- `pre_dbs_testable`: Yes/No/Unclear
- `pre_confidence`: 0.0-1.0
- `pre_likely_method`: LC-MS/MS, immunoassay, etc.
- `pre_short_reason`: Brief explanation

### Final Results (after deep research if triggered)
- `dbs_testable`: Yes/No/Unclear
- `confidence`: 0.0-1.0
- `best_method`: Validated analytical method
- `sample_prep_notes`: Extraction and preparation details
- `limitations`: JSON array of technical limitations
- `evidence`: JSON array of cited sources
- `research_ran`: true/false/failed
- `deep_research_report`: Full research report (if triggered)

## Configuration

Edit `main.py` constants (top of file):

```python
FAST_MODEL = "gemini-2.5-flash"  # Preclassification model
STRUCTURED_MODEL = "gemini-2.5-flash"  # Extraction model
DEEP_RESEARCH_AGENT = "deep-research-pro-preview-12-2025"  # Research agent

POLL_SECONDS = 10  # How often to check research status
MAX_POLL_MINUTES = 90  # Timeout for batched deep research

SEND_TO_RESEARCH_IF_CONFIDENCE_BELOW = 0.55  # Threshold for triggering research

# Batching configuration (NEW - for EXTREME speedup!)
PRECLASSIFY_BATCH_SIZE = 40  # Process 40 biomarkers per preclassify call
MAX_CONCURRENT_PRECLASSIFY = 25  # Run 25 batches in parallel = 1000 biomarkers at once

BIOMARKERS_PER_BATCH = 25  # Process 25 biomarkers in one Deep Research call
MAX_CONCURRENT_RESEARCH = 8  # Keep under Deep Research rate limits
DEEP_RESEARCH_BATCH_DELAY = 15  # Seconds between starting batch groups
```

### Cost Optimization (with EXTREME Batching!)

**Conservative (default)**: `SEND_TO_RESEARCH_IF_CONFIDENCE_BELOW = 0.55`
- ~20-30% of biomarkers trigger deep research (1,200-1,800 biomarkers)
- Batch size: `BIOMARKERS_PER_BATCH = 25` (default)
- Approx batches: ~48-72 batches (25 biomarkers each)
- Cost: **~$144-216** (still far cheaper than 1-by-1)

**Moderate**: `SEND_TO_RESEARCH_IF_CONFIDENCE_BELOW = 0.65`
- ~35-45% trigger deep research (2,100-2,700 biomarkers)
- Approx batches: ~84-108 batches
- Cost: **~$252-324**

**Thorough**: `SEND_TO_RESEARCH_IF_CONFIDENCE_BELOW = 0.75`
- ~50-60% trigger deep research (3,000-3,600 biomarkers)
- Approx batches: ~120-144 batches
- Cost: **~$360-432**

**Batching Advantage**: Processing a whole batch costs ~$3 per batch instead of ~$3 per biomarker (when done individually) = **large cost reduction**.

## DBS Testability Criteria

The system evaluates based on:

### Small Molecules (CHEBI)
- ✅ YES if: Stable at room temp, detectable in blood, MW < 2000 Da
- ❌ NO if: Volatile, rapidly degraded, plasma-only
- Method: Typically LC-MS/MS

### Proteins (UniProt)
- ✅ YES if: Abundant (>1 µg/mL), stable when dried, validated DBS assays
- ❌ NO if: Very large (>300 kDa), extremely low abundance
- ❓ UNCLEAR if: Moderate abundance with no validation
- Method: Typically immunoassay (ELISA, Luminex)

### Nucleic Acids (ENSEMBL)
- ✅ YES if: DNA/RNA extraction from DBS is established
- Method: PCR, qPCR, sequencing

### Key Technical Factors
1. **Stability**: Room temperature for days-weeks
2. **Hematocrit Effect**: Measurement consistency across blood cell concentrations
3. **Matrix Effects**: Interference from dried blood components
4. **Recovery**: Extraction efficiency from filter paper
5. **Sensitivity**: LOD/LOQ achievable from 15-50 µL blood

## Checkpointing & Resume

The system automatically saves progress:
- **checkpoint_partial.csv**:
  - Stage A: periodically (and at the end of Stage A)
  - Stage B: after each Deep Research batch completes
- Use `--resume` flag to continue from checkpoint
- Failed deep research marked as "failed" for manual review

## Error Handling

- Automatic retry with exponential backoff (5 attempts for preclassify, 4 for deep research batches)
- Failed operations logged with error messages
- Checkpoint saved before errors to prevent data loss
- Use `research_ran = "failed"` to identify items needing manual review

## Performance

### Expected Runtime (with EXTREME Batching & Parallelization!)
- **Stage A**: 5-10 minutes for 6K biomarkers (1000 at once!) ⚡⚡⚡
- **Stage B**: 5-15 minutes per batch (25 biomarkers) with conservative parallelism (default: 8 concurrent)
  - Total wall time depends heavily on Deep Research availability and rate limits; expect **several hours** for 6K-scale runs.

**Speed Comparison**:
- Old (1 at a time): 6,000 biomarkers × 10 min = 1,000 hours (42 days!)
- New (1000 at once): 6,000 ÷ 1000 × 10 min = **~1 hour** 🚀🚀🚀 (1000x faster!)

**Total Processing Time for 6K biomarkers**: **4-11 hours** (instead of 42 days!)

### API Rate Limits
- Gemini API limits vary by tier/model.
- Stage A batching: 1 request per `PRECLASSIFY_BATCH_SIZE` biomarkers (default: 40)
- Deep Research: strict ~10 RPM limit; the code defaults to `MAX_CONCURRENT_RESEARCH = 8` and staggers groups with `DEEP_RESEARCH_BATCH_DELAY`.

## Validation

The Deep Research agent:
1. Searches PubMed, scientific journals, FDA/EMA databases
2. Evaluates stability studies and DBS validation papers
3. Assesses technical feasibility based on analyte properties
4. Provides citations for all major claims
5. Marks "Unclear" when no direct DBS validation exists

## Troubleshooting

### "Missing GEMINI_API_KEY"
- Ensure `.env` file exists with `GEMINI_API_KEY=your_key_here` (or `GOOGLE_API_KEY=your_key_here`)
- Get API key from: https://aistudio.google.com/apikey

### "Model not found" errors
- Verify model names are correct (as of Feb 2026)
- Check API access level (some models require paid tier)

### Deep Research timeouts
- Increase `MAX_POLL_MINUTES` in config
- Check internet connection stability
- Review Deep Research agent availability

### High costs
- Run `--dry-run` first to estimate
- Lower `SEND_TO_RESEARCH_IF_CONFIDENCE_BELOW` threshold
- Process in batches with manual review between

## Output Analysis

### Review Priorities
1. Check `research_ran = "failed"` entries
2. Review `confidence < 0.3` entries for manual validation
3. Verify `dbs_testable = "Unclear"` have adequate justification
4. Spot-check `evidence` citations for quality

### Export for Review
```python
import pandas as pd
df = pd.read_csv('results.csv')

# High-confidence positives
testable = df[(df['dbs_testable'] == 'Yes') & (df['confidence'] > 0.7)]

# Needs manual review
review = df[(df['confidence'] < 0.4) | (df['research_ran'] == 'failed')]
```

## Export Clean Results (from checkpoint)

If you want to export the latest 6-column output from an in-progress run:

```bash
python export_clean.py --checkpoint checkpoint_partial.csv --output results_clean.csv
```

## License

MIT License (see `LICENSE`).

## Last Updated

February 2026
