# Counter Verification Workflow

This document describes the counter verification system for quality control of counter-to-OSM-edge assignments.

## Overview

The verification system helps identify and correct incorrect OSM way assignments for bicycle counting stations (Dauerzählstellen). It provides an interactive web interface for manual verification with persistent storage of verified assignments.

## Architecture

```
┌─────────────────┐
│  ETL Pipeline   │
│ (verification)  │
└────────┬────────┘
         │
         ├─→ Calculate DTV/volume ratios
         ├─→ Flag outliers (IQR method)
         └─→ Export verification data
                  │
                  ├─→ MBTiles (traffic volumes)
                  ├─→ GeoJSON (counter points)
                  └─→ JSON (metadata)
                           │
                           ▼
                 ┌───────────────────┐
                 │ Verification UI   │
                 │ (MapLibre + JS)   │
                 └─────────┬─────────┘
                           │
                           ▼
              ┌──────────────────────────┐
              │ Verified Counter Dataset │
              │  (per-year CSV, git)     │
              └──────────────────────────┘
                           │
                           ▼
                 ┌─────────────────┐
                 │  ETL Pipeline   │
                 │  (production)   │
                 └─────────────────┘
```

## Key Concepts

### 1. **Persistent Verified Datasets**

Counter-to-OSM assignments are **stationary** (counters don't move), so we verify them once per year and reuse:

```
data/
  verified/
    counters_verified_2023.csv  # Git-tracked
    counters_verified_2024.csv
    counters_verified_2025.csv
```

Each year's dataset includes:
- `counter_id`: Counter identifier
- `base_id`: OSM way ID (verified)
- `count`: Bicycle volume on that edge (verified)
- `bicycle_infrastructure`: Infrastructure type
- `verification_status`: `auto | verified | carryover | unverified`
- `verification_metadata`: Human-readable notes
- `osm_source`: OSM data version (e.g., `osm-2023-01-01`)

### 2. **ETL Modes**

#### Production Mode (default)
```bash
uv run hochrechnung etl --config config/hessen_2023.yaml
```

- Loads verified counter dataset if available
- Skips spatial matching (faster!)
- Uses locked `base_id` and `count` values
- Fresh DTV calculated from latest measurements

#### Verification Mode
```bash
uv run hochrechnung etl --config config/hessen_2023.yaml --mode verification
```

- Performs spatial matching (auto-assigns OSM ways)
- Calculates DTV/volume ratios
- Flags outliers using IQR method
- Exports verification data for UI

### 3. **Outlier Detection**

Uses **IQR (Interquartile Range)** method:

```
Q1 = 25th percentile of ratio
Q3 = 75th percentile of ratio
IQR = Q3 - Q1

Lower bound = Q1 - 1.5 × IQR
Upper bound = Q3 + 1.5 × IQR

Flagged if: ratio < lower OR ratio > upper
```

**Severity levels:**
- `critical`: Ratio > 2× upper threshold (very suspicious)
- `warning`: Outlier but less extreme
- `ok`: Within normal range
- `verified`: Human-verified

### 4. **MBTiles Generation**

To keep file size manageable, tiles include only edges around flagged counters:

```python
# Buffer 250m around each flagged counter
buffer_area = flagged_counters.buffer(250)

# Filter edges to buffer area
relevant_edges = edges[edges.intersects(buffer_area)]

# Generate tiles with tippecanoe
tippecanoe -o volumes.mbtiles -Z10 -z16 relevant_edges.geojson
```

**Result:** 5-20 MB (vs 500+ MB for full coverage)

## Workflow

### Initial Verification (First Year)

1. **Run ETL in verification mode:**
   ```bash
   uv run hochrechnung etl --config config/hessen_2023.yaml --mode verification
   ```

   This:
   - Auto-matches counters to nearest OSM edges
   - Calculates DTV/volume ratios
   - Flags outliers

2. **Launch verification UI:**
   ```bash
   uv run hochrechnung verify --config config/hessen_2023.yaml
   ```

   Opens web UI at `http://localhost:8000`

3. **Verify counters in UI:**
   - Click flagged counter in sidebar
   - Map zooms to counter location
   - Click on correct edge (or enter OSM ID manually)
   - Add notes explaining the correction
   - Click "Save"

4. **Review and commit:**
   ```bash
   git add data/verified/counters_verified_2023.csv
   git commit -m "Verify 2023 counters: corrected 12 outliers"
   ```

### Production Runs (After Verification)

```bash
# Just run ETL in production mode
uv run hochrechnung etl --config config/hessen_2023.yaml

# It automatically loads verified counters
# No spatial matching needed!
```

### Next Year (Reuse Verified Assignments)

**Option 1: Manual carry-over**

```python
from hochrechnung.verification.persistence import init_verified_counters_from_previous_year

# Bootstrap 2024 from 2023
new_verified = init_verified_counters_from_previous_year(
    data_root=Path("data"),
    current_year=2024,
    previous_year=2023,
    new_traffic_volumes=traffic_2024
)
```

This inherits `base_id` assignments but updates `count` from new STADTRADELN data.

**Option 2: Re-verify from scratch**

```bash
# Run verification workflow for new year
uv run hochrechnung verify --config config/hessen_2024.yaml
```

Most counters will show `carryover` status (inherited from 2023). Only review if:
- Ratio changed significantly
- New counters added
- OSM data updated

## Verification UI

### Layout

```
┌─────────────────────────────────────────────────────┐
│  Counter Verification - Year 2023                    │
│  Total: 50  Flagged: 12  Median Ratio: 2.3         │
└─────────────────────────────────────────────────────┘
┌─────────────┬───────────────────────────────────────┐
│ Counter     │                                       │
│ List        │                                       │
│             │                                       │
│ ┌─────────┐ │          Map View                     │
│ │ 071     │ │    (MapLibre + MBTiles)              │
│ │ Critical│ │                                       │
│ │ DTV: 850│ │                                       │
│ │ Vol: 120│ │                                       │
│ │ Ratio:7 │ │                                       │
│ └─────────┘ │                                       │
│             │                                       │
│ ┌─────────┐ │                                       │
│ │ 064b    │ │                                       │
│ │ Warning │ │                                       │
│ └─────────┘ │                                       │
├─────────────┤                                       │
│ Edit 071    │                                       │
│ ┌─────────┐ │                                       │
│ │OSM ID   │ │                                       │
│ │[67890]  │ │                                       │
│ │         │ │                                       │
│ │Volume   │ │                                       │
│ │[450]    │ │                                       │
│ │         │ │                                       │
│ │Notes    │ │                                       │
│ │[Correc..│ │                                       │
│ │         │ │                                       │
│ │[Save]   │ │                                       │
│ └─────────┘ │                                       │
└─────────────┴───────────────────────────────────────┘
```

### Features

**Counter List:**
- Sorted by severity (critical → warning → ok)
- Color-coded borders
- Click to select and edit

**Map:**
- Traffic volume layer (MBTiles)
- Counter markers (color = severity)
- Tooltips on hover (show DTV, volume, ratio)
- Click edge to auto-fill OSM ID and volume

**Editor:**
- OSM Way ID: Auto-filled from map click or manual entry
- Volume: Auto-filled from edge or manual entry
- Notes: Free-text explanation

**Persistence:**
- Changes saved immediately to CSV
- Git-tracked for history
- Atomic updates (no partial saves)

## Data Flow

### Verification Mode

```
Counter Locations + Measurements
         │
         ├─→ Calculate DTV
         ├─→ Spatial matching (auto)
         ├─→ Calculate ratios
         └─→ Flag outliers
                  │
                  ▼
         Verification Data
         (counters + outlier flags)
                  │
                  ├─→ Generate MBTiles
                  ├─→ Export JSON
                  └─→ Launch UI
                          │
                          ▼
                  User verifies/corrects
                          │
                          ▼
              Save verified_counters_{year}.csv
```

### Production Mode

```
Counter Measurements
         │
         └─→ Calculate DTV (fresh)
                  │
                  ▼
         Load verified_counters_{year}.csv
         (pre-verified base_id + count)
                  │
                  ├─→ Join with DTV
                  ├─→ Feature engineering
                  └─→ Training data
```

## File Structure

```
hochrechnung/
├── data/
│   └── verified/
│       ├── counters_verified_2023.csv  # Git-tracked
│       └── counters_verified_2024.csv
│
├── cache/
│   └── verification/
│       └── 2023/
│           ├── verification_data.json
│           ├── volumes.mbtiles
│           └── counters.geojson
│
└── src/hochrechnung/
    └── verification/
        ├── __init__.py
        ├── persistence.py      # Load/save verified datasets
        ├── outliers.py         # Ratio calculation + flagging
        ├── tiles.py            # MBTiles generation
        ├── export.py           # JSON export
        ├── server.py           # HTTP server
        └── static/
            ├── index.html      # UI template
            └── app.js          # Frontend logic
```

## Edge Cases

### 1. Wrong OSM Way Assignment

**Symptom:** Ratio extremely high (e.g., 7.0)

**Cause:** Counter assigned to parallel main road instead of bike lane

**Solution:**
1. Click counter in UI
2. Map zooms to location
3. Click on correct bike lane edge
4. OSM ID and volume auto-fill
5. Add note: "Corrected: bike lane not main road"
6. Save

### 2. Parallel Edges Divide Volume

**Symptom:** Ratio high, but two edges nearby

**Example:**
- DTV: 1000 bikes/day
- Bike lane edge: 80 bikes/day
- Main road edge: 250 bikes/day
- Total: 330 bikes/day (should be assigned to counter)

**Solution:**
1. Click counter
2. Enter OSM ID of primary edge (e.g., bike lane)
3. Manually enter volume: `330` (sum of both)
4. Add note: "Manual sum: bike lane 80 + main road 250"
5. Save

### 3. Very Low Bicycle Volume

**Symptom:** Ratio high, but assigned edge genuinely has low volume

**Cause:** Counter placed on busy route, but STADTRADELN had few participants

**Solution:**
- Verify OSM assignment is correct
- If correct, mark as verified anyway
- Add note: "Low GPS coverage, assignment correct"

### 4. New Counters (Not in Previous Year)

**Status:** `unverified`

**Workflow:**
1. Run verification mode
2. Auto-matched to nearest edge
3. Check ratio
4. Verify or correct as needed

### 5. OSM Data Changed

If OSM ways were modified between years:

```bash
# Re-run verification from scratch
uv run hochrechnung verify --config config/hessen_2024.yaml

# Counters will show mismatches
# Review and correct as needed
```

## Tips

### Efficient Verification

1. **Sort by severity:** Critical counters first
2. **Use map clicks:** Faster than typing OSM IDs
3. **Keep notes concise:** "Bike lane not road" is sufficient
4. **Batch similar corrections:** Commit multiple counters together

### Quality Checks

```bash
# Check verification status
grep -c "verified" data/verified/counters_verified_2023.csv

# See what changed
git diff data/verified/counters_verified_2023.csv

# View verification history
git log -p -- data/verified/counters_verified_2023.csv
```

### Performance

- MBTiles generation: ~10-30 seconds for 50 flagged counters
- UI load time: <2 seconds
- Save operation: ~100ms per counter

## Troubleshooting

### "tippecanoe not found"

```bash
# macOS
brew install tippecanoe

# Debian/Ubuntu
apt install tippecanoe

# Or use --prepare-only to skip MBTiles
uv run hochrechnung verify --config config.yaml --prepare-only
```

### "No verified counter dataset found"

First run for this year. Expected! Run verification workflow.

### "Verification UI shows no tiles"

MBTiles generation failed. UI still works, but map won't show traffic volumes.
Use QGIS to inspect edges manually if needed.

## Best Practices

1. **Verify once per year:** Don't skip this step!
2. **Commit early, commit often:** Commit after verifying each batch
3. **Document corrections:** Use the notes field
4. **Review before production:** Check git diff before running training
5. **Backup verified datasets:** They're your source of truth

## Future Enhancements

Potential improvements:

- [ ] Bulk edit (select multiple counters)
- [ ] Undo/redo changes
- [ ] Export verification report (PDF)
- [ ] Automatic parallel edge detection
- [ ] Machine learning to suggest corrections
- [ ] Multi-user collaboration

## References

- [IQR Outlier Detection](https://en.wikipedia.org/wiki/Interquartile_range)
- [MapLibre GL JS](https://maplibre.org/)
- [MBTiles Specification](https://github.com/mapbox/mbtiles-spec)
- [Tippecanoe](https://github.com/felt/tippecanoe)
