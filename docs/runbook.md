# Runbook

## Prerequisites

### System Requirements
- Python 3.11+
- uv package manager
- 16GB RAM (for full Hessen processing)
- 50GB disk space (for data + cache)

### Installation

#### Windows 10/11

```powershell
# Install uv (if not already installed)
pip install uv

# IMPORTANT: After installing uv, restart your terminal
# or refresh PATH manually:
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

# Navigate to project
cd C:\Users\Porojkow\Documents\Projekte\VV01_Hessen\hochrechnung

# Install dependencies
uv sync

# Install pre-commit hooks
uv run pre-commit install
```

#### Linux/macOS

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Navigate to project
cd /path/to/hochrechnung

# Install dependencies
uv sync

# Install pre-commit hooks
uv run pre-commit install
```

---

## Running for a New Year/Region

### Step 1: Prepare Configuration

1. Copy an existing region/year config:
   ```bash
   cp configs/hessen_2024.yaml configs/hessen_2025.yaml
   ```

2. Update the configuration:
   ```yaml
   temporal:
     year: 2025
     campaign_start: "2025-05-01"
     campaign_end: "2025-09-30"
     # Update holiday dates for the year

   data_paths:
     counter_locations: "counter-locations/DZS_ecovisio_2025.csv"
     counter_measurements: "counts/DZS_counts_ecovisio_2025.csv"
     traffic_volumes: "trafficvolumes/SR25_Hessen_VM_assessed.fgb"

   mlflow:
     experiment_name: "hessen-dtv-2025"
   ```

### Step 2: Prepare Data

1. **Counter Data**:
   - Download Eco-Visio data for the year
   - Place in `data/counter-locations/` and `data/counts/`

2. **STADTRADELN Data**:
   - Obtain aggregated GPS volumes (requires processing)
   - Run osmcategorizer on OSM data if new edges needed
   - Place FlatGeoBuf file in `data/trafficvolumes/`

3. **Structural Data** (update if needed):
   - VG250 from BKG (if new year release)
   - RegioStaR from BBSR (if classification changed)

### Step 3: Validate Data

```bash
uv run hochrechnung validate --config configs/hessen_2025.yaml
```

This checks:
- All required files exist
- Schemas are valid
- No unexpected missing data

### Step 3.5: Run ETL Pipeline

```bash
uv run hochrechnung etl --config configs/hessen_2025.yaml
```

This will:
- Load counter data and calculate DTV
- Load traffic volumes
- Match counters to OSM edges
- Add structural data (RegioStaR, demographics, etc.)
- Compute derived features
- Save training data to `cache/training_data_2025.csv`

### Step 3.6: Assess ETL Output

**Purpose**: Verify that ETL output matches reference data.

**Smart Assessment Strategy**:
1. ✅ **If reference data exists** (`data/validation/hessen_dauerzählstellen_{year}_osmid.csv`):
   - **Fast comparison**: Loads 2 CSVs and compares directly
   - **No source data reload** - very fast!

2. 🔄 **If no reference data**:
   - **Full validation**: Reloads all source data and recalculates
   - Verifies transformations were correct

**Usage**:
```bash
# Auto-detects cache/training_data_2023.csv
uv run hochrechnung assess --config configs/hessen_2023.yaml
```

Optional: Specify custom ETL output path
```bash
uv run hochrechnung assess --config configs/hessen_2023.yaml --etl-output cache/custom_output.csv
```

**What gets checked (fast mode with reference data)**:
- Direct comparison of all common columns between ETL output and reference
- Matches by counter `id`
- Numeric values compared with tolerance (rtol=1e-5)
- String values compared (trimmed)

**What gets checked (full validation mode - no reference)**:
- **DTV values** (`DZS_mean_SR`) recalculated from counter measurements
- **Counter locations** (`lat`, `lon`) compared with source CSV
- **Infrastructure categories** (`OSM_Radinfra`) validated against source
- **RegioStaR values** (`RegioStaR5`) range checked
- **Traffic volumes** (`Erh_SR`) compared with source FlatGeoBuf
- **Derived features** (`TN_SR_relativ`, `Streckengewicht_SR`) recalculated
- **Hub distances** (`HubDist`) range validation
- **Value ranges** for all numeric columns

**Assessment Report**:
The command will display:
- Overall status (PASS/WARN/FAIL)
- Individual check results with match percentages
- Sample failures for debugging (if any)

**Reference Data**:
For Hessen 2023, assessment compares against:
- `data/validation/hessen_dauerzählstellen_2023_osmid.csv` (legacy reference)
- Source files (counter measurements, traffic volumes, etc.)

**Exit codes**:
- `0` - Assessment passed (all checks PASS or WARN with >95% match)
- `1` - Assessment failed (one or more checks below 80% match)

**Note**: Assessment is currently designed for Hessen 2023/2024 data. A warning will be shown for other years/regions.

### Step 4: Run Training

```bash
# Start MLflow server (in separate terminal)
uv run mlflow server --host 127.0.0.1 --port 5000

# Run training
uv run hochrechnung train --config configs/hessen_2025.yaml
```

### Step 5: Generate Predictions

```bash
uv run hochrechnung predict \
  --config configs/hessen_2025.yaml \
  --model models:/hessen-dtv-random-forest/candidate \
  --output predictions/hessen_2025
```

---

## Common Operations

### Data Validation & Assessment

```bash
# Validate data schemas before ETL
uv run hochrechnung validate --config configs/hessen_2023.yaml

# Run ETL pipeline (creates cache/training_data_2023.csv)
uv run hochrechnung etl --config configs/hessen_2023.yaml

# Assess ETL output quality (auto-loads cache/training_data_2023.csv)
# No need to re-run ETL! Just loads the cached file.
uv run hochrechnung assess --config configs/hessen_2023.yaml

# Optional: Assess with custom ETL output path
uv run hochrechnung assess --config configs/hessen_2023.yaml \
  --etl-output cache/custom_output.csv
```

### Running Tests

```bash
# All tests
uv run pytest

# With coverage
uv run pytest --cov=hochrechnung

# Specific module
uv run pytest tests/test_schemas.py
```

### Code Quality Checks

```bash
# Format code
uv run ruff format .

# Lint
uv run ruff check .

# Auto-fix lint issues
uv run ruff check . --fix

# Type checking
uv run pyright
```

### Clearing Cache

```bash
# Clear all cached data
rm -rf cache/

# Clear specific cache
rm cache/traffic_data_*.pkl
```

### MLflow Operations

```bash
# Start server
uv run mlflow server --host 127.0.0.1 --port 5000

# View experiments (open browser)
# http://127.0.0.1:5000

# Export model
uv run mlflow models download -m "models:/hessen-dtv-random-forest/1" -o ./exported_model
```

---

## Troubleshooting

### Schema Validation Errors

**Problem**: `SchemaError: column 'X' not in dataframe`

**Solution**:
1. Check if source data format changed
2. Update column mapping in `normalization/columns.py`
3. Or update schema if new format is correct

### Memory Errors

**Problem**: `MemoryError` when loading traffic volumes

**Solution**:
1. Use chunked loading:
   ```python
   loader = TrafficVolumeLoader(config, chunk_size=50000)
   ```
2. Increase available RAM or swap space
3. Use region-specific PBF/FGB files (already done by default)

### Missing OSM Categorizer

**Problem**: `FileNotFoundError: osmcategorizer_rust not found`

**Solution**:

1. **Place the executable** in `./bin/osmcategorizer_rust.exe` (Windows) or `./bin/osmcategorizer_rust` (Linux/macOS)

2. **Or install to system PATH**:
   ```bash
   # Build from source
   git clone https://github.com/1prk/osmcategorizer_rust
   cd osmcategorizer_rust
   cargo build --release

   # Copy to project bin/
   cp target/release/osmcategorizer_rust ../hochrechnung/bin/
   ```

3. **Or use pre-assessed data**:
   - If `data/osm-data/hessen-230101-assessed.csv` exists, categorization is skipped
   - The ETL will load cached assessments instead

**Search order**:
1. System PATH
2. `./bin/osmcategorizer_rust.exe` (project directory)
3. `~/.cargo/bin/osmcategorizer_rust`
4. `/usr/local/bin/osmcategorizer_rust`

### MLflow Connection Errors

**Problem**: `mlflow.exceptions.MlflowException: Connection refused`

**Solution**:
1. Ensure MLflow server is running
2. Check tracking URI in config matches server address
3. Check firewall settings

---

## Performance Tuning

### Large Datasets

For datasets with >1M edges:

1. Use chunked loading:
   ```python
   loader = TrafficVolumeLoader(config, chunk_size=50000)
   ```

2. Enable Parquet caching:
   ```python
   cache_dataframe(gdf, path, format="parquet")
   ```

3. Use region-specific data files:
   - OSM PBF: `hessen-230101.osm.pbf` (not full planet)
   - Traffic volumes: `SR23_Hessen_VM.fgb` (region-filtered)
   - These are already filtered by region; no additional filtering needed

### Training Speed

1. Reduce hyperparameter grid size
2. Use `n_jobs=-1` for parallelization
3. Reduce CV folds during development

---

## Data Updates Checklist

When updating data for a new year:

- [ ] Counter locations CSV updated
- [ ] Counter measurements CSV updated
- [ ] STADTRADELN traffic volumes FlatGeoBuf updated
- [ ] Campaign statistics CSV updated with new year
- [ ] VG250 updated (if new release)
- [ ] Configuration YAML created for new year
- [ ] Validation passes without errors
- [ ] Test predictions look reasonable
