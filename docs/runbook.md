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
2. Reduce bbox to smaller area
3. Increase swap space

### Missing OSM Categorizer

**Problem**: `FileNotFoundError: osmcategorizer_rust not found`

**Solution**:
1. Install osmcategorizer_rust from source
2. Or use pre-categorized data (skip OSM processing)

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
   ```yaml
   # Not in config, but in code
   chunk_size: 50000
   ```

2. Enable Parquet caching:
   ```python
   cache_dataframe(gdf, path, format="parquet")
   ```

3. Use spatial filtering:
   ```python
   loader = TrafficVolumeLoader(config, bbox=(8.0, 50.0, 9.0, 51.0))
   ```

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
