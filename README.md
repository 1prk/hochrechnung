# Hochrechnung - Bicycle Traffic Estimation Pipeline

ML pipeline for estimating average daily bicycle traffic (DTV) per OSM edge using regression models.

## Quick Start

### Installation

```bash
# Install uv package manager
pip install uv

# Install dependencies
uv sync

# Install pre-commit hooks
uv run pre-commit install
```

### Running the Pipeline

The standard workflow uses `configs/germany_2025.yaml`. Each step below builds on the previous one.

#### 1. Validate Data Schemas

```bash
uv run hochrechnung validate --config configs/germany_2025.yaml
```

Checks that all required files exist and schemas are valid.

#### 2. Run ETL Pipeline

```bash
uv run hochrechnung etl --config configs/germany_2025.yaml
```

Outputs: `cache/training_data_2025.csv`

Two modes are available:
- `--mode production` (default) - uses verified counters
- `--mode verification` - creates verification data and flags outliers

#### 3. Verify Counters (optional)

```bash
uv run hochrechnung verify --config configs/germany_2025.yaml
```

Runs the verification workflow:
1. Runs ETL in verification mode
2. Generates MBTiles for map visualization
3. Exports verification data
4. Launches interactive web UI for manual counter verification

Verified counters are saved to `data/verified/counters_verified_{year}.csv`.

#### 4. Assess ETL Output Quality

```bash
uv run hochrechnung assess --config configs/germany_2025.yaml
```

Automatically loads `cache/training_data_2025.csv` (no need to re-run ETL).

**Exit codes**: `0` = passed, `1` = failed

#### 5. Train Models

```bash
# Start MLflow server (in separate terminal)
uv run mlflow server --host 127.0.0.1 --port 5000

# Train using ETL output
uv run hochrechnung train --config configs/germany_2025.yaml

# Or train using curated Germany-wide counter data
uv run hochrechnung train --config configs/germany_2025.yaml \
  --curated --year 2025
```

Trains both baseline (single predictor) and enhanced (all features) model variants.
Models are saved to `cache/models/`.

#### 6. Generate Predictions

```bash
uv run hochrechnung predict \
  --config configs/germany_2025.yaml \
  --model cache/models/random_forest_enhanced_2025.joblib \
  --output predictions/germany_2025
```

Output formats: `fgb` (default), `gpkg`, `parquet`, `csv`.

#### 7. Calibrate Predictions

```bash
uv run hochrechnung calibrate \
  --config configs/germany_2025.yaml \
  --model cache/models/random_forest_curated_2025.joblib \
  --calibrator log_linear
```

Calibrator types: `global_multiplicative`, `log_linear`, `stratified`.

Options:
- `--verify` - launch verification UI for calibration stations before calibrating
- `--no-loocv` - skip Leave-One-Out Cross-Validation

Calibrated predictions are saved to `predictions/`.

## Project Structure

```
hochrechnung/
├── src/hochrechnung/
│   ├── schemas/         # Pandera data contracts
│   ├── ingestion/       # Data loading
│   ├── normalization/   # Data standardization
│   ├── features/        # Feature engineering
│   ├── targets/         # DTV calculation
│   ├── etl/             # ETL pipeline orchestration
│   ├── assessment/      # ETL output validation
│   ├── validation/      # Schema validation runner
│   ├── verification/    # Counter verification UI + tiles
│   ├── modeling/        # Training, inference, curated data
│   ├── evaluation/      # Experiment tracking, metrics, reports
│   ├── calibration/     # Post-prediction calibration
│   ├── config/          # Configuration system
│   └── utils/           # Logging, caching
├── configs/             # Region/year YAML configs
├── tests/               # Test suite
├── data/                # Data files (not in git)
│   ├── counts/          # Counter measurements
│   ├── counter-locations/  # Counter locations + images DB
│   ├── trafficvolumes/  # STADTRADELN GPS volumes
│   ├── osm-data/        # OSM infrastructure
│   ├── structural-data/ # Gebietseinheiten, city centroids
│   ├── kommunen-stats/  # Commune statistics JSON
│   ├── campaign/        # Campaign data
│   ├── validation/      # Reference data
│   └── verified/        # Verified counter datasets
├── predictions/         # Prediction outputs (.fgb, diagnostics)
└── cache/               # Cached ETL outputs + trained models
```

## Configuration

Each region/year has its own config file (e.g., `configs/germany_2025.yaml`):

```yaml
project: germany-2025

ars: "000000000000"
region_name: "Deutschland"

year: 2025

period:
  start: "2025-05-01"
  end: "2025-09-30"

data:
  traffic_volumes: "trafficvolumes/SR25_DE_VM.fgb"
  osm_pbf: "osm-data/germany-250101.osm.pbf"
  city_centroids: "structural-data/places.gpkg"
  counter_locations: "counter-locations/germany_dzs_2025_gesamt.csv"
  gebietseinheiten: "structural-data/DE_Gebietseinheiten.gpkg"
  images_db: "counter-locations/germany_dzs.db"
  kommunen_stats: "kommunen-stats/SR25_Commune_Statistics.json"

training:
  deduplicate_edges: true
  min_volume_ratio: 0.27
  max_volume_ratio: 9.81

stats:
  approach: gebietseinheiten
  admin_level: Verwaltungsgemeinschaft
```

## Development

### Code Quality

```bash
# Format
uv run ruff format .

# Lint
uv run ruff check .

# Type check
uv run pyright
```

### Testing

```bash
# All tests
uv run pytest

# With coverage
uv run pytest --cov=hochrechnung

# Specific test
uv run pytest tests/test_assessment.py
```

## Documentation

- **[Runbook](docs/runbook.md)** - Complete operational guide
- **[Development Guidelines](.claude/CLAUDE.md)** - Code standards and practices

## Requirements

- Python 3.11+
- uv package manager
- 16GB RAM (for full Germany processing)
- 50GB disk space (for data + cache)

## Data Pipeline Overview

```
Counter Data + Traffic Volumes + Structural Data
                    |
             [ETL Pipeline]
          - Load & validate schemas
          - Calculate DTV from measurements
          - Match counters to OSM edges
          - Join structural data
          - Compute derived features
                    |
          training_data.csv
                    |
          [Verify] (optional)
     Flag outliers, manual review
                    |
            [Assessment]
      Compare with source data
                    |
           [ML Training]
     Train regression models for DTV
      (baseline + enhanced variants)
                    |
           [Prediction]
     Generate DTV for all OSM edges
                    |
          [Calibration]
  Adjust predictions using independent
         counting stations
```
