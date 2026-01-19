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

#### 1. Validate Data Schemas

```bash
uv run hochrechnung validate --config configs/hessen_2023.yaml
```

#### 2. Run ETL Pipeline

```bash
uv run hochrechnung etl --config configs/hessen_2023.yaml
```

Outputs: `cache/training_data_2023.csv`

#### 3. Assess ETL Output Quality

```bash
# Automatically loads cache/training_data_2023.csv (no need to re-run ETL!)
uv run hochrechnung assess --config configs/hessen_2023.yaml
```

**Smart Assessment**:
- ✅ If `data/validation/hessen_dauerzählstellen_2023_osmid.csv` exists: **Fast comparison** (just 2 CSVs)
- 🔄 If no reference: Full validation (reloads all source data)

Compares ETL output with reference data:
- All common columns matched by counter `id`
- Numeric values (tolerance: rtol=1e-5)
- String values (trimmed comparison)

**Exit codes**:
- `0` - Assessment passed (≥95% match)
- `1` - Assessment failed (<80% match)

#### 4. Train Models (TODO)

```bash
# Start MLflow server
uv run mlflow server --host 127.0.0.1 --port 5000

# Run training
uv run hochrechnung train --config configs/hessen_2023.yaml
```

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
│   ├── config/          # Configuration system
│   └── utils/           # Logging, caching
├── configs/             # Region/year YAML configs
├── tests/               # Test suite
├── data/                # Data files (not in git)
│   ├── counts/          # Counter measurements
│   ├── counter-locations/  # Counter locations
│   ├── trafficvolumes/  # STADTRADELN GPS volumes
│   ├── osm-data/        # OSM infrastructure
│   ├── structural-data/ # VG250, RegioStaR
│   └── validation/      # Reference data
└── cache/               # Cached ETL outputs
```

## Key Files

- **configs/hessen_2023.yaml** - Configuration for Hessen 2023
- **data/validation/hessen_dauerzählstellen_2023_osmid.csv** - Legacy reference for assessment
- **cache/training_data_2023.csv** - ETL output (training-ready dataset)

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
- 16GB RAM (for full Hessen processing)
- 50GB disk space (for data + cache)

## Data Pipeline Overview

```
Counter Data + Traffic Volumes + Structural Data
                    ↓
            [ETL Pipeline]
         - Load & validate schemas
         - Calculate DTV from measurements
         - Match counters to OSM edges
         - Join structural data
         - Compute derived features
                    ↓
         training_data.csv
                    ↓
           [Assessment]
     Compare with source data
                    ↓
          [ML Training]
    Train regression models for DTV

```

## Configuration

Each region/year has its own config file (e.g., `configs/hessen_2023.yaml`):

```yaml
region:
  code: "06"
  name: "Hessen"
  bbox: [7.77, 49.39, 10.24, 51.66]

temporal:
  year: 2023
  campaign_start: "2023-05-01"
  campaign_end: "2023-09-30"

data_paths:
  counter_measurements: "counts/DZS_counts_ecovisio_2023.csv"
  traffic_volumes: "trafficvolumes/SR23_Hessen_VM.fgb"
  osm_pbf: "osm-data/hessen-230101.osm.pbf"
  # ... more paths
```

## License

Internal project - Technische Universität Darmstadt
