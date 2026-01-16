# Architecture Overview

## System Design

The hochrechnung pipeline estimates average daily bicycle traffic (DTV) per OSM edge using regression models trained on counter data and STADTRADELN GPS traces.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            DATA SOURCES                                  │
├─────────────┬─────────────┬─────────────┬─────────────┬─────────────────┤
│   Counter   │ STADTRADELN │    OSM      │  VG250      │   RegioStaR     │
│  Locations  │  GPS Traces │Infrastructure│Municipalities│ Classification │
└──────┬──────┴──────┬──────┴──────┬──────┴──────┬──────┴────────┬────────┘
       │             │             │             │               │
       ▼             ▼             ▼             ▼               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         INGESTION LAYER                                  │
│  • Schema validation at boundaries                                       │
│  • Type coercion                                                         │
│  • Missing data handling                                                 │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       NORMALIZATION LAYER                                │
│  • Column name standardization                                           │
│  • Temporal alignment (campaign periods)                                 │
│  • Spatial matching (counter-to-edge)                                    │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      FEATURE ENGINEERING                                 │
│  • Derived features (participation_rate, route_intensity, etc.)         │
│  • Infrastructure category mapping                                       │
│  • Feature validation                                                    │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        TARGET PROCESSING                                 │
│  • DTV calculation from counter data                                     │
│  • Quality scoring                                                       │
│  • Filtering by quality thresholds                                       │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          MODELING LAYER                                  │
│  • Preprocessing pipeline (ColumnTransformer)                            │
│  • Model training with CV                                                │
│  • Hyperparameter tuning                                                 │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         EVALUATION & INFERENCE                           │
│  • Standardized metrics (R², RMSE, MAE, MAPE)                           │
│  • MLflow experiment tracking                                            │
│  • Prediction generation                                                 │
└─────────────────────────────────────────────────────────────────────────┘
```

## Module Structure

```
src/hochrechnung/
├── __init__.py           # Package root
├── cli.py                # Command-line interface
├── schemas/              # Pandera data contracts
│   ├── counter.py        # Counter location/measurement schemas
│   ├── campaign.py       # STADTRADELN campaign schemas
│   ├── traffic.py        # Traffic volume schemas
│   ├── infrastructure.py # OSM infrastructure schemas
│   ├── structural.py     # Municipality/RegioStaR schemas
│   ├── output.py         # Prediction output schemas
│   └── registry.py       # Schema versioning & discovery
├── config/               # Configuration management
│   ├── settings.py       # Pydantic config models
│   └── loader.py         # YAML loading with inheritance
├── ingestion/            # Data loading
│   ├── base.py           # Base loader classes
│   ├── counter.py        # Counter data loaders
│   ├── campaign.py       # Campaign data loaders
│   ├── traffic.py        # Traffic volume loaders
│   ├── structural.py     # Structural data loaders
│   └── osm.py            # OSM categorizer integration
├── normalization/        # Data standardization
│   ├── columns.py        # Column name mapping
│   ├── temporal.py       # Temporal alignment
│   └── spatial.py        # Spatial matching
├── features/             # Feature engineering
│   ├── definitions.py    # Declarative feature registry
│   ├── infrastructure.py # Infrastructure mapping
│   └── pipeline.py       # Feature computation pipeline
├── targets/              # Target variable processing
│   ├── dtv.py            # DTV calculation
│   └── quality.py        # Quality assessment
├── modeling/             # Model training & inference
│   ├── preprocessing.py  # sklearn transformers
│   ├── models.py         # Model registry
│   ├── training.py       # Training pipeline
│   └── inference.py      # Prediction generation
├── evaluation/           # Metrics & experiments
│   ├── metrics.py        # Regression metrics
│   └── experiment.py     # MLflow experiment classes
└── utils/                # Utilities
    ├── logging.py        # Structured logging
    ├── cache.py          # Content-addressable caching
    └── hashing.py        # Deterministic hashing
```

## Key Design Decisions

### 1. Schema-First Approach

All data structures are defined as Pandera schemas before any processing code. This ensures:
- Explicit data contracts
- Validation at system boundaries
- Clear documentation of field semantics

### 2. Configuration as Code

Configuration uses Pydantic models with:
- Type validation
- Environment variable interpolation
- YAML inheritance (base.yaml + region_year.yaml)
- No hardcoded year/region logic in processing code

### 3. Declarative Features

Features are defined declaratively with explicit dependencies:
```python
DerivedFeature(
    name="participation_rate",
    formula=lambda df: df["n_users"] / df["population"],
    dependencies=("n_users", "population"),
)
```

### 4. Content-Addressable Caching

Large intermediate results are cached based on:
- Source file modification times
- Configuration hash
- Schema version

### 5. MLflow Integration

Each experiment answers ONE question:
- `model_comparison/`: Which model works best?
- `feature_selection/`: Which features matter?
- `hyperparameter/`: Optimal hyperparameters?
- `temporal_stability/`: Does model generalize across years?

## Data Flow

1. **Raw Data** → Ingestion (schema validation)
2. **Validated Data** → Normalization (column names, temporal alignment)
3. **Normalized Data** → Features (derived calculations)
4. **Feature Data** → Training (model fitting)
5. **Trained Model** → Inference (predictions)
6. **Predictions** → Output (schema validation)
