# Data Dictionary

## Overview

This document describes all data products in the hochrechnung pipeline, their schemas, and semantic meanings.

---

## Source Data

### Counter Locations (`CounterLocationSchema`)

Permanent bicycle counting station (DZS) locations.

| Column | Type | Description |
|--------|------|-------------|
| `id` | string | Unique counter ID (e.g., "001") |
| `name` | string | Station name (nullable) |
| `latitude` | float | WGS84 latitude (-90 to 90) |
| `longitude` | float | WGS84 longitude (-180 to 180) |
| `ars` | string | 12-digit municipality code (nullable) |

**Source**: Eco-Visio counter management system
**Update Frequency**: When new counters are installed

---

### Counter Measurements (`CounterMeasurementSchema`)

Daily bicycle counts from permanent stations.

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | datetime | Measurement date |
| `counter_id` | string | Counter ID (FK to locations) |
| `count` | int | Daily bicycle count (≥0) |

**Source**: Eco-Visio API
**Update Frequency**: Daily

---

### Traffic Volumes (`TrafficVolumeSchema`)

Aggregated bicycle traffic per OSM edge from GPS traces.

| Column | Type | Description |
|--------|------|-------------|
| `edge_id` | int | Unique edge identifier |
| `base_id` | int | OSM way ID |
| `count` | int | Aggregated traffic count (≥0) |
| `bicycle_infrastructure` | string | Infrastructure category |
| `geometry` | LineString | Edge geometry |

**Valid Infrastructure Categories**:
- `no`: No dedicated infrastructure
- `mixed_way`: Shared pedestrian/bicycle path
- `mit_road`: Bicycle allowed on road
- `bicycle_lane`: Dedicated lane on road
- `bicycle_road`: Bicycle street (Fahrradstraße)
- `bicycle_way`: Separated bicycle path

**Source**: STADTRADELN GPS traces + osmcategorizer
**Update Frequency**: Annually (after STADTRADELN campaign)

---

### Municipalities (`MunicipalitySchema`)

Administrative boundaries from VG250.

| Column | Type | Description |
|--------|------|-------------|
| `ars` | string | 12-digit Amtlicher Regionalschlüssel |
| `name` | string | Municipality name |
| `population` | int | Population count (Einwohnerzahl) |
| `land` | string | Federal state code (2-digit) |
| `geometry` | Polygon | Municipality boundary |

**Source**: BKG VG250
**Update Frequency**: Annually (January 1st release)

---

### RegioStaR Classification (`RegioStarSchema`)

BBSR regional classification.

| Column | Type | Description |
|--------|------|-------------|
| `ars` | string | 12-digit municipality code |
| `regiostar5` | int | RegioStaR5 class (1-5) |
| `regiostar7` | int | RegioStaR7 class (1-7, nullable) |

**RegioStaR5 Categories**:
1. Metropolitan core
2. Metropolitan ring
3. Regiopolis / Large city
4. Medium-sized city area
5. Small town / Rural area

**Source**: BBSR
**Update Frequency**: Every few years

---

### STADTRADELN Demographics (`DemographicsSchema`)

Campaign participation statistics per municipality.

| Column | Type | Description |
|--------|------|-------------|
| `ars` | string | 12-digit municipality code |
| `n_users` | int | Number of participants |
| `n_trips` | int | Number of recorded trips |
| `total_km` | float | Total kilometers cycled |
| `bundesland` | string | Federal state name (nullable) |

**Source**: STADTRADELN statistics
**Update Frequency**: Annually (post-campaign)

---

## Derived Features

### Participation Rate

```
participation_rate = n_users / population
```

Relative STADTRADELN participation rate. Higher values indicate more engaged cycling communities.

---

### Route Intensity

```
route_intensity = (n_users * stadtradeln_volume) / population
```

Population-weighted cycling intensity on a route segment.

---

### Volume Per Trip

```
volume_per_trip = (stadtradeln_volume / n_trips) * n_users
```

User-adjusted traffic volume per trip.

---

## Target Variable

### Daily Traffic Volume (DTV)

Average daily bicycle count during the measurement period.

| Field | Description |
|-------|-------------|
| `value` | Mean daily count |
| `observation_count` | Days with measurements |
| `missing_days` | Days without measurements |
| `quality_score` | Completeness metric (0-1) |

**Quality Score Calculation**:
```
quality_score = completeness * (1 - zero_ratio * 0.5)
```

Where:
- `completeness` = observation_count / expected_days
- `zero_ratio` = zero_days / observation_count

**Validity Criteria**:
- quality_score ≥ 0.5
- observation_count ≥ 7

---

## Output Data

### Predictions (`PredictionOutputSchema`)

Model prediction results.

| Column | Type | Description |
|--------|------|-------------|
| `edge_id` | int | Edge identifier (nullable) |
| `predicted_dtv` | float | Predicted daily traffic (≥0) |
| `prediction_lower` | float | Lower CI bound (nullable) |
| `prediction_upper` | float | Upper CI bound (nullable) |

---

## Column Name Mapping

| Internal Name | Common External Names |
|--------------|----------------------|
| `population` | `EWZ`, `Einwohnerzahl_EWZ`, `Bev_insg` |
| `stadtradeln_volume` | `count`, `n`, `Erh_SR` |
| `infra_category` | `bicycle_infrastructure`, `OSM_Radinfra` |
| `dtv` | `dtv_value`, `DZS_mean_SR`, `DTV` |
| `ars` | `ARS`, `Regionalschlüssel_ARS`, `ags` |
