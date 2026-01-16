# Troubleshooting Guide

## Common Issues and Solutions

---

### Windows-Specific Issues

#### `uv : The term 'uv' is not recognized as the name of a cmdlet`

**Cause**: PATH environment variable not refreshed after uv installation.

**Solution**:
1. Restart your PowerShell/terminal window, OR
2. Refresh PATH manually in current session:
   ```powershell
   $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
   ```
3. Or log out and log back in to Windows

See: https://github.com/astral-sh/uv/issues/3116

---

### Import Errors

#### `ModuleNotFoundError: No module named 'hochrechnung'`

**Cause**: Package not installed in editable mode.

**Solution**:
```bash
uv pip install -e .
# or
uv sync
```

#### `ImportError: cannot import name 'X' from 'hochrechnung'`

**Cause**: Stale bytecode or missing dependency.

**Solution**:
```bash
# Clear bytecode
find . -type d -name __pycache__ -exec rm -rf {} +

# Reinstall
uv sync
```

---

### Configuration Errors

#### `ValidationError: value is not a valid integer`

**Cause**: Type mismatch in YAML configuration.

**Solution**: Check that numeric values aren't quoted as strings:
```yaml
# Wrong
year: "2024"

# Correct
year: 2024
```

#### `KeyError: 'region'`

**Cause**: Required config section missing.

**Solution**: Ensure config inherits from base.yaml or includes all required sections.

---

### Data Loading Errors

#### `FileNotFoundError: Traffic volumes file not found`

**Cause**: Data file missing or wrong path in config.

**Solution**:
1. Check file exists at expected path
2. Verify `data_paths.data_root` is correct
3. Ensure relative paths are relative to data_root

#### `pyogrio.errors.DataSourceError: Unable to open`

**Cause**: Corrupted or incompatible geospatial file.

**Solution**:
1. Verify file format (FlatGeoBuf, GeoPackage, etc.)
2. Try opening with QGIS to check integrity
3. Regenerate file from source

#### `pandera.errors.SchemaError: series 'X' contains invalid values`

**Cause**: Data doesn't match schema constraints.

**Solution**:
1. Check which values failed validation
2. Either fix source data or update schema if new values are valid
3. Use `validate=False` temporarily to inspect data

---

### Spatial Errors

#### `CRSError: Invalid CRS`

**Cause**: Missing or incompatible coordinate reference system.

**Solution**:
```python
# Set CRS if missing
gdf = gdf.set_crs("EPSG:4326")

# Transform if mismatched
gdf = gdf.to_crs("EPSG:4326")
```

#### `GEOSException: TopologyException`

**Cause**: Invalid geometry (self-intersecting, etc.)

**Solution**:
```python
# Attempt to fix
gdf["geometry"] = gdf.geometry.buffer(0)
# or
gdf = gdf[gdf.geometry.is_valid]
```

#### No spatial matches found

**Cause**: CRS mismatch or non-overlapping bounding boxes.

**Solution**:
1. Check both datasets have same CRS
2. Verify bounding boxes overlap:
   ```python
   print(f"Data 1 bounds: {gdf1.total_bounds}")
   print(f"Data 2 bounds: {gdf2.total_bounds}")
   ```

---

### Training Errors

#### `ValueError: Input contains NaN`

**Cause**: Missing values in features.

**Solution**:
1. Check which columns have NaN:
   ```python
   print(X.isna().sum())
   ```
2. Either impute missing values or drop rows
3. Add imputation to preprocessing pipeline

#### `ValueError: Found input variables with inconsistent numbers of samples`

**Cause**: X and y have different lengths.

**Solution**:
1. Ensure filtering was applied to both X and y
2. Check for index alignment issues

#### `ConvergenceWarning: Solver terminated early`

**Cause**: Model didn't converge within max iterations.

**Solution**:
1. Increase `max_iter` parameter
2. Scale features if not already done
3. Try different solver

---

### MLflow Errors

#### `MlflowException: API request failed with status 404`

**Cause**: Model or run not found.

**Solution**:
1. Check model name spelling
2. Verify run ID exists in MLflow UI
3. Ensure tracking URI is correct

#### `MlflowException: The following failures occurred while calling start_run`

**Cause**: Nested runs without parent.

**Solution**:
```python
# Ensure parent run exists
with mlflow.start_run():
    with mlflow.start_run(nested=True):
        # Nested run here
        pass
```

---

### Performance Issues

#### Processing takes too long

**Solution**:
1. Use spatial bbox filtering to reduce data size
2. Enable chunked processing
3. Use Parquet caching for repeated loads
4. Reduce hyperparameter grid size during development

#### Out of memory

**Solution**:
1. Process data in chunks
2. Use `dtype` optimization:
   ```python
   df["count"] = df["count"].astype("int32")
   ```
3. Clear unused variables with `del`
4. Use Dask for larger-than-memory datasets

---

### Test Failures

#### `FAILED tests/test_package.py::test_package_imports`

**Cause**: Circular imports or missing dependencies.

**Solution**:
1. Run import manually to see full error:
   ```python
   import hochrechnung
   ```
2. Check for typos in import statements
3. Verify all dependencies installed

#### `pytest: error: unrecognized arguments`

**Cause**: Wrong pytest options or outdated config.

**Solution**:
1. Update pytest.ini_options in pyproject.toml
2. Run with `uv run pytest` to use project pytest

---

## Getting Help

1. Check existing issues in the repository
2. Search error message in documentation
3. Create minimal reproducible example
4. File issue with:
   - Python version
   - Package versions (`uv pip list`)
   - Full error traceback
   - Minimal code to reproduce
