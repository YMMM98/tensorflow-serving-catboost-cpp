# CatBoost Servable for TensorFlow Serving

This directory contains the CatBoost servable implementation for TensorFlow Serving, mirroring the XGBoost implementation architecture.

## Overview

The CatBoost servable enables TensorFlow Serving to load and serve CatBoost models using the CatBoost C API.

## Implementation Files

### Core Implementation (11 files)

#### 1. Constants & Configuration
- **[catboost_constants.h](catboost_constants.h)** - Defines model filename constant (`catboost.cbm`) and feature names
- **[catboost_source_adapter.proto](catboost_source_adapter.proto)** - Protocol buffer config for source adapter

#### 2. Model Bundle (Wrapper)
- **[catboost_bundle.h](catboost_bundle.h)** - Header defining CatBoostBundle class
- **[catboost_bundle.cc](catboost_bundle.cc)** - Implementation using CatBoost C API:
  - `ModelCalcerCreate()` - Create model handle
  - `LoadFullModelFromFile()` - Load model from disk
  - `ModelCalcerDelete()` - Free resources

#### 3. Source Adapter (Integration)
- **[catboost_source_adapter.h](catboost_source_adapter.h)** - Adapter interface
- **[catboost_source_adapter.cc](catboost_source_adapter.cc)** - Registers CatBoost with TF Serving's loading system using `REGISTER_STORAGE_PATH_SOURCE_ADAPTER`

#### 4. Prediction Logic
- **[predict_impl.h](predict_impl.h)** - CatBoostPredictor interface
- **[predict_impl.cc](predict_impl.cc)** - Implementation:
  - Converts sparse features (CSR format) to dense arrays
  - Calls `CalcModelPredictionFlat()` with correct `const float**` signature
  - Returns predictions as TensorProto
  - Includes latency recording with bvar

#### 5. Utilities
- **[util.h](util.h)** - Helper function declarations
- **[util.cc](util.cc)** - Helper functions for ModelSpec creation and metrics tracking

#### 6. Build Configuration
- **[BUILD](BUILD)** - Bazel build targets for:
  - `catboost_bundle` - Model wrapper library
  - `catboost_source_adapter` - Adapter library
  - `predict_impl` - Prediction implementation
  - `util` - Utility functions
  - Tests for all components

### Test Files (3 files)

- **[catboost_bundle_test.cc](catboost_bundle_test.cc)** - Tests model loading functionality
- **[catboost_source_adapter_test.cc](catboost_source_adapter_test.cc)** - Tests adapter integration with TF Serving
- **[predict_impl_test.cc](predict_impl_test.cc)** - Tests prediction logic

### Test Data Structure

- **[testdata/BUILD](testdata/BUILD)** - Build file for test data
- **[testdata/export_test_model.py](testdata/export_test_model.py)** - Python script to generate test CatBoost model
- **testdata/test_model/1/** - Directory for versioned test model (needs `catboost.cbm` file)

## CatBoost C API Functions Used

| Function | Purpose |
|----------|---------|
| `ModelCalcerCreate()` | Creates an empty model handle |
| `LoadFullModelFromFile(handle, path)` | Loads model from file |
| `CalcModelPredictionFlat(handle, docCount, features, featuresSize, result, resultSize)` | Makes predictions |
| `GetErrorString()` | Retrieves error messages |
| `ModelCalcerDelete(handle)` | Frees model handle |

## Input/Output Format

### Input
Uses the same `FeatureScore` sparse format as XGBoost (defined in `tensorflow_serving/apis/predict.proto`):

```protobuf
message FeatureScore {
  repeated uint64 id = 1;      // Feature indices
  repeated float score = 2;     // Feature values
}

message PredictRequest {
  ModelSpec model_spec = 1;
  map<string, FeatureScoreVector> inputs = 2;  // Key: "catboost_features"
}
```

### Output
Returns predictions as TensorProto in the response:

```protobuf
message PredictResponse {
  ModelSpec model_spec = 2;
  map<string, TensorProto> outputs = 1;  // Key: "predictions"
}
```

## Sparse to Dense Conversion

The implementation converts sparse CSR format to dense feature arrays because CatBoost C API expects `const float**`:

1. Find maximum feature ID to determine vector size
2. Create dense vectors initialized with zeros
3. Fill in sparse values at their indices
4. Create array of pointers for CatBoost API

## Dependencies

### Required
- CatBoost library with C API (`catboost/libs/model_interface/c_api.h`)
- TensorFlow Serving core libraries
- Protocol Buffers
- Bazel build system
- brpc (for bvar latency recording)

### Bazel Workspace Configuration
Requires `@catboost//:catboost` external dependency to be configured in WORKSPACE file.

## Setup and Testing

### 1. Add CatBoost Dependency
Add to WORKSPACE file:
```python
# CatBoost dependency configuration
# TODO: Add catboost external dependency
```

### 2. Generate Test Model
```bash
cd tensorflow_serving/servables/catboost/testdata
python3 export_test_model.py
```

This creates `testdata/test_model/1/catboost.cbm` for testing.

### 3. Build and Test
```bash
# Build all CatBoost targets
bazel build //tensorflow_serving/servables/catboost/...

# Run tests
bazel test //tensorflow_serving/servables/catboost:catboost_bundle_test
bazel test //tensorflow_serving/servables/catboost:catboost_source_adapter_test
bazel test //tensorflow_serving/servables/catboost:predict_impl_test
```

## Integration Steps (TODO)

### Phase 1: Core Implementation ✅
- [x] Create constants and configuration
- [x] Implement model bundle wrapper
- [x] Implement source adapter
- [x] Implement prediction logic
- [x] Create utility functions
- [x] Write tests

### Phase 2: Dependencies and Testing (In Progress)
- [ ] Add CatBoost to WORKSPACE
- [ ] Generate test model
- [ ] Run and verify tests

### Phase 3: Server Integration (Future)
- [ ] Register CatBoost platform in model_servers
- [ ] Add CatBoost predictor to prediction service
- [ ] Update platform configuration
- [ ] Integration tests with model server

### Phase 4: API Integration (Future)
- [ ] Review if `predict.proto` needs CatBoost-specific extensions
- [ ] Update `prediction_service.proto` if needed
- [ ] Add CatBoost examples and documentation

## Model File Format

CatBoost models should be saved in `.cbm` (CatBoost Binary Model) format:

```python
import catboost

model = CatBoostClassifier(...)
model.fit(X_train, y_train)
model.save_model('model.cbm', format='cbm')
```

## Directory Structure

```
tensorflow_serving/servables/catboost/
├── BUILD                              # Bazel build configuration
├── README.md                          # This file
├── catboost_bundle.h/cc              # Model wrapper
├── catboost_bundle_test.cc           # Bundle tests
├── catboost_constants.h              # Constants
├── catboost_source_adapter.h/cc      # Source adapter
├── catboost_source_adapter.proto     # Adapter config proto
├── catboost_source_adapter_test.cc   # Adapter tests
├── predict_impl.h/cc                 # Prediction implementation
├── predict_impl_test.cc              # Prediction tests
├── util.h/cc                         # Utility functions
└── testdata/
    ├── BUILD                          # Test data build file
    ├── export_test_model.py          # Model generation script
    └── test_model/
        └── 1/
            └── catboost.cbm          # Test model (generated)
```

## Key Differences from XGBoost

| Aspect | XGBoost | CatBoost |
|--------|---------|----------|
| Model file | `deploy.model` | `catboost.cbm` |
| C API handle | `BoosterHandle` | `ModelCalcerHandle*` |
| Load function | `XGBoosterLoadModel` | `LoadFullModelFromFile` |
| Predict function | `XGBoosterPredict` | `CalcModelPredictionFlat` |
| Input format | CSR sparse matrix | Dense float arrays (`const float**`) |
| Feature name | `xgboost_features` | `catboost_features` |

## References

- [CatBoost C API Documentation](https://catboost.ai/docs/en/concepts/c-plus-plus-api_dynamic-c-pluplus-wrapper)
- [CatBoost C API Header](https://github.com/catboost/catboost/blob/master/catboost/libs/model_interface/c_api.h)
- [TensorFlow Serving Architecture](https://www.tensorflow.org/tfx/serving/architecture)
- XGBoost servable implementation (reference architecture)
