/*

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow_serving/servables/catboost/predict_impl.h"

#include "tensorflow_serving/core/servable_handle.h"
#include "tensorflow_serving/servables/catboost/catboost_bundle.h"
#include "tensorflow_serving/servables/catboost/catboost_constants.h"

#include "tensorflow_serving/servables/catboost/util.h"

#include <chrono>

namespace tensorflow {
namespace serving {

bvar::LatencyRecorder
    CatBoostPredictor::catboost_latency_recorder("catboost_predict");

CatBoostPredictor::CatBoostPredictor() {}

Status CatBoostPredictor::Predict(ServerCore *core,
                                  const PredictRequest &request,
                                  PredictResponse *response) {
  if (!request.has_model_spec()) {
    return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                              "Missing ModelSpec");
  }
  return PredictWithModelSpec(core, request.model_spec(), request, response);
}

Status CatBoostPredictor::PredictWithModelSpec(ServerCore *core,
                                               const ModelSpec &model_spec,
                                               const PredictRequest &request,
                                               PredictResponse *response) {
  ServableHandle<CatBoostBundle> bundle;
  TF_RETURN_IF_ERROR(core->GetServableHandle(model_spec, &bundle));

  if (!request.inputs().contains(kCatBoostFeaturesName)) {
    return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                              "No catboost_features input found");
  }

  int32_t batch_size =
      request.inputs().at(kCatBoostFeaturesName).feature_score_size();

  // Convert sparse features to dense format for CatBoost
  // CatBoost C API expects: const float** (array of pointers to feature arrays)

  // First, find the maximum feature ID to determine feature vector size
  size_t max_feature_id = 0;
  for (int32_t i = 0; i < batch_size; i++) {
    const auto& feature_score = request.inputs().at(kCatBoostFeaturesName).feature_score(i);
    if (feature_score.id_size() != feature_score.score_size()) {
      return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                                "Sizes(catboost_features) of id and score must be the same");
    }
    for (int j = 0; j < feature_score.id_size(); j++) {
      if (feature_score.id(j) > max_feature_id) {
        max_feature_id = feature_score.id(j);
      }
    }
  }

  // Create dense feature vectors (initialize with zeros)
  size_t num_features = max_feature_id + 1;
  std::vector<std::vector<float>> dense_features(batch_size, std::vector<float>(num_features, 0.0f));

  // Fill in the sparse values
  for (int32_t i = 0; i < batch_size; i++) {
    const auto& feature_score = request.inputs().at(kCatBoostFeaturesName).feature_score(i);
    for (int j = 0; j < feature_score.id_size(); j++) {
      dense_features[i][feature_score.id(j)] = feature_score.score(j);
    }
  }

  // Create array of pointers for CatBoost API (const float**)
  std::vector<const float*> float_features_ptrs(batch_size);
  for (int32_t i = 0; i < batch_size; i++) {
    float_features_ptrs[i] = dense_features[i].data();
  }

  // Prepare output buffer
  // Result size should be: batch_size * approxDimension
  // For most models, approxDimension is 1 (single prediction per doc)
  std::vector<double> predictions(batch_size);

  auto start = std::chrono::system_clock::now();

  // Call CatBoost prediction API
  bool success = CalcModelPredictionFlat(
      bundle->GetModelHandle(),
      batch_size,
      float_features_ptrs.data(),
      num_features,
      predictions.data(),
      predictions.size()
  );

  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  catboost_latency_recorder << elapsed_seconds.count() * 1000000;

  if (!success) {
    const char* error_msg = GetErrorString();
    std::string error_string = error_msg ? error_msg : "Unknown error";
    return tensorflow::Status(tensorflow::error::UNKNOWN,
                              "CatBoost prediction failed: " + error_string);
  }

  // Convert predictions to Tensor
  Tensor output_tensor{DT_DOUBLE, {static_cast<int64_t>(predictions.size())}};
  std::copy_n(predictions.data(), predictions.size(),
              output_tensor.flat<double>().data());

  MakeModelSpec(request.model_spec().name(), /*signature_name=*/{},
                bundle.id().version, response->mutable_model_spec());
  output_tensor.AsProtoField(&((*response->mutable_outputs())["predictions"]));

  return Status::OK();
}

CatBoostPredictor::~CatBoostPredictor() {}
}
}
