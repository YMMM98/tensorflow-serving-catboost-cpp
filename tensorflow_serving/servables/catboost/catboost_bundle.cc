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

#include "tensorflow_serving/servables/catboost/catboost_bundle.h"

#include "tensorflow_serving/servables/catboost/catboost_constants.h"

#include "tensorflow_serving/util/file_probing_env.h"

namespace tensorflow {
namespace serving {

CatBoostBundle::CatBoostBundle() : catboost_model_(nullptr) {
  LOG(INFO) << "Call the constructor CatBoostBundle().";
}

CatBoostBundle::CatBoostBundle(std::string model_path) : catboost_model_(nullptr) {
  LOG(INFO) << "Call the constructor CatBoostBundle(string)";
  Status status = LoadCatBoostModel(model_path);
  if (status.ok()) {
    LOG(INFO) << "Load the CatBoost model successfully.";
  } else {
    LOG(ERROR) << status.error_message();
  }
}

Status CatBoostBundle::LoadCatBoostModel(std::string model_path) {
  // Load The CatBoost Model
  std::string catboost_model_path = model_path + "/" + kCatBoostModelFileName;
  if (!Env::Default()->FileExists(catboost_model_path).ok()) {
    return errors::Unknown("CatBoost Model Path is empty: " +
                           catboost_model_path);
  }

  catboost_model_ = ModelCalcerCreate();
  if (catboost_model_ == nullptr) {
    return errors::Unknown("Failed to call ModelCalcerCreate");
  }

  if (!LoadFullModelFromFile(catboost_model_, catboost_model_path.c_str())) {
    const char* error_msg = GetErrorString();
    std::string error_string = error_msg ? error_msg : "Unknown error";
    ModelCalcerDelete(catboost_model_);
    catboost_model_ = nullptr;
    return errors::Unknown("Failed to load CatBoost model: " + error_string);
  }

  return Status::OK();
}

Status CatBoostBundle::UnloadCatBoostModel() {
  if (catboost_model_ != nullptr) {
    ModelCalcerDelete(catboost_model_);
    catboost_model_ = nullptr;
  }
  return Status::OK();
}

ModelCalcerHandle* CatBoostBundle::GetModelHandle() const {
  return catboost_model_;
}

CatBoostBundle::~CatBoostBundle() {
  // Unload The CatBoost Model
  LOG(INFO) << "Call the destructor ~CatBoostBundle().";
  Status status = UnloadCatBoostModel();
  if (status.ok()) {
    LOG(INFO) << "Unload the CatBoost model successfully.";
  } else {
    LOG(ERROR) << status.error_message();
  }
}

} // namespace serving
} // namespace tensorflow
