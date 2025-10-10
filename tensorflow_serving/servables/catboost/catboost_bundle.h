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

#ifndef TENSORFLOW_SERVING_SERVABLES_CATBOOST_CATBOOST_BUNDLE_H_
#define TENSORFLOW_SERVING_SERVABLES_CATBOOST_CATBOOST_BUNDLE_H_

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "catboost/libs/model_interface/c_api.h"
#include <memory>
#include <string>

namespace tensorflow {
namespace serving {
class CatBoostBundle {
public:
  CatBoostBundle();

  CatBoostBundle(std::string model_path);

  // Load the CatBoost model from the given path.
  Status LoadCatBoostModel(std::string model_path);

  // Unload the CatBoost model.
  Status UnloadCatBoostModel();

  // Get the ModelHandle.
  ModelCalcerHandle* GetModelHandle() const;

  ~CatBoostBundle();

private:
  ModelCalcerHandle* catboost_model_; // Handle to the CatBoost model.
};
}
}

#endif // TENSORFLOW_SERVING_SERVABLES_CATBOOST_CATBOOST_BUNDLE_H_
