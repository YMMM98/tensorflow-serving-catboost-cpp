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

#include "catboost_source_adapter.h"

#include <memory>

namespace tensorflow {
namespace serving {

CatBoostSourceAdapter::CatBoostSourceAdapter()
    : SimpleLoaderSourceAdapter<StoragePath, CatBoostBundle>(
          [](const StoragePath &path, std::unique_ptr<CatBoostBundle> * bundle) {
            bundle->reset(new CatBoostBundle());
            return (*bundle)->LoadCatBoostModel(path);
          },
          SimpleLoaderSourceAdapter<StoragePath,
                                    CatBoostBundle>::EstimateNoResources()) {}
CatBoostSourceAdapter::~CatBoostSourceAdapter() { Detach(); }

// Register the source adapter.
class CatBoostSourceAdapterCreator {
public:
  static Status
  Create(const CatBoostSourceAdapterConfig &config,
         std::unique_ptr<SourceAdapter<StoragePath, std::unique_ptr<Loader>>>
             *adapter) {
    adapter->reset(new CatBoostSourceAdapter());
    return Status::OK();
  }
};
REGISTER_STORAGE_PATH_SOURCE_ADAPTER(CatBoostSourceAdapterCreator,
                                     CatBoostSourceAdapterConfig);

} // namespace serving
} // namespace tensorflow
