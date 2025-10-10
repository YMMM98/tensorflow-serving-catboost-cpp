# Description: CatBoost BUILD File.

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
)

include_files = [
    "catboost/catboost/libs/model_interface/c_api.h",
]

lib_files = [
    "catboost/catboost/libs/model_interface/libcatboostmodel.a",
]

genrule(
    name = "catboost-srcs",
    srcs = ["@catboost//:CMakeLists.txt"],
    outs = include_files + lib_files,
    cmd = "\n".join([
        "export INSTALL_DIR=$$(pwd)/$(@D)/catboost",
        "export TMP_DIR=$$(mktemp -d -t catboost.XXXXXX)",
        "mkdir -p $$TMP_DIR",
        "cp -R $$(pwd)/external/catboost/* $$TMP_DIR",
        "cd $$TMP_DIR/catboost/libs/model_interface",
        "mkdir -p build",
        "cd build",
        "cmake .. -DCATBOOST_COMPONENTS=model_interface",
        "make -j4",
        "mkdir -p $$INSTALL_DIR/catboost/libs/model_interface",
        "mkdir -p $$INSTALL_DIR/catboost/libs/model_interface/include",
        "cp libcatboostmodel.a $$INSTALL_DIR/catboost/libs/model_interface/",
        "cp ../c_api.h $$INSTALL_DIR/catboost/libs/model_interface/",
        "rm -rf $$TMP_DIR",
    ]),
)

cc_library(
    name = "catboost",
    srcs = ["catboost/catboost/libs/model_interface/libcatboostmodel.a"],
    hdrs = include_files,
    includes = ["catboost"],
    linkopts = ["-lpthread", "-ldl"],
    linkstatic = 1,
)
