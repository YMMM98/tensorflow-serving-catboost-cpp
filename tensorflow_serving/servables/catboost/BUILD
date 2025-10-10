# Description: TensorFlow Serving CatBoost servable.

package(
    default_visibility = [
        "//tensorflow_serving:internal",
    ],
    features = ["-layering_check"],
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

load("//tensorflow_serving:serving.bzl", "serving_proto_library")

serving_proto_library(
    name = "catboost_source_adapter_proto",
    srcs = ["catboost_source_adapter.proto"],
    cc_api_version = 2,
    visibility = [
        "//visibility:public",
    ],
    deps = [
    ],
)

cc_library(
    name = "util",
    srcs = ["util.cc"],
    hdrs = ["util.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//tensorflow_serving/apis:model_proto",
        "//tensorflow_serving/util:optional",
        "@com_google_protobuf//:cc_wkt_protos",
        "@org_tensorflow//tensorflow/core:framework",
    ],
)

cc_library(
    name = "catboost_bundle",
    srcs = ["catboost_bundle.cc"],
    hdrs = ["catboost_bundle.h", "catboost_constants.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "@catboost//:catboost",
        "//tensorflow_serving/util:file_probing_env",
    ],
)

cc_library(
    name = "catboost_source_adapter",
    srcs = ["catboost_source_adapter.cc"],
    hdrs = ["catboost_source_adapter.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":catboost_bundle",
        ":catboost_source_adapter_proto",
        "//tensorflow_serving/core:simple_loader",
        "//tensorflow_serving/core:source_adapter",
        "//tensorflow_serving/core:storage_path",
    ],
    alwayslink = 1,
)

cc_library(
    name = "predict_impl",
    srcs = ["predict_impl.cc"],
    hdrs = ["predict_impl.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":catboost_bundle",
        "//tensorflow_serving/apis:predict_proto",
        "//tensorflow_serving/servables/catboost:util",
        "//tensorflow_serving/model_servers:server_core",
        "@brpc//:bvar",
    ]
)

cc_test(
    name = "catboost_source_adapter_test",
    size = "medium",
    srcs = ["catboost_source_adapter_test.cc"],
    deps = [
        ":catboost_source_adapter",
        "//tensorflow_serving/core/test_util:test_main",
        "@org_tensorflow//tensorflow/core:test",
        "//tensorflow_serving/sources/storage_path:file_system_storage_path_source",
        "//tensorflow_serving/core:aspired_versions_manager",
        "//tensorflow_serving/core:availability_preserving_policy",
    ],
)

cc_test(
    name = "catboost_bundle_test",
    size = "medium",
    srcs = ["catboost_bundle_test.cc"],
    data = [
        "//tensorflow_serving/servables/catboost/testdata:test_model/1/catboost.cbm",
    ],
    deps = [
        ":catboost_bundle",
        "//tensorflow_serving/core/test_util:test_main",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_test(
    name = "predict_impl_test",
    size = "medium",
    srcs = ["predict_impl_test.cc"],
    data = [
        "//tensorflow_serving/servables/catboost/testdata:test_model/1/catboost.cbm",
    ],
    deps = [
        ":predict_impl",
        "//tensorflow_serving/core/test_util:test_main",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)
