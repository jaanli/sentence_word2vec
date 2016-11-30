# Description:
# TensorFlow model for sentence word2vec

package(default_visibility = ["//tensorflow:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")

py_library(
    name = "package",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        ":gen_sentence_word2vec",
        ":word2vec",
        ":word2vec_optimized",
    ],
)

py_binary(
    name = "word2vec",
    srcs = [
        "word2vec.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":gen_sentence_word2vec",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:platform",
	"//tensorflow/models/embedding:gen_word2vec",
    ],
)

py_binary(
    name = "word2vec_optimized",
    srcs = [
        "word2vec_optimized.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":gen_sentence_word2vec",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:platform",
	"//tensorflow/models/embedding:gen_word2vec",
    ],
)

py_test(
    name = "word2vec_test",
    size = "small",
    srcs = ["word2vec_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "notsan",  # b/25864127
    ],
    deps = [
        ":word2vec",
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "word2vec_optimized_test",
    size = "small",
    srcs = ["word2vec_optimized_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "notsan",
    ],
    deps = [
        ":word2vec_optimized",
        "//tensorflow:tensorflow_py",
    ],
)

cc_library(
    name = "sentence_word2vec_ops",
    srcs = [
        "sentence_word2vec_ops.cc",
    ],
    linkstatic = 1,
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/core:framework",
    ],
    alwayslink = 1,
)

cc_library(
    name = "sentence_word2vec_kernels",
    srcs = [
        "sentence_word2vec_kernels.cc",
    ],
    linkstatic = 1,
    visibility = ["//tensorflow:internal"],
    deps = [
        ":sentence_word2vec_ops",
	"//tensorflow/models/embedding:word2vec_ops",
        "//tensorflow/core",
    ],
    alwayslink = 1,
)

tf_gen_op_wrapper_py(
    name = "gen_sentence_word2vec",
    out = "gen_sentence_word2vec.py",
    deps = [":sentence_word2vec_ops"],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
