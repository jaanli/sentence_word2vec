#!/bin/bash

# compile ops, as specified in https://github.com/tensorflow/models/tree/master/tutorials/embedding
# and https://www.tensorflow.org/how_tos/adding_an_op/
# NB: '-undefined dynamic_lookup' is needed on mac


# get tensorflow includes
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')

# compile the word2vec ops
g++ -v -std=c++11 -shared models/tutorials/embedding/word2vec_ops.cc models/tutorials/embedding/word2vec_kernels.cc -o word2vec_ops.so -fPIC -I $TF_INC -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -undefined dynamic_lookup

# compile the sentence_word2vec op
g++ -v -std=c++11 -shared sentence_word2vec_ops.cc sentence_word2vec_kernels.cc -o sentence_word2vec_ops.so -fPIC -I $TF_INC -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -undefined dynamic_lookup
