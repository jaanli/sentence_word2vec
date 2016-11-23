# sentence_word2vec
word2vec with a context based on sentences, in C++.

This is based on the tensorflow implementation of word2vec.

However, the context for the model is defined differently:

* the context for the model is defined in terms of sentences.
* the context for a given word is the rest of words in a sentence.

This is implemented in C++ in the `word2vec_kernels.cc` file.

Why might this be useful? This can be used to model playlists or
user histories for recommendation! Or any other kind of 'bagged' data.

To split the `text8` dataset (described / available below) into sentences:
```
cp text8 text8_split
# add newlines after every 'the' just for debugging
perl -i -pe 's/the/\nthe/g' text8_split
# remove lines that have a single word, because they will have no context
# may need to install gawk
gawk -iinplace 'NF>=2' text8_split
```

Because we need bazel to compile the C++ ops, this code needs to be in the main tensorflow repo in the models directory.
```
git clone https://github.com/tensorflow/tensorflow
cd tensorflow
git checkout v0.11.0
./configure  # answer the prompts
git clone https://github.com/altosaar/sentence_word2vec tensorflow/models/sentence_word2vec
```

Then, to run the code with a sentence-level context window:
```
# needs to be run from the tensorflow directory (where bazel WORKSPACE file is)
bazel run -c opt tensorflow/models/sentence_word2vec/word2vec_optimized -- \
    --train_data /path/to/text8_split \
    --eval_data /path/to/questions-words.txt \
    --save_path /tmp \
    --sentence_level True
```

On a Macbook Air with the following config, the speed is around 17k words/second. This is up from around 2k words/second with a manual python implementation.
```
➜  ~ sysctl -n machdep.cpu.brand_string
Intel(R) Core(TM) i7-4650U CPU @ 1.70GHz
```

Original word2vec code from TensorFlow v0.11.0 ([link to source at this commit](https://github.com/tensorflow/tensorflow/tree/v0.11.0/tensorflow/models/embedding)).

This directory contains models for unsupervised training of word embeddings
using the model described in:
(Mikolov, et. al.) [Efficient Estimation of Word Representations in Vector Space](http://arxiv.org/abs/1301.3781),
ICLR 2013.

Detailed instructions on how to get started and use them are available in the
tensorflow tutorials. Brief instructions are below.

* [Word2Vec Tutorial](http://tensorflow.org/tutorials/word2vec/index.md)

To download the example text and evaluation data:

```shell
wget http://mattmahoney.net/dc/text8.zip -O text8.zip
unzip text8.zip
wget https://storage.googleapis.com/google-code-archive-source/v2/code.google.com/word2vec/source-archive.zip
unzip -p source-archive.zip  word2vec/trunk/questions-words.txt > questions-words.txt
rm source-archive.zip
```
Here is a short overview of what is in this directory.

File | What's in it?
--- | ---
`word2vec.py` | A version of word2vec implemented using TensorFlow ops and minibatching.
`word2vec_test.py` | Integration test for word2vec.
`word2vec_optimized.py` | A version of word2vec implemented using C ops that does no minibatching.
`word2vec_optimized_test.py` | Integration test for word2vec_optimized.
`word2vec_kernels.cc` | Kernels for the custom input and training ops, including sentence-level contexts.
`word2vec_ops.cc` | The declarations of the custom ops.
