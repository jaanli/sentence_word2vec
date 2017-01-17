# sentence_word2vec
word2vec with a context based on sentences, in C++.

This is based on the tensorflow implementation of word2vec.

However, the context for the model is defined differently:

* the context for the model is defined in terms of sentences.
* the context for a given word is the rest of words in a sentence.

This is implemented in C++ in the `sentence_word2vec_kernels.cc` file.

Why might this be useful? This can be used to model playlists or
user histories for recommendation! Or any other kind of 'bagged' data.

## Usage

To compile the C++ ops used:
```
git clone https://github.com/altosaar/sentence_word2vec
cd sentence_word2vec
./compile_ops.sh
```

To get the text8 data and split it into sentences for testing:
```
./get_data.sh
```

To run the code with a sentence-level context window:
```
python word2vec_optimized.py -- \
    --train_data text8_split \
    --eval_data questions-words.txt \
    --save_path /tmp \
    --sentence_level True
```

On a Macbook Air with the following config, the speed is around 17k words/second. This is up from around 2k words/second with a [manual python implementation](https://github.com/altosaar/scirec).
```
➜  ~ sysctl -n machdep.cpu.brand_string
Intel(R) Core(TM) i7-4650U CPU @ 1.70GHz
```

This directory contains models for unsupervised training of word embeddings
using the model described in:
(Mikolov, et. al.) [Efficient Estimation of Word Representations in Vector Space](http://arxiv.org/abs/1301.3781),
ICLR 2013.

Detailed instructions and description of this model is available in the
tensorflow tutorials:

* [Word2Vec Tutorial](http://tensorflow.org/tutorials/word2vec/index.md)

File | What's in it?
--- | ---
`word2vec.py` | A version of word2vec implemented using TensorFlow ops and minibatching.
`word2vec_optimized.py` | A version of word2vec implemented using C ops that does no minibatching.
`sentence_word2vec_kernels.cc` | Kernels for the custom input and training ops, including sentence-level contexts.
`sentence_word2vec_ops.cc` | The declarations of the custom ops.
